#!/usr/bin/env python3
"""Optional Apple-GPU cached-marker scorer; Go retains act/calibration/export.

Float32 GPU arithmetic differs from Go's float64 accumulators. Forward/gradient
parity and Go export NLL checks bound that difference; bit equality is not claimed.
No encoder conversion, full Laya RLCD or CPU fallback is implied.
"""
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

PARAMS = ('type_emb.weight', 'scorer.0.weight', 'scorer.0.bias',
          'scorer.1.weight', 'scorer.1.bias', 'scorer.3.weight', 'scorer.3.bias')


def mlx(memory_limit):
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError('Apple Metal GPU is required; use --backend go on other hosts')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(min(memory_limit // 4, 128 * 2**20))
    return mx


def scores(mx, p, markers, qtypes):
    z = markers + p['type_emb.weight'][qtypes][:, None, :]
    mean = mx.mean(z, axis=-1, keepdims=True)
    x = (z-mean) * mx.rsqrt(mx.mean((z-mean)**2, axis=-1, keepdims=True)+1e-5)
    x = x*p['scorer.0.weight']+p['scorer.0.bias']
    x = x @ p['scorer.1.weight'].T + p['scorer.1.bias']
    x = 0.5*x*(1+mx.erf(x/math.sqrt(2)))
    return (x @ p['scorer.3.weight'].T)[..., 0] + p['scorer.3.bias'][0]


def losses(mx, p, batch, reg_weight):
    markers, targets, mask, meta = batch
    logits = scores(mx, p, markers, meta[:, 0].astype(mx.int32))
    logits = mx.where(mask > 0, logits, -1e9)
    logp = logits-mx.logsumexp(logits, axis=1, keepdims=True)
    ce = -mx.sum(targets*logp, axis=1)
    levels = mx.arange(logits.shape[1], dtype=mx.float32)
    expected = mx.sum(mx.exp(logp)*levels, axis=1)
    gold = mx.sum(targets*levels, axis=1)
    reg = reg_weight*((expected-gold)/mx.maximum(meta[:, 3]-1, 1))**2*meta[:, 4]
    return (ce+reg)*meta[:, 2]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def fit(directory):
    directory = Path(directory)
    request = json.loads((directory/'request.json').read_text())
    if request['contract'] != 'ul.layago-mlx-fit/v1':
        raise ValueError('Unsupported fit contract')
    cfg = request['config']
    mx = mlx(request['memory_limit'])
    arrays = mx.load(str(directory/'input.safetensors'))
    p = {name: arrays.pop(name) for name in PARAMS}
    train = tuple(arrays['train.'+name] for name in ('markers','target','mask','meta'))
    dev = tuple(arrays['dev.'+name] for name in ('markers','target','mask','meta'))
    mx.eval(p, train, dev)
    n, batch_size = train[0].shape[0], cfg['batch_size']
    steps = (n+batch_size-1)//batch_size
    total = steps*(cfg.get('schedule_epochs') or cfg['max_epochs'])
    warmup = int(cfg['warmup_fraction']*total)
    gradient = mx.value_and_grad(lambda parameters, batch: mx.mean(losses(mx,parameters,batch,cfg['score_regression_weight'])))
    moments = {name: mx.zeros_like(value) for name,value in p.items()}
    variance = {name: mx.zeros_like(value) for name,value in p.items()}
    best, bad, step, best_epoch = math.inf, 0, 0, 0
    history = []
    best_file = None
    # Memory budgets may change on resume; fitting data, optimizer config and
    # implementation must not. The encoder/base identity is in the cached tensors.
    binding = hashlib.sha256(json.dumps(dict(config=cfg,permutations=request['permutations'],mlx_version=mx.__version__,python_version=list(sys.version_info[:2])),sort_keys=True).encode()
                             + (directory/'input.safetensors').read_bytes()
                             + Path(__file__).read_bytes()).hexdigest()
    resume_epoch = 0
    state_path = directory/'checkpoint.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['binding'] != binding:
            raise ValueError('Resume data/config/code differs; use a new MLX workspace')
        file = directory/state['weights_file']
        if hashlib.sha256(file.read_bytes()).hexdigest() != state['weights_sha256']:
            raise ValueError('Optimizer checkpoint hash mismatch')
        best_file = state['best_file']
        if hashlib.sha256((directory/best_file).read_bytes()).hexdigest() != state['best_sha256']:
            raise ValueError('Best scorer checkpoint hash mismatch')
        saved = mx.load(str(file))
        p = {name:saved['param.'+name] for name in PARAMS}
        moments = {name:saved['moment.'+name] for name in PARAMS}
        variance = {name:saved['variance.'+name] for name in PARAMS}
        best,bad,step,best_epoch,history = (state[k] for k in ('best','bad','step','best_epoch','history'))
        resume_epoch = len(history)
        mx.eval(p,moments,variance)
        print(json.dumps(dict(kind='resume',epoch=resume_epoch,step=step)),file=sys.stderr,flush=True)
    start = time.monotonic()
    for epoch, permutation in enumerate(request['permutations'], 1):
        if epoch <= resume_epoch:
            continue
        if bad >= cfg['patience']:
            break
        if sorted(permutation) != list(range(n)):
            raise ValueError('Invalid deterministic shuffle')
        t0, summed = time.monotonic(), 0.0
        for offset in range(0,n,batch_size):
            indices = mx.array(permutation[offset:offset+batch_size],dtype=mx.int32)
            batch = tuple(value[indices] for value in train)
            loss, grads = gradient(p,batch)
            step += 1
            if warmup and step <= warmup:
                lr = cfg['learning_rate']*step/warmup
            else:
                fraction = min(1,(step-warmup)/max(1,total-warmup))
                lr = cfg['learning_rate']*(0.1+0.9*0.5*(1+math.cos(math.pi*fraction)))
            updated = {}
            for name, value in p.items():
                moments[name] = 0.9*moments[name]+0.1*grads[name]
                variance[name] = 0.999*variance[name]+0.001*grads[name]**2
                update = (moments[name]/(1-0.9**step))/(mx.sqrt(variance[name]/(1-0.999**step))+1e-8)
                if name in ('scorer.1.weight','scorer.3.weight'):
                    value = value*(1-lr*cfg['weight_decay'])
                updated[name] = value-lr*update
            p = updated
            mx.eval(loss,p,moments,variance)
            value = float(loss.item())
            if not math.isfinite(value) or not all(bool(mx.all(mx.isfinite(v)).item()) for v in p.values()):
                raise ValueError('Non-finite loss or parameters')
            summed += value*len(permutation[offset:offset+batch_size])
            active = mx.get_active_memory()
            if active > request['memory_limit']:
                raise MemoryError(f'MLX active allocation {active} exceeds budget')
            print(json.dumps(dict(kind='progress',step=step,epoch=epoch,loss=value,
                                  active_bytes=active,peak_bytes=mx.get_peak_memory())),file=sys.stderr,flush=True)
        # Chunk development evaluation too; training bounds must also cover validation.
        nll_sum, correct, dev_n = 0.0, 0, dev[0].shape[0]
        for offset in range(0,dev_n,batch_size):
            markers,_,mask,meta = tuple(v[offset:offset+batch_size] for v in dev)
            logits = scores(mx,p,markers,meta[:,0].astype(mx.int32))
            logits = mx.where(mask>0,logits,-1e9)
            labels = meta[:,1].astype(mx.int32)
            nll = mx.logsumexp(logits,axis=1)-mx.take_along_axis(logits,labels[:,None],axis=1)[:,0]
            mx.eval(nll,logits)
            nll_sum += float(mx.sum(nll).item())
            correct += int(mx.sum(mx.argmax(logits,axis=1)==labels).item())
            print(json.dumps(dict(kind='progress',step=step+offset+1,epoch=epoch,phase='dev')),file=sys.stderr,flush=True)
        nll = nll_sum/dev_n
        if not math.isfinite(nll):
            raise ValueError('Non-finite development NLL')
        secs = time.monotonic()-t0
        history.append(dict(epoch=epoch,train_loss=summed/n,dev_cal_nll=nll,
                            dev_cal_accuracy=correct/dev_n,learning_rate_at_end=lr,
                            seconds=secs,examples_per_second=n/max(secs,1e-9)))
        print(json.dumps(dict(kind='epoch',**history[-1])),file=sys.stderr,flush=True)
        if nll < best-1e-9:
            best, bad, best_epoch = nll, 0, epoch
            # Durable best scorer is replaced atomically; calibration is fitted later in Go.
            best_file = f'best-epoch-{epoch:05d}.safetensors'
            mx.save_safetensors(str(directory/(best_file+'.tmp.safetensors')),p)
            os.replace(directory/(best_file+'.tmp.safetensors'),directory/best_file)
            with (directory/best_file).open('rb') as f: os.fsync(f.fileno())
        else:
            bad += 1
        saved = {**{'param.'+name:v for name,v in p.items()},
                 **{'moment.'+name:v for name,v in moments.items()},
                 **{'variance.'+name:v for name,v in variance.items()}}
        weights_name = f'optimizer-epoch-{epoch:05d}.safetensors'
        mx.save_safetensors(str(directory/(weights_name+'.tmp.safetensors')),saved)
        os.replace(directory/(weights_name+'.tmp.safetensors'),directory/weights_name)
        with (directory/weights_name).open('rb') as f: os.fsync(f.fileno())
        checkpoint = dict(binding=binding,weights_file=weights_name,best_file=best_file,
                          weights_sha256=hashlib.sha256((directory/weights_name).read_bytes()).hexdigest(),
                          best_sha256=hashlib.sha256((directory/best_file).read_bytes()).hexdigest(),
                          best=best,bad=bad,step=step,best_epoch=best_epoch,history=history)
        atomic_json(state_path,checkpoint)
        for pattern, retained in [('optimizer-epoch-*.safetensors',weights_name),('best-epoch-*.safetensors',best_file)]:
            for old in directory.glob(pattern):
                if old.name != retained:
                    old.unlink()
        print(json.dumps(dict(kind='checkpoint',epoch=epoch,step=step)),file=sys.stderr,flush=True)
        mx.clear_cache()
    shutil.copyfile(directory/best_file,directory/'best.output.tmp')
    os.replace(directory/'best.output.tmp',directory/'best.safetensors')
    info = dict(resumed_from_epoch=resume_epoch,checkpoint_binding=binding,backend='mlx',mlx_version=mx.__version__,device=str(mx.default_device()),
                precision='float32',encoder_backend='go_frozen_cache',act_calibration_backend='go',
                memory_limit=request['memory_limit'],peak_mlx_bytes=mx.get_peak_memory(),
                scorer_seconds=time.monotonic()-start,optimizer='AdamW float32 GPU moments',
                limitation='Go uses float64 accumulators/master moments; trajectory is not bit-identical')
    atomic_json(directory/'result.json',dict(history=history,best_epoch=best_epoch,info=info))


def probe(input_path, output_path):
    spec = json.loads(Path(input_path).read_text())
    mx = mlx(256*2**20)
    p = {name: mx.array(spec['params'][name]['data'],dtype=mx.float32).reshape(spec['params'][name]['shape']) for name in PARAMS}
    case = spec['sample']
    batch = (mx.array(case['markers'],dtype=mx.float32).reshape(1,case['k'],-1),
             mx.array([case['target']],dtype=mx.float32),mx.ones((1,case['k'])),
             mx.array([[case['qtype'],case['label'],case['weight'] or 1,case['k'],1 if case['type']=='score' else 0]],dtype=mx.float32))
    loss, grads = mx.value_and_grad(lambda parameters: mx.mean(losses(mx,parameters,batch,spec['reg_weight'])))(p)
    logits = scores(mx,p,batch[0],batch[3][:,0].astype(mx.int32))
    mx.eval(loss,grads,logits)
    Path(output_path).write_text(json.dumps(dict(logits=logits[0].tolist(),loss=loss.item(),
                                                gradients={name:g.flatten().tolist() for name,g in grads.items()})))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-dir')
    ap.add_argument('--probe')
    ap.add_argument('--output')
    args = ap.parse_args()
    try:
        if args.fit_dir:
            fit(args.fit_dir)
        elif args.probe and args.output:
            probe(args.probe,args.output)
        else:
            ap.error('Use --fit-dir or --probe INPUT --output OUTPUT')
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
