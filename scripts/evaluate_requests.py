"""Evaluate a frozen JSONL request/expected corpus against one resident model process."""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def evaluate(binary, model_args, cases, timeout=600):
    if not cases or len({c['request']['id'] for c in cases}) != len(cases):
        raise ValueError('nonempty cases and unique request IDs are required')
    run = subprocess.run([str(binary), *model_args], input=''.join(json.dumps(c['request'], ensure_ascii=False)+'\n' for c in cases), text=True, capture_output=True, timeout=timeout, check=True)
    responses = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    if len(responses) != len(cases):
        raise ValueError('response count differs from case count')
    rows = []
    for case, response in zip(cases, responses):
        request = case['request']
        if response.get('id') != request['id']:
            raise ValueError('response identity mismatch')
        if 'error' in response:
            rows.append({'id': request['id'], 'correct': False, 'error': response['error']})
            continue
        decisions = {d['id']: d for d in response['decisions']}
        if set(decisions) != set(case['expected']):
            raise ValueError('expected question IDs differ from decisions')
        for qid, gold in case['expected'].items():
            decision = decisions[qid]
            rows.append({'id': request['id'], 'question_id': qid, 'expected': gold, 'predicted': decision['value'], 'correct': decision['value'] == gold, 'decision': decision['decision'], 'probabilities': decision['probabilities'], 'truncated': decision['truncated'], 'elapsed_ms': response['elapsed_ms'], 'input_tokens': response['input_tokens'], 'model': response['model']})
    latencies = [r['elapsed_ms'] for r in rows if 'elapsed_ms' in r]
    return {'purpose': 'Frozen corpus engineering evaluation; requires domain validation', 'correct': sum(r['correct'] for r in rows), 'total': len(rows), 'false_supports': sum(r.get('predicted') == 'supports' and r.get('expected') != 'supports' for r in rows), 'errors': sum('error' in r for r in rows), 'median_ms': statistics.median(latencies) if latencies else None, 'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=Path('bin/layago'))
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    model = p.add_mutually_exclusive_group(required=True)
    model.add_argument('--checkpoint', type=Path)
    model.add_argument('--artifact', type=Path)
    p.add_argument('--base', type=Path)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.artifact and not args.base:
        p.error('--artifact requires --base')
    model_args = ['--workers', str(args.workers)]
    model_args += ['--checkpoint', str(args.checkpoint)] if args.checkpoint else ['--artifact', str(args.artifact), '--base', str(args.base)]
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    report = evaluate(args.binary.resolve(), model_args, cases)
    report['cases_sha256'] = hashlib.sha256(raw).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
