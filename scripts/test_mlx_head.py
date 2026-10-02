import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx_head


class MLXCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                raise unittest.SkipTest('Actual Metal GPU required')
            cls.mx=mx
        except ImportError:
            raise unittest.SkipTest('Optional MLX not installed')

    def fixture(self, path):
        mx=self.mx; path.mkdir()
        cfg=dict(max_epochs=4,patience=4,batch_size=3,learning_rate=.001,
                 warmup_fraction=0.1,weight_decay=.01,score_regression_weight=.5)
        request=dict(contract='ul.layago-mlx-fit/v1',config=cfg,
                     permutations=[list(range(9)) for _ in range(4)],memory_limit=128*2**20)
        (path/'request.json').write_text(json.dumps(request))
        mx.random.seed(13)
        params={'type_emb.weight':mx.zeros((3,4)), 'scorer.0.weight':mx.ones((4,)),
                'scorer.0.bias':mx.zeros((4,)), 'scorer.1.weight':mx.random.normal((4,4))*.2,
                'scorer.1.bias':mx.zeros((4,)), 'scorer.3.weight':mx.random.normal((1,4))*.2,
                'scorer.3.bias':mx.zeros((1,))}
        for split,n in [('train',9),('dev',6)]:
            labels=mx.arange(n)%3
            params.update({split+'.markers':mx.random.normal((n,3,4)),
                           split+'.mask':mx.ones((n,3)), split+'.target':mx.eye(3)[labels],
                           split+'.meta':mx.stack([mx.zeros((n,)),labels,mx.ones((n,)),mx.full((n,),3),mx.zeros((n,))],axis=1)})
        mx.save_safetensors(str(path/'input.safetensors'),params)

    def test_interrupted_epoch_optimizer_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as root:
            full,resume=Path(root)/'full',Path(root)/'resume'
            self.fixture(full);self.fixture(resume)
            with contextlib.redirect_stderr(io.StringIO()):
                mlx_head.fit(full)
                original=mlx_head.atomic_json
                def crash_before_second_commit(path,state):
                    if path.name=='checkpoint.json' and len(state['history'])==2:
                        raise RuntimeError('simulated death between tensor writes and state commit')
                    original(path,state)
                with patch.object(mlx_head,'atomic_json',crash_before_second_commit):
                    with self.assertRaises(RuntimeError):mlx_head.fit(resume)
                mlx_head.fit(resume)
            result=json.loads((resume/'result.json').read_text())
            self.assertEqual(result['info']['resumed_from_epoch'],1)
            a,b=self.mx.load(str(full/'best.safetensors')),self.mx.load(str(resume/'best.safetensors'))
            for name in a:self.assertTrue(bool(self.mx.allclose(a[name],b[name],atol=1e-7,rtol=1e-6).item()),name)
            for a,b in zip(json.loads((full/'result.json').read_text())['history'],result['history']):
                self.assertAlmostEqual(a['dev_cal_nll'],b['dev_cal_nll'],places=6)

    def test_corrupt_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'run';self.fixture(path)
            with contextlib.redirect_stderr(io.StringIO()):mlx_head.fit(path)
            state=json.loads((path/'checkpoint.json').read_text())
            (path/state['weights_file']).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):mlx_head.fit(path)


if __name__=='__main__':unittest.main()
