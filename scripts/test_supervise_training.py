import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from supervise_training import supervise


@unittest.skipUnless(os.name == 'posix','Process groups require POSIX')
class SupervisorTests(unittest.TestCase):
    def run_case(self, code, **kwargs):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        run = Path(root.name)/'run'
        state = supervise([sys.executable,'-u','-c',code],run,interval=.04,
                          grace=.05,deadline=3,stall=1,min_free=0,
                          sampler=lambda _pid:(1000,90),**kwargs)
        self.assertTrue((run/'health-report.html').exists())
        self.assertEqual(json.loads((run/'status.json').read_text())['outcome'],state['outcome'])
        return state, run

    def test_progress_and_success(self):
        state,run = self.run_case("import os,json,time; from pathlib import Path; p=Path(os.environ['LAYAGO_PROGRESS_FILE']); p.write_text(json.dumps({'sequence':1,'stage':'scorer','step':1})); print('loss=0.3'); time.sleep(.2)")
        self.assertEqual(state['outcome'],'success')
        self.assertEqual(state['progress_sequence'],1)
        self.assertIn('loss=0.3',(run/'stdout.log').read_text())

    def test_heartbeat_does_not_mask_stall(self):
        # Short polling keeps the test fast; only progress can reset the timer.
        state,_ = self.run_case("import time;\nwhile True: print('health alive'); time.sleep(.02)")
        self.assertEqual(state['outcome'],'stalled')

    def test_sigkill_not_invented_oom(self):
        state,_ = self.run_case('import os,signal; os.kill(os.getpid(),signal.SIGKILL)')
        self.assertEqual(state['outcome'],'child_failed')
        self.assertEqual(state['child_signal'],signal.SIGKILL)

    def test_rss_guard_and_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
        try:
            root = tempfile.TemporaryDirectory(); self.addCleanup(root.cleanup)
            state = supervise([sys.executable,'-c','import time; time.sleep(20)'],Path(root.name)/'run',
                              interval=.03,grace=.05,deadline=3,stall=2,rss_limit=100,
                              min_free=0,sampler=lambda _pid:(101,90))
            self.assertEqual(state['outcome'],'rss_limit')
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate(); unrelated.wait()

    def test_pressure_and_monitor_failures(self):
        for sampler,expected in [(lambda _p:(100,1),'host_memory_pressure'),
                                 (lambda _p: (_ for _ in ()).throw(OSError('probe failed')),'monitor_failed')]:
            root=tempfile.TemporaryDirectory(); self.addCleanup(root.cleanup)
            state=supervise([sys.executable,'-c','import time; time.sleep(20)'],Path(root.name)/'run',
                            interval=.03,grace=.05,deadline=3,stall=2,min_free=5,sampler=sampler)
            self.assertEqual(state['outcome'],expected)

    def test_deadline_despite_progress(self):
        root=tempfile.TemporaryDirectory(); self.addCleanup(root.cleanup)
        code="import os,json,time; from pathlib import Path; p=Path(os.environ['LAYAGO_PROGRESS_FILE']); i=0\nwhile True:\n i+=1; p.write_text(json.dumps({'sequence':i})); time.sleep(.02)"
        state=supervise([sys.executable,'-u','-c',code],Path(root.name)/'run',
                        interval=.03,grace=.05,deadline=.3,stall=2,min_free=0,sampler=lambda _p:(100,90))
        self.assertEqual(state['outcome'],'deadline')

    def test_logs_bounded_and_existing_output_refused(self):
        state,run=self.run_case("print('x'*100000)",log_limit=1024)
        self.assertEqual(state['outcome'],'success')
        self.assertLessEqual((run/'stdout.log').stat().st_size,1024)
        self.assertGreater(state['discarded_log_bytes']['stdout'],0)
        with self.assertRaises(FileExistsError):
            supervise([sys.executable,'-c','pass'],run)

    def test_crashed_leader_descendants_cleaned(self):
        # A descendant's signal handler records termination; it would survive
        # a leader-only kill, which was a failure mode we need to prevent.
        root=tempfile.TemporaryDirectory(); self.addCleanup(root.cleanup)
        marker=Path(root.name)/'child-stopped'
        child=f"import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,lambda *_:(Path({str(marker)!r}).write_text('stopped'),exit(0))); time.sleep(20)"
        leader=f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(.2); sys.exit(3)"
        state=supervise([sys.executable,'-c',leader],Path(root.name)/'run',
                        interval=.03,grace=.1,deadline=3,stall=2,min_free=0,sampler=lambda _p:(100,90))
        self.assertEqual(state['outcome'],'child_failed')
        self.assertTrue(marker.exists())


if __name__=='__main__': unittest.main()
