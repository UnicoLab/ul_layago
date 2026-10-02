#!/usr/bin/env python3
"""Bound a training process group; persist telemetry even when its child dies.

Only the command launched in this run is signalled. Progress is supplied through
LAYAGO_PROGRESS_FILE; ordinary stdout/stderr and health messages do not reset
stall detection. An observed SIGKILL is not silently classified as an OOM.
"""
import argparse
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resources(pgid):
    """RSS of this process group, including the Go trainer's MLX worker."""
    raw = subprocess.run(['ps', '-axo', 'pgid=,rss='], capture_output=True,
                         text=True, timeout=2, check=True).stdout
    rss = sum(int(row[1]) * 1024 for line in raw.splitlines()
              if len(row := line.split()) == 2 and int(row[0]) == pgid)
    free = None
    if sys.platform.startswith('linux'):
        mem = dict((line.split(':')[0], int(line.split()[1]))
                   for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith(('MemAvailable:', 'MemTotal:')))
        free = 100 * mem['MemAvailable'] / mem['MemTotal']
    elif sys.platform == 'darwin':
        # Kernel pressure estimate, not an inference from our own process RSS.
        out = subprocess.run(['memory_pressure', '-Q'], capture_output=True,
                             text=True, timeout=2, check=True).stdout
        for line in out.splitlines():
            if 'System-wide memory free percentage:' in line:
                free = float(line.rsplit(':', 1)[1].strip().rstrip('%'))
    return rss, free


def stop_group(proc, grace):
    # start_new_session assigns the child PID to its new process-group ID.
    # This also covers descendants after the immediate child has exited.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    alive = True
    while time.monotonic() < deadline:
        proc.poll()
        # macOS may return EPERM for a signal-0 probe of a disappearing group.
        # Use the process table instead; do not confuse that with a live child.
        raw = subprocess.run(['ps', '-axo', 'pgid=,stat='], capture_output=True,
                             text=True, timeout=2, check=True).stdout
        alive = any(len(row := line.split()) == 2 and int(row[0]) == proc.pid
                    and not row[1].startswith('Z') for line in raw.splitlines())
        if not alive:
            break
        time.sleep(0.02)
    if alive:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=max(2, grace))


def make_report(run, state, samples):
    esc = lambda value: html.escape(str(value))
    peak = state.get('peak_rss_bytes',0) / 2**20
    points = ' '.join(f'{30 + i * 740/max(1,len(samples)-1):.1f},{210-170*s.get("rss_bytes",0)/max(1,peak*2**20):.1f}'
                      for i, s in enumerate(samples))
    rows = ''.join(f'<tr><td>{esc(k.replace("_"," "))}</td><td>{esc(v)}</td></tr>'
                   for k, v in state.items() if k != 'command')
    report = f'''<!doctype html><meta charset="utf-8"><title>LayaGo run health</title>
<style>body{{font:16px system-ui;max-width:960px;margin:40px auto;padding:20px;color:#22333a;background:#f5f8fa}}section{{background:white;padding:24px;border-radius:16px;margin:20px 0}}td{{padding:8px;vertical-align:top}}h1{{margin:0}}svg{{width:100%;height:auto}}pre{{white-space:pre-wrap}}</style>
<h1>Training run health</h1><p>{esc(state['outcome'])} · peak sampled process-group RSS {peak:.1f} MiB</p>
<section><h2>Memory over time</h2><svg viewBox="0 0 800 240" role="img" aria-label="Sampled process-group memory"><polyline points="{points}" stroke="#247a7c" stroke-width="3" fill="none"/></svg><p>Sampled RSS; brief allocation spikes may be missed. MLX allocation telemetry is recorded separately by the worker.</p></section>
<section><table>{rows}</table></section><section><h2>Command</h2><pre>{esc(state['command'])}</pre><p>stdout.log, stderr.log, telemetry.jsonl, progress.json and status.json are retained beside this report. A child SIGKILL alone does not prove an OOM.</p></section>'''
    (run / 'health-report.html').write_text(report, encoding='utf-8')


def supervise(command, run, *, rss_limit=6 * 2**30, min_free=5, deadline=1800,
              stall=120, interval=1, grace=5, log_limit=8 * 2**20, sampler=resources):
    if os.name != 'posix':
        raise ValueError('Process-group supervision currently requires macOS or Linux')
    if not command or any(not math.isfinite(v) or v <= 0 for v in
                          (rss_limit, deadline, stall, interval, grace, log_limit)) or not 0 <= min_free < 100:
        raise ValueError('Invalid supervisor bounds')
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress_file = run / 'progress.json'
    env = dict(os.environ, LAYAGO_PROGRESS_FILE=str(progress_file))
    state = dict(contract='ul.layago-supervision/v1', command=command, started_at=utc(),
                 outcome='starting', progress_sequence=0, peak_rss_bytes=0,
                 rss_limit_bytes=rss_limit, min_host_free_percent=min_free,
                 max_seconds=deadline, stall_seconds=stall)
    atomic_json(run / 'status.json', state)
    proc, selector = None, selectors.DefaultSelector()
    streams, samples = {}, []
    old_handlers = {}
    requested_signal = []
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, start_new_session=True)
        state.update(pid=proc.pid, outcome='running')
        for name, pipe in [('stdout', proc.stdout), ('stderr', proc.stderr)]:
            os.set_blocking(pipe.fileno(), False)
            sink = (run / f'{name}.log').open('wb')
            streams[name] = [sink, 0, 0]
            selector.register(pipe, selectors.EVENT_READ, name)
        if __import__('threading').current_thread() is __import__('threading').main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda s, _f: requested_signal.append(s))
        last_progress, next_sample, sequence = started, started, 0
        with (run / 'telemetry.jsonl').open('a', encoding='utf-8') as journal:
            while True:
                now = time.monotonic()
                for key, _ in selector.select(timeout=min(interval, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    sink, used, discarded = streams[key.data]
                    kept = chunk[:max(0, log_limit-used)]
                    sink.write(kept)
                    sink.flush()
                    streams[key.data][1] += len(kept)
                    streams[key.data][2] += len(chunk)-len(kept)
                try:
                    progress = json.loads(progress_file.read_text())
                    newseq = progress['sequence']
                    if type(newseq) is int and newseq > sequence:
                        sequence, last_progress = newseq, now
                        state.update(progress_sequence=sequence, stage=progress.get('stage'),
                                     step=progress.get('step'))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                code = proc.poll()
                if requested_signal:
                    state.update(outcome='interrupted', supervisor_signal=requested_signal[0])
                    break
                if code is not None:
                    state.update(outcome='success' if code == 0 else 'child_failed', returncode=code,
                                 child_signal=-code if code < 0 else None)
                    break
                if now >= next_sample:
                    try:
                        rss, free = sampler(proc.pid)
                        sample = dict(at=utc(), elapsed_seconds=now-started, rss_bytes=rss,
                                      host_free_percent=free, progress_sequence=sequence,
                                      seconds_since_progress=now-last_progress)
                        samples.append(sample)
                        if len(samples) > 4096:
                            samples[:] = samples[::2]
                        journal.write(json.dumps(sample, allow_nan=False)+'\n')
                        journal.flush()
                        os.fsync(journal.fileno())
                        state['peak_rss_bytes'] = max(state['peak_rss_bytes'], rss)
                        if rss > rss_limit:
                            state['outcome'] = 'rss_limit'
                        elif free is not None and free < min_free:
                            state['outcome'] = 'host_memory_pressure'
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        # Fail closed instead of silently running without the sentinel.
                        state.update(outcome='monitor_failed', monitor_error=str(exc))
                    next_sample = now + interval
                    atomic_json(run / 'status.json', state)
                if now-started >= deadline:
                    state['outcome'] = 'deadline'
                elif now-last_progress >= stall:
                    state['outcome'] = 'stalled'
                if state['outcome'] != 'running':
                    break
    except BaseException as exc:
        state.update(outcome='supervisor_failed', error=str(exc))
    finally:
        if proc is not None:
            try:
                stop_group(proc, grace)
            except (OSError, subprocess.SubprocessError) as exc:
                state.update(previous_outcome=state["outcome"], outcome="cleanup_failed", cleanup_error=str(exc))
            state['returncode'] = proc.returncode
            # Drain bounded residual output after termination without waiting for EOF.
            for key in list(selector.get_map().values()):
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    sink, used, _ = streams[key.data]
                    kept = chunk[:max(0, log_limit-used)]
                    sink.write(kept)
                    streams[key.data][2] += len(chunk)-len(kept)
                except (BlockingIOError, OSError):
                    pass
            for pipe in (proc.stdout, proc.stderr):
                pipe.close()
        selector.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for sink, _, _ in streams.values():
            sink.close()
        state.update(finished_at=utc(), elapsed_seconds=time.monotonic()-started,
                     discarded_log_bytes={name: info[2] for name, info in streams.items()})
        atomic_json(run / 'status.json', state)
        make_report(run, state, samples)
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True, help='New directory; existing runs are never overwritten')
    ap.add_argument('--rss-limit-mib', type=int, default=6144)
    ap.add_argument('--min-host-free-percent', type=float, default=5)
    ap.add_argument('--max-seconds', type=float, default=1800)
    ap.add_argument('--stall-seconds', type=float, default=120)
    ap.add_argument('--sample-seconds', type=float, default=1)
    ap.add_argument('--term-grace-seconds', type=float, default=5)
    ap.add_argument('--log-limit-mib', type=int, default=8)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        state = supervise(command, args.run_dir, rss_limit=args.rss_limit_mib*2**20,
                          min_free=args.min_host_free_percent, deadline=args.max_seconds,
                          stall=args.stall_seconds, interval=args.sample_seconds,
                          grace=args.term_grace_seconds, log_limit=args.log_limit_mib*2**20)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    print(json.dumps(state, indent=2))
    return 0 if state['outcome'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
