"""Persistent execution worker: reads JSON jobs from stdin, runs a program on each input in a forked child.

Copied and adapted from:
  exec_select_20261009/code/exec_worker.py

Job:  {"code": str, "inputs": [str, ...], "timeout": float, "expected": [str|null, ...] (optional),
       "early_stop": bool (optional), "out_limit": int (optional), "keep_out": bool (optional)}
Result: {"res": [[status, out_or_detail, wall_seconds, cpu_seconds], ...]}
  status: ok / pass / wrong / error / timeout / compile_error. With "expected", ok becomes pass/wrong.
  Output is returned only when keep_out is true (else "" for ok/pass, the head of the output for wrong).
Sandbox: fresh process group per input, isolated temp directory, 2 GiB address space (RLIMIT_AS),
  CPU limit (RLIMIT_CPU), 10,000 recursion limit, output limit (default 1 MiB), and network socket disabled.
"""
import builtins
import io
import json
import os
import resource
import signal
import socket
import sys
import tempfile
import time
import traceback
import math, collections, itertools, heapq, bisect, functools, re, string, fractions, decimal, random, copy, operator  # noqa: F401 (preload standard library modules for fast fork)


EXACT_INT = os.environ.get('EC_EXACT_INT') == '1'  # integer-looking tokens must match exactly; tolerance only for decimals


def _disable_network():
    """Block all network socket creation inside the sandboxed child process."""
    def _blocked_socket(*_args, **_kwargs):
        raise PermissionError("Network access is disabled inside SSPO exec_worker sandbox")

    socket.socket = _blocked_socket  # type: ignore[misc]
    socket.create_connection = _blocked_socket  # type: ignore[assignment]


def same_output(got: str, exp: str) -> bool:
    g = [l.rstrip() for l in got.strip().splitlines()]
    e = [l.rstrip() for l in exp.strip().splitlines()]
    if g == e:
        return True
    gt, et = got.split(), exp.split()
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        if EXACT_INT and not any(ch in a or ch in b for ch in '.eE'):
            return False
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False
        if abs(fa - fb) > 1e-6 * max(1.0, abs(fb)):
            return False
    return True


def run_one(code_obj, inp: str, timeout: float, out_limit: int):
    inf = tempfile.TemporaryFile()
    inf.write(inp.encode('utf-8', 'replace'))
    inf.flush()
    inf.seek(0)
    outf = tempfile.TemporaryFile()
    errf = tempfile.TemporaryFile()
    work_dir = tempfile.mkdtemp(prefix='sspo_exec_')
    t0 = time.time()
    pid = os.fork()
    if pid == 0:
        rc = 1
        try:
            os.setsid()
            os.chdir(work_dir)
            _disable_network()
            ct = max(1, int(timeout) + 1)
            for lim, val in (
                (resource.RLIMIT_AS, (2 << 30, 2 << 30)),
                (resource.RLIMIT_CPU, (ct, ct + 1)),
            ):
                try:
                    resource.setrlimit(lim, val)
                except (ValueError, OSError):  # e.g. RLIMIT_AS on macOS (local tests only)
                    pass
            os.dup2(inf.fileno(), 0)
            os.dup2(outf.fileno(), 1)
            os.dup2(errf.fileno(), 2)
            sys.stdin = io.TextIOWrapper(io.BufferedReader(io.FileIO(0, 'r', closefd=False)))
            sys.stdout = io.TextIOWrapper(io.BufferedWriter(io.FileIO(1, 'w', closefd=False)))
            sys.stderr = io.TextIOWrapper(io.BufferedWriter(io.FileIO(2, 'w', closefd=False)))
            sys.setrecursionlimit(10000)
            g = {'__name__': '__main__', '__builtins__': builtins}
            try:
                exec(code_obj, g)
                rc = 0
            except SystemExit as e:
                rc = 0 if e.code in (None, 0) else 1
            except BaseException:
                traceback.print_exc()
                rc = 1
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except BaseException:
                rc = 1
        finally:
            os._exit(rc)

    deadline = t0 + timeout
    status = None
    cpu = 0.0
    try:
        while True:
            r, st, ru = os.wait4(pid, os.WNOHANG)
            if r:
                status = st
                cpu = ru.ru_utime + ru.ru_stime
                break
            if time.time() > deadline:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    os.waitpid(pid, 0)
                except Exception:
                    pass
                for f in (inf, outf, errf):
                    f.close()
                return 'timeout', '', timeout, timeout
            time.sleep(0.002)
        secs = time.time() - t0
        outf.seek(0)
        out = outf.read(out_limit).decode('utf-8', 'replace')
        errf.seek(0)
        err = errf.read()[-2000:].decode('utf-8', 'replace')
        for f in (inf, outf, errf):
            f.close()
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            return (
                ('timeout', '', timeout, timeout)
                if sig == signal.SIGXCPU
                else ('error', f'killed by signal {sig}\n' + err[-500:], secs, cpu)
            )
        if os.WEXITSTATUS(status) != 0:
            return 'error', err[-600:], secs, cpu
        return 'ok', out, secs, cpu
    finally:
        # Clean up temporary working directory
        try:
            for root_d, dirs_d, files_d in os.walk(work_dir, topdown=False):
                for name in files_d:
                    try:
                        os.unlink(os.path.join(root_d, name))
                    except OSError:
                        pass
                for name in dirs_d:
                    try:
                        os.rmdir(os.path.join(root_d, name))
                    except OSError:
                        pass
            os.rmdir(work_dir)
        except OSError:
            pass


def handle(job):
    code, inputs, to = job['code'], job['inputs'], float(job.get('timeout', 6.0))
    exp = job.get('expected')
    es = job.get('early_stop', False)
    out_limit = int(job.get('out_limit', 1 << 20))
    keep = bool(job.get('keep_out', False))
    try:
        co = compile(code, 'prog.py', 'exec')
    except BaseException:
        msg = traceback.format_exc(limit=0)[-600:]
        return [['compile_error', msg, 0.0, 0.0] for _ in inputs]
    res = []
    for k, inp in enumerate(inputs):
        st, out, secs, cpu = run_one(co, inp, to, out_limit)
        if st == 'ok' and exp is not None and k < len(exp) and exp[k] is not None:
            st = 'pass' if same_output(out, exp[k]) else 'wrong'
        if st in ('ok', 'pass'):
            res.append([st, out if keep else '', round(secs, 3), round(cpu, 3)])
        elif st == 'wrong':
            res.append([st, out if keep else out[:300], round(secs, 3), round(cpu, 3)])
        else:
            res.append([st, out, round(secs, 3), round(cpu, 3)])
        if es and st not in ('ok', 'pass'):
            break
    return res


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        try:
            r = handle(job)
        except BaseException as e:
            r = [['error', 'worker failure ' + repr(e)[:200], 0.0, 0.0]]
        sys.__stdout__.write(json.dumps({'res': r}) + '\n')
        sys.__stdout__.flush()


if __name__ == '__main__':
    main()
