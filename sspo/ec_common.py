"""Shared helpers for the execution-check pipeline: paths, sandboxed exec worker pool, output keys, data loading.

Copied and adapted from:
  exec_select_20261009/code/ec_common.py
Hardcoded cluster paths (R1/R2) removed and replaced with repository-relative paths overridable via environment variables.
"""
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
R1 = os.environ.get('EC_R1', str(REPO_ROOT))
R2 = os.environ.get('EC_R2', str(REPO_ROOT / 'cache' / 'checkers'))
TAG = 'sspo'
VAR = os.environ.get('EC_VARIANT', '')
NCPU = int(os.environ.get('SSPO_NCPU', os.environ.get('EC_NCPU', str(max(2, (os.cpu_count() or 4))))))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', 'replace')).hexdigest()[:16]


def norm_code(code: str) -> str:
    """Whitespace/comment-insensitive key for de-duplicating programs."""
    if not code:
        return ''
    lines = []
    for l in code.splitlines():
        l = l.rstrip()
        if not l.strip() or l.strip().startswith('#'):
            continue
        lines.append(l)
    return '\n'.join(lines)


def code_hash(code: str | None) -> str:
    """Return canonical 16-hex SHA-1 of normalized code (or 'nocode' if None/empty)."""
    if not code or not norm_code(code):
        return 'nocode'
    return sha(norm_code(code))


def main_problems(problems_path: str | None = None) -> dict:
    """Load problems dictionary `{pid: prob_dict}` from JSON or JSONL."""
    if problems_path is None:
        env_p = os.environ.get('SSPO_PROBLEMS_FILE')
        if env_p and os.path.exists(env_p):
            problems_path = env_p
        elif os.path.exists(f'{R1}/data/train.jsonl'):
            problems_path = f'{R1}/data/train.jsonl'
        elif os.path.exists(f'{R1}/data/lcb_main.json'):
            problems_path = f'{R1}/data/lcb_main.json'
        else:
            problems_path = f'{R1}/data/sample_problems.jsonl'

    if problems_path.endswith('.jsonl'):
        with open(problems_path, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(problems_path, 'r', encoding='utf-8') as f:
            rows = json.load(f)
    return {p['pid']: p for p in rows}


def hidden_tests(prob: dict) -> list:
    if 'hidden' in prob and prob['hidden'] is not None:
        return prob['hidden']
    hf = prob.get('hidden_file')
    if hf:
        p = Path(hf)
        if not p.is_absolute():
            p = REPO_ROOT / p
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def last_code_block(text: str | None) -> str | None:
    """Code of the last ```python block (or last ``` block) in a model response."""
    if text is None:
        return None
    blocks = re.findall(r'```(?:python3|python|py|Python)?[ \t]*\n(.*?)```', text, re.S)
    if not blocks:
        return None
    code = blocks[-1].strip()
    return (code + '\n') if code else None


# ---------------------------------------------------------------- sandboxed execution pool
_POOL: queue.Queue | None = None
_LOCK = threading.Lock()


def _worker_cmd() -> list[str]:
    worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exec_worker.py')
    cmd = [sys.executable, worker_py]
    if os.environ.get('SSPO_UNSHARE_NET', '0') == '1' and sys.platform.startswith('linux') and shutil.which('unshare'):
        cmd = ['unshare', '-rn'] + cmd
    return cmd


def _proc() -> subprocess.Popen:
    return subprocess.Popen(
        _worker_cmd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _pool() -> queue.Queue:
    global _POOL
    with _LOCK:
        if _POOL is None:
            _POOL = queue.Queue()
            n_workers = max(8, min(12, NCPU * 2))
            for _ in range(n_workers):
                _POOL.put(_proc())
    return _POOL


def shutdown_pool() -> None:
    """Terminate all persistent worker processes in the pool."""
    global _POOL
    with _LOCK:
        if _POOL is None:
            return
        while not _POOL.empty():
            try:
                w = _POOL.get_nowait()
                w.kill()
                w.wait(timeout=1.0)
            except Exception:
                pass
        _POOL = None


def run_prog(
    code: str | None,
    inputs: list[str],
    timeout: float = 6.0,
    expected: list[str | None] | None = None,
    early_stop: bool = False,
    keep_out: bool = False,
    out_limit: int = 1 << 20,
) -> list[tuple[str, str, float, float]]:
    """Run a program on several inputs in one sandboxed worker.
    Returns [(status, out_or_detail, wall_s, cpu_s), ...].
    """
    if not inputs:
        return []
    if code is None or not str(code).strip():
        return [('nocode', '', 0.0, 0.0) for _ in inputs]
    q = _pool()
    w = q.get()
    try:
        job = {
            'code': code,
            'inputs': inputs,
            'timeout': float(timeout),
            'early_stop': bool(early_stop),
            'keep_out': bool(keep_out),
            'out_limit': int(out_limit),
        }
        if expected is not None:
            job['expected'] = expected
        assert w.stdin is not None and w.stdout is not None
        w.stdin.write(json.dumps(job) + '\n')
        w.stdin.flush()
        line = w.stdout.readline()
        if not line:
            raise RuntimeError('worker died')
        res = [tuple(x) for x in json.loads(line)['res']]
    except Exception as e:
        try:
            w.kill()
        except Exception:
            pass
        w = _proc()
        res = [('error', 'worker restart ' + repr(e)[:200], 0.0, 0.0) for _ in inputs]
    finally:
        q.put(w)
    return res


def pmap(fn, items, workers: int | None = None):
    items_list = list(items)
    if not items_list:
        return []
    n_w = workers or max(8, min(12, NCPU * 2))
    with ThreadPoolExecutor(n_w) as ex:
        return list(ex.map(fn, items_list))


def same_output(got: str, exp: str) -> bool:
    from sspo.exec_worker import same_output as so
    return so(got, exp)


def out_key(out: str | None) -> str:
    """Canonical form of a program output for majority votes & signatures:
    whitespace-normalized, float tokens rounded to 6 significant digits (%.6g), hashed with SHA-1[:16].
    """
    if out is None:
        return sha('')
    toks = out.split()
    res = []
    for t in toks:
        try:
            if any(c in t for c in '.eE') and not t.lstrip('+-').isdigit():
                val = float(t)
                if math.isfinite(val):
                    # Normalize -0.0 to 0.0
                    if val == 0.0:
                        val = 0.0
                    res.append(f'{val:.6g}')
                    continue
        except ValueError:
            pass
        res.append(t)
    return sha(' '.join(res))
