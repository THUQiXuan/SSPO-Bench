"""Stage 3 of Checker Pipeline: run distinct candidate programs on the execution-check cases of their problem.

Copied and adapted from:
  exec_select_20261009/code/ec_run.py

Runs each distinct candidate program on:
- public examples (to record `public_all` and `public_frac` and `public_keys`),
- `max` inputs (valid or unverified): status, wall_s, cpu_s, and canonical `out_key` with 6s limit,
- `small`/`edge` inputs (valid or unverified): status and canonical `out_key`.

Results are cached by program hash in `<exec_dir>/<pid>.json` with atomic writes.
Usage: python -m sspo.ec_run [shard nshards] [--problems PATH] [--candidates PATH] [--cases-dir DIR] [--exec-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os

from sspo.atomic_io import atomic_write_json
from sspo.ec_common import R2, VAR, code_hash, log, main_problems, out_key, pmap, run_prog


def run_candidate_on_cases(code: str | None, prob: dict, cases: dict) -> dict:
    """Execute one candidate program on public examples and check inputs (`max` and `small`)."""
    tl = min(1.5, float(prob.get('timeout', 1.5)))
    ex_in = [t[0] for t in prob.get('public', [])]
    ex_out = [t[1] for t in prob.get('public', [])]

    if not code or not str(code).strip():
        return {
            'public_all': False,
            'public_frac': 0.0,
            'public_keys': [],
            'max': [],
            'small': [],
        }

    pub_res = run_prog(code, ex_in, tl, expected=ex_out, keep_out=True) if ex_in else []
    n_pub_pass = sum(r[0] == 'pass' for r in pub_res)
    pub_all = (n_pub_pass == len(ex_in)) if ex_in else True
    pub_frac = (n_pub_pass / len(ex_in)) if ex_in else 1.0
    pub_keys = [out_key(r[1]) if r[0] in ('ok', 'pass', 'wrong') else f'ERR:{r[0]}' for r in pub_res]

    mx = [x for x in cases.get('inputs', []) if x['mode'] == 'max' and x.get('valid') is not False and x.get('inp')]
    sm = [x for x in cases.get('inputs', []) if x['mode'] != 'max' and x.get('valid') is not False and x.get('inp')]

    rm = run_prog(code, [x['inp'] for x in mx], tl, early_stop=True, keep_out=True)
    rs = run_prog(code, [x['inp'] for x in sm], tl, keep_out=True)

    return {
        'public_all': pub_all,
        'public_frac': pub_frac,
        'public_keys': pub_keys,
        'max': [
            [x['id'], r[0], r[2], r[3], out_key(r[1]) if r[0] == 'ok' else None]
            for x, r in zip(mx, rm)
        ],
        'small': [
            [x['id'], r[0], out_key(r[1]) if r[0] == 'ok' else None]
            for x, r in zip(sm, rs)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run candidate programs on execution-check cases.")
    parser.add_argument('shard', nargs='?', type=int, default=0)
    parser.add_argument('nshards', nargs='?', type=int, default=1)
    parser.add_argument('--problems', type=str, default=None)
    parser.add_argument('--candidates', type=str, default=None)
    parser.add_argument('--cases-dir', type=str, default=None)
    parser.add_argument('--exec-dir', type=str, default=None)
    args = parser.parse_args()

    cases_dir = args.cases_dir or f'{R2}/cases{VAR}'
    exec_dir = args.exec_dir or f'{R2}/exec{VAR}'
    cand_file = args.candidates or f'{R2}/data/candidates.jsonl'
    os.makedirs(exec_dir, exist_ok=True)

    P = main_problems(args.problems)
    with open(cand_file, 'r', encoding='utf-8') as f:
        C = [json.loads(l) for l in f if l.strip()]

    progs: dict[str, dict[str, str]] = {}
    for c in C:
        if c.get('code'):
            k = c.get('key') or code_hash(c['code'])
            progs.setdefault(c['pid'], {}).setdefault(k, c['code'])

    def missing(p):
        ef = f'{exec_dir}/{p}.json'
        if not os.path.exists(ef):
            return bool(progs.get(p))
        with open(ef, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        return bool(set(progs.get(p, {})) - set(existing))

    pids = [
        p
        for p in sorted(P)[args.shard :: args.nshards]
        if os.path.exists(f'{cases_dir}/{p}.json') and missing(p)
    ]
    log('problems', len(pids))
    for b in range(0, len(pids), 6):
        batch = pids[b : b + 6]
        cases = {p: json.load(open(f'{cases_dir}/{p}.json', 'r', encoding='utf-8')) for p in batch}
        old = {
            p: (
                json.load(open(f'{exec_dir}/{p}.json', 'r', encoding='utf-8'))
                if os.path.exists(f'{exec_dir}/{p}.json')
                else {}
            )
            for p in batch
        }
        jobs = [(p, k) for p in batch for k in progs.get(p, {}) if k not in old[p]]

        def go(job):
            p, k = job
            rec = run_candidate_on_cases(progs[p][k], P[p], cases[p])
            return p, k, rec

        res = {p: dict(old[p]) for p in batch}
        for p, k, v in pmap(go, jobs):
            res[p][k] = v
        for p in batch:
            atomic_write_json(f'{exec_dir}/{p}.json', res[p])
        log('ran', b + len(batch), '/', len(pids), 'programs', len(jobs))


if __name__ == '__main__':
    main()
