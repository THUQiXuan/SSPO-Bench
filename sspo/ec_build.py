"""Stage 2 of Checker Construction: build execution-check cases per problem from the generated helper programs.

Copied and adapted from:
  exec_select_20261009/code/ec_build.py

1. Validators are kept only if they accept every example input of the statement.
2. Each generator is run with MODE/SEED = max x2, small x6, edge x2; inputs are de-duplicated.
3. An input is valid if every kept validator prints VALID (unverified if the problem has no kept validator).
4. Brute-force solutions are kept only if they pass every example; the reference output of a small/edge input is the
   majority output of the kept brute-force solutions that finish within 10 s (a(x) >= 2 = confirmed reference).
Usage: python -m sspo.ec_build [shard nshards] [--problems PATH] [--artifacts-dir DIR] [--cases-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

from sspo.atomic_io import atomic_write_json
from sspo.ec_common import NCPU, R2, VAR, log, main_problems, out_key, run_prog, sha

RUNS = [('max', 1), ('max', 2)] + [('small', s) for s in range(1, 7)] + [('edge', 1), ('edge', 2)]
BIG = 64 << 20


def build(pid: str, prob: dict, art: dict) -> dict:
    """Build execution-check cases and confirmed references for one problem."""
    ex_in = [t[0] for t in prob['public']]
    ex_out = [t[1] for t in prob['public']]
    vals, val_info = [], []
    for code in art.get('val', []):
        if not code:
            val_info.append('nocode')
            continue
        r = run_prog(code, ex_in, 10.0, keep_out=True)
        ok = all(x[0] == 'ok' and x[1].strip() == 'VALID' for x in r)
        val_info.append('kept' if ok else 'rejected_examples')
        if ok:
            vals.append(code)

    gen_info, raw = [], []
    for gi, code in enumerate(art.get('gen', [])):
        if not code:
            gen_info.append('nocode')
            continue
        r = run_prog(code, [f'{m} {s}\n' for m, s in RUNS], 30.0, keep_out=True, out_limit=BIG)
        n_ok = 0
        for (m, s), x in zip(RUNS, r):
            if x[0] == 'ok' and x[1].strip():
                raw.append({
                    'mode': m,
                    'gen': gi,
                    'seed': s,
                    'inp': x[1] if x[1].endswith('\n') else x[1] + '\n',
                })
                n_ok += 1
        gen_info.append(f'{n_ok}/{len(RUNS)}')

    seen, inputs = set(), []
    for x in raw:
        h = sha(x['inp'])
        if h in seen:
            continue
        seen.add(h)
        x['id'] = h
        inputs.append(x)

    for x in inputs:
        if vals:
            vr = [run_prog(v, [x['inp']], 30.0, keep_out=True, out_limit=1 << 16)[0] for v in vals]
            x['valid'] = all(r[0] == 'ok' and r[1].strip() == 'VALID' for r in vr)
        else:
            x['valid'] = None

    bfs, bf_info = [], []
    for code in art.get('bf', []):
        if not code:
            bf_info.append('nocode')
            continue
        r = run_prog(code, ex_in, 10.0, expected=ex_out)
        ok = all(x[0] == 'pass' for x in r)
        bf_info.append('kept' if ok else 'failed_examples')
        if ok:
            bfs.append(code)

    small = [x for x in inputs if x['mode'] != 'max' and x['valid'] is not False]
    outs = [run_prog(b, [x['inp'] for x in small], 10.0, keep_out=True) for b in bfs]
    for k, x in enumerate(small):
        oks = [o[k][1] for o in outs if o[k][0] == 'ok']
        x['n_bf_ok'] = len(oks)
        if oks:
            cnt = collections.Counter(out_key(o) for o in oks)
            key, n = cnt.most_common(1)[0]
            x['ref'] = next(o for o in oks if out_key(o) == key)
            x['ref_key'] = key
            x['ref_agree'] = n
        else:
            x['ref'] = None
            x['ref_key'] = None
            x['ref_agree'] = 0

    for x in inputs:
        if x['valid'] is False and x['mode'] == 'max':
            x['inp'] = None  # keep only metadata of rejected large inputs

    return {
        'pid': pid,
        'val_info': val_info,
        'gen_info': gen_info,
        'bf_info': bf_info,
        'n_val_kept': len(vals),
        'n_bf_kept': len(bfs),
        'bf_codes': bfs,
        'inputs': inputs,
    }


def build_all(
    problems: dict,
    artifacts_dir: str,
    cases_dir: str,
    shard: int = 0,
    nshards: int = 1,
) -> dict:
    os.makedirs(cases_dir, exist_ok=True)
    pids = [
        p
        for p in sorted(problems)[shard::nshards]
        if os.path.exists(f'{artifacts_dir}/{p}.json') and not os.path.exists(f'{cases_dir}/{p}.json')
    ]
    log('problems to build', len(pids))
    results = {}

    def go(p):
        try:
            with open(f'{artifacts_dir}/{p}.json', 'r', encoding='utf-8') as f:
                art = json.load(f)
            c = build(p, problems[p], art)
            atomic_write_json(f'{cases_dir}/{p}.json', c)
            inp = c['inputs']
            return (
                p,
                c,
                len(inp),
                sum(x['valid'] is not False for x in inp if x['mode'] == 'max'),
                sum(x.get('ref') is not None and x.get('ref_agree', 0) >= 2 for x in inp),
                c['n_val_kept'],
                c['n_bf_kept'],
            )
        except Exception as e:
            return p, None, 'error', repr(e)[:200], 0, 0, 0

    with ThreadPoolExecutor(max(2, min(8, NCPU - 1))) as ex:
        for r in ex.map(go, pids):
            p, c, *stats = r
            if c is not None:
                results[p] = c
            log('built', p, *stats)
    return results


def main():
    parser = argparse.ArgumentParser(description="Build execution checker cases from helper artifacts.")
    parser.add_argument('shard', nargs='?', type=int, default=0)
    parser.add_argument('nshards', nargs='?', type=int, default=1)
    parser.add_argument('--problems', type=str, default=None)
    parser.add_argument('--artifacts-dir', type=str, default=None)
    parser.add_argument('--cases-dir', type=str, default=None)
    args = parser.parse_args()

    art_dir = args.artifacts_dir or f'{R2}/artifacts{VAR}'
    cases_dir = args.cases_dir or f'{R2}/cases{VAR}'
    probs = main_problems(args.problems)
    build_all(probs, art_dir, cases_dir, shard=args.shard, nshards=args.nshards)


if __name__ == '__main__':
    main()
