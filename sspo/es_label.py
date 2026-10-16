"""Hidden-test labeling for distinct candidate programs (used for gold reward g(y) and evaluation).

Copied and adapted from:
  exec_select_20261009/code/es_label.py

Usage: python -m sspo.es_label [SHARD NSHARDS] [--problems PATH] [--candidates PATH] [--out-dir DIR]
Resumable with atomic writes.
"""
from __future__ import annotations

import argparse
import json
import os

from sspo.atomic_io import atomic_write_json
from sspo.ec_common import R2, code_hash, hidden_tests, log, main_problems, pmap, run_prog


def label_program_on_hidden(code: str | None, prob: dict) -> dict:
    """Run one program on all hidden tests of `prob` and return `{"all": bool, "frac": float, "n": int, "fail_type": str|None}`."""
    tests = hidden_tests(prob)[:16]
    if not tests:
        return {'all': False, 'frac': 0.0, 'n': 0, 'fail_type': 'no_tests'}
    if not code or not str(code).strip():
        return {'all': False, 'frac': 0.0, 'n': len(tests), 'fail_type': 'nocode'}
    res = run_prog(
        code,
        [t[0] for t in tests],
        min(1.5, float(prob.get('timeout', 1.5))),
        expected=[t[1] for t in tests],
        early_stop=True,
    )
    npass = sum(r[0] == 'pass' for r in res)
    bad = [r[0] for r in res if r[0] != 'pass']
    return {
        'all': npass == len(tests),
        'frac': npass / max(1, len(tests)),
        'n': len(tests),
        'fail_type': None if not bad else max(set(bad), key=bad.count),
    }


def main():
    parser = argparse.ArgumentParser(description="Label candidate programs against hidden tests.")
    parser.add_argument('shard', nargs='?', type=int, default=0)
    parser.add_argument('nshards', nargs='?', type=int, default=1)
    parser.add_argument('--problems', type=str, default=None)
    parser.add_argument('--candidates', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or f'{R2}/data'
    cand_file = args.candidates or f'{R2}/data/candidates.jsonl'
    os.makedirs(out_dir, exist_ok=True)

    P = main_problems(args.problems)
    pids = set(sorted(P)[args.shard :: args.nshards])
    progs = {}
    with open(cand_file, 'r', encoding='utf-8') as f:
        for l in f:
            if not l.strip():
                continue
            c = json.loads(l)
            if c['pid'] in pids and c.get('code'):
                k = c.get('key') or code_hash(c['code'])
                progs.setdefault(f"{c['pid']}:{k}", (c['pid'], c['code']))

    out = f'{out_dir}/labels_s{args.shard}.json'
    done = json.load(open(out, 'r', encoding='utf-8')) if os.path.exists(out) else {}
    todo = [k for k in sorted(progs) if k not in done]
    log('programs', len(progs), 'todo', len(todo))

    def go(k):
        pid, code = progs[k]
        return k, label_program_on_hidden(code, P[pid])

    for i in range(0, len(todo), 200):
        for k, v in pmap(go, todo[i : i + 200]):
            done[k] = v
        atomic_write_json(out, done)
        log('labelled', len(done), '/', len(progs))
    log('done')


if __name__ == '__main__':
    main()
