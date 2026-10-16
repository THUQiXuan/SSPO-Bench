"""Stage 1 of Checker Construction: helper programs written by the frozen base policy model (thinking off) from the problem statement only.

Copied and adapted from:
  exec_select_20261009/code/ec_gen.py
Hardcoded 35B model path removed; defaults to `SSPO_HELPER_MODEL` / `Qwen/Qwen3-8B` with thinking disabled (`EC_THINK=0`).

Per problem: 4 input generators (`gen`), 4 brute-force reference solutions (`bf`), 2 input validators (`val`).
vLLM offline (or mock generator in local/CPU mode), sharded by problem index, resumable with atomic writes.
Usage: python -m sspo.ec_gen [SHARD NSHARDS] [--problems PATH] [--out-dir DIR] [--model MODEL] [--mock]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, Optional

from sspo.atomic_io import atomic_write_json
from sspo.ec_common import R2, VAR, last_code_block, log, main_problems, sha

DEFAULT_MODEL = os.environ.get('SSPO_HELPER_MODEL', os.environ.get('SSPO_MODEL', 'Qwen/Qwen3-8B'))
K = {'gen': 4, 'bf': 4, 'val': 2}

HEAD = (
    "Below is a competitive programming problem (standard input / standard output).\n\n"
    "[PROBLEM]\n{q}\n[END OF PROBLEM]\n\n"
)
PROMPTS = {
    'gen': HEAD + (
        "Write a Python 3 program that generates ONE valid test input for this problem.\n"
        "The program reads two values from standard input: MODE and SEED (for example `max 3`). Call random.seed(SEED).\n"
        "MODE is one of:\n"
        "- max: an input as large as the constraints allow (maximum sizes and value ranges), designed to be a worst case for "
        "running time, e.g. structures that make naive or quadratic algorithms slow;\n"
        "- small: a small random valid input (tiny sizes, e.g. at most 6 elements, and small values) that a brute-force "
        "solution can handle;\n"
        "- edge: a small input that exercises boundary cases (minimum sizes, all values equal, extreme values, and similar).\n"
        "The generated input must strictly satisfy every constraint and the exact input format of the statement, including "
        "any guarantees the statement makes about the input. Print only the input, nothing else.\n"
        "Give the complete program in a single ```python code block."
    ),
    'bf': HEAD + (
        "Write a brute-force Python 3 solution that is obviously correct, intended only for very small inputs. It may be "
        "exponentially slow. Prefer directly enumerating all possibilities exactly as the statement defines them over any "
        "clever algorithm, so that its correctness is easy to trust. Read from standard input and print the answer in the "
        "required output format.\nGive the complete program in a single ```python code block."
    ),
    'val': HEAD + (
        "Write a Python 3 input validator. It reads one test input from standard input and prints exactly VALID if the "
        "input follows the input format and satisfies every constraint and guarantee in the statement, otherwise exactly "
        "INVALID. Be strict about the number of tokens and lines and about all value ranges.\n"
        "Give the complete program in a single ```python code block."
    ),
}


def default_mock_helper_generator(prob: dict, kind: str, n: int) -> list[str]:
    """Generate deterministic helper programs for CPU/smoke testing from problem metadata or examples."""
    if kind in prob.get('mock_helpers', {}):
        codes = prob['mock_helpers'][kind]
        return [f"```python\n{c}\n```" for c in codes[:n]]

    ex_in = [t[0] for t in prob.get('public', [])]
    if kind == 'gen':
        # Deterministic generator that replays public inputs or small variations based on seed
        sample_repr = repr(ex_in if ex_in else ["1\n"])
        code = (
            "import sys, random\n"
            "parts = sys.stdin.read().split()\n"
            "mode = parts[0] if parts else 'small'\n"
            "seed = int(parts[1]) if len(parts) > 1 else 1\n"
            "random.seed(seed)\n"
            f"samples = {sample_repr}\n"
            "print(samples[(seed - 1) % len(samples)], end='')\n"
        )
        return [f"```python\n{code}```" for _ in range(n)]
    if kind == 'bf':
        ref_sol = prob.get('reference_solution')
        if ref_sol:
            return [f"```python\n{ref_sol}\n```" for _ in range(n)]
        pairs_repr = repr({t[0].strip(): t[1] for t in prob.get('public', [])})
        code = (
            "import sys\n"
            "inp = sys.stdin.read().strip()\n"
            f"table = {pairs_repr}\n"
            "if inp in table:\n"
            "    print(table[inp], end='')\n"
            "else:\n"
            "    sys.exit(1)\n"
        )
        return [f"```python\n{code}```" for _ in range(n)]
    if kind == 'val':
        code = (
            "import sys\n"
            "data = sys.stdin.read()\n"
            "print('VALID' if data.strip() else 'INVALID')\n"
        )
        return [f"```python\n{code}```" for _ in range(n)]
    return ["```python\npass\n```" for _ in range(n)]


def generate_helpers(
    problems: Dict[str, dict],
    out_dir: str,
    shard: int = 0,
    nshards: int = 1,
    model_name: str = DEFAULT_MODEL,
    think: bool = False,
    tensor_parallel_size: int = 1,
    mock_fn: Optional[Callable[[dict, str, int], list[str]]] = None,
) -> Dict[str, dict]:
    """Generate and persist helper artifacts (`artifacts/<pid>.json`) for a problem shard."""
    os.makedirs(out_dir, exist_ok=True)
    pids = sorted(problems)[shard::nshards]
    todo = [p for p in pids if not os.path.exists(f'{out_dir}/{p}.json')]
    log('problems', len(pids), 'todo', len(todo))
    if not todo:
        return {p: json.load(open(f'{out_dir}/{p}.json', 'r', encoding='utf-8')) for p in pids}

    res: Dict[str, dict] = {}
    if mock_fn is not None or os.environ.get('SSPO_MOCK_LLM', '0') == '1':
        fn = mock_fn or default_mock_helper_generator
        for p in todo:
            r = {'pid': p}
            for kind, k in K.items():
                replies = fn(problems[p], kind, k)
                r[kind] = [last_code_block(ans) for ans in replies]
                r[kind + '_meta'] = [{'finish': 'stop', 'tokens': len((ans or '').split())} for ans in replies]
            atomic_write_json(f'{out_dir}/{p}.json', r)
            res[p] = r
        return res

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=int(os.environ.get('SSPO_MAX_MODEL_LEN', '16384')),
        gpu_memory_utilization=float(os.environ.get('SSPO_GPU_MEM_UTIL', '0.85')),
        enable_prefix_caching=True,
        max_num_seqs=128,
        seed=shard,
    )
    tok = llm.get_tokenizer()
    reqs = []
    for p in todo:
        for kind, k in K.items():
            msgs = [{'role': 'user', 'content': PROMPTS[kind].format(q=problems[p]['question'])}]
            try:
                prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think
                )
            except TypeError:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            seed_val = (int(sha(str(p)), 16) * 10 + list(K).index(kind)) % (2 ** 31)
            sp = SamplingParams(
                n=k,
                temperature=0.6 if think else 0.7,
                top_p=0.95 if think else 0.8,
                top_k=20,
                max_tokens=16000 if think else 8000,
                seed=seed_val,
            )
            reqs.append((p, kind, prompt, sp))

    t0 = time.time()
    outs = llm.generate([r[2] for r in reqs], [r[3] for r in reqs], use_tqdm=False)
    ntok = sum(len(c.token_ids) for o in outs for c in o.outputs)
    log('generated', len(reqs), 'requests', ntok, 'tokens', f'{time.time() - t0:.0f}s')

    for (p, kind, _, _), o in zip(reqs, outs):
        r = res.setdefault(p, {'pid': p})
        for c in o.outputs:
            text = c.text
            if think:
                answer = (
                    text.split('</think>')[-1]
                    if '</think>' in text
                    else ('' if '<think>' in text or c.finish_reason == 'length' else text)
                )
            else:
                answer = text.split('</think>')[-1]
            r.setdefault(kind, []).append(last_code_block(answer))
            r.setdefault(kind + '_meta', []).append({'finish': c.finish_reason, 'tokens': len(c.token_ids)})

    for p, r in res.items():
        atomic_write_json(f'{out_dir}/{p}.json', r)
    n_ok = {k: sum(x is not None for r in res.values() for x in r.get(k, [])) for k in K}
    log('written', len(res), 'code blocks found', n_ok)
    return res


def main():
    parser = argparse.ArgumentParser(description="Generate execution checker helper programs (gen/bf/val).")
    parser.add_argument('shard', nargs='?', type=int, default=0, help="Shard index (0-based)")
    parser.add_argument('nshards', nargs='?', type=int, default=1, help="Total number of shards")
    parser.add_argument('--problems', type=str, default=None, help="Path to problems JSON/JSONL")
    parser.add_argument('--out-dir', type=str, default=None, help="Directory for helper artifacts")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument('--tp', type=int, default=int(os.environ.get('SSPO_TP', '1')), help="Tensor parallel size")
    parser.add_argument('--mock', action='store_true', help="Use deterministic mock helpers (no GPU/vLLM)")
    args = parser.parse_args()

    out_dir = args.out_dir or f'{R2}/artifacts{VAR}'
    think = os.environ.get('EC_THINK', '0') == '1'
    probs = main_problems(args.problems)
    generate_helpers(
        problems=probs,
        out_dir=out_dir,
        shard=args.shard,
        nshards=args.nshards,
        model_name=args.model,
        think=think,
        tensor_parallel_size=args.tp,
        mock_fn=default_mock_helper_generator if args.mock else None,
    )


if __name__ == '__main__':
    main()
