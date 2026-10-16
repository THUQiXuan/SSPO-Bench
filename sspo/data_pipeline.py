"""Data preparation, unique-answer filtering, representative program extraction, and hidden-test labeling pipeline.

Supports:
1. Hugging Face `deepmind/code_contests` (primary), `BAAI/TACO` (optional), `livecodebench/code_generation_lite` (optional),
   and local `data/sample_problems.jsonl` (offline / smoke / unit test mode).
2. Filtering to unique-output problems:
   - Statement heuristic regex (`any of them`, `print any`, `multiple answers`, `interactive`, `任意`, etc.)
   - Custom checker / interactive flags
   - Public examples >= 1 and hidden tests >= 5
   - Consistency of expected outputs across duplicate inputs (`out_key`)
   - Reference solution output uniqueness check (when multiple reference solutions are provided)
   - Records exact removal counts and `removed_fraction` in `data/filter_stats.json`.
3. Splitting into `train.jsonl`, `dev.jsonl` (200 held-out problems in pilot, configurable), and `test.jsonl`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sspo.atomic_io import REPO_ROOT, atomic_write_json, atomic_write_jsonl, read_jsonl, to_rel_path
from sspo.checker import extract_representative_program
from sspo.ec_common import code_hash, out_key, run_prog
from sspo.es_label import label_program_on_hidden

MULTI_ANSWER_REGEX = re.compile(
    r"any of them|if there are (several|multiple)|print any|output any|return any|"
    r"any (valid|possible|such|order)|multiple (valid )?(answers|solutions)|"
    r"interactive problem|flush the output|sys\.stdout\.flush|"
    r"任意一个|输出任意|任一合法|多种答案|如果有多种|交互题",
    re.IGNORECASE,
)


def is_unique_answer_problem(prob: Dict[str, Any], check_ref_exec: bool = False) -> Tuple[bool, str]:
    """Check whether `prob` has a unique canonical output and sufficient public/hidden tests.

    Returns `(keep: bool, reason: str)`.
    """
    if prob.get("is_interactive") or prob.get("custom_checker"):
        return False, "custom_checker_or_interactive"

    question = str(prob.get("question", ""))
    if MULTI_ANSWER_REGEX.search(question):
        return False, "multi_answer_statement"

    public = prob.get("public") or []
    hidden = prob.get("hidden") or []
    min_hidden = int(prob.get("min_hidden", 5))
    if len(public) < 1 or len(hidden) < min_hidden:
        return False, "insufficient_tests"

    # Check that identical inputs in public + hidden do not map to distinct canonical outputs
    seen_io: Dict[str, str] = {}
    for inp, exp in list(public) + list(hidden):
        if inp is None or exp is None or not str(exp).strip():
            return False, "insufficient_tests"
        inp_norm = str(inp).strip()
        exp_k = out_key(str(exp))
        if inp_norm in seen_io and seen_io[inp_norm] != exp_k:
            return False, "inconsistent_expected_outputs"
        seen_io[inp_norm] = exp_k

    # If multiple reference solutions are provided, check if they disagree on public examples
    ref_solutions = prob.get("reference_solutions") or []
    if check_ref_exec and len(ref_solutions) >= 2 and public:
        ex_in = [t[0] for t in public[:2]]
        keys_per_sol = set()
        for sol in ref_solutions[:3]:
            res = run_prog(sol, ex_in, timeout=4.0, keep_out=True)
            if all(r[0] == "ok" for r in res):
                keys_per_sol.add(tuple(out_key(r[1]) for r in res))
        if len(keys_per_sol) > 1:
            return False, "multi_answer_ref_disagree"

    return True, "kept"


def filter_problems(
    raw_problems: Iterable[Dict[str, Any]],
    check_ref_exec: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter raw problem records and return `(kept_problems, filter_stats)`."""
    kept: List[Dict[str, Any]] = []
    reasons: Dict[str, int] = collections.Counter()
    total = 0

    for prob in raw_problems:
        total += 1
        ok, reason = is_unique_answer_problem(prob, check_ref_exec=check_ref_exec)
        if ok:
            # Normalize schema fields
            norm_prob = {
                "pid": str(prob["pid"]),
                "source": str(prob.get("source", "code_contests")),
                "split": str(prob.get("split", "train")),
                "contest_id": str(prob.get("contest_id", f"contest_{prob['pid']}")),
                "difficulty": str(prob.get("difficulty", "unknown")),
                "timeout": float(prob.get("timeout", 6.0)),
                "question": str(prob["question"]),
                "public": [[str(i), str(o)] for i, o in prob["public"]],
                "hidden": [[str(i), str(o)] for i, o in prob["hidden"][:40]],
                "reference_solution": prob.get("reference_solution"),
            }
            if "wrong_solutions" in prob:
                norm_prob["wrong_solutions"] = prob["wrong_solutions"]
            if "tle_solution" in prob:
                norm_prob["tle_solution"] = prob["tle_solution"]
            kept.append(norm_prob)
        else:
            reasons[reason] += 1

    removed_total = total - len(kept)
    stats = {
        "total_raw": total,
        "kept": len(kept),
        "removed_total": removed_total,
        "removed_fraction": round(removed_total / max(1, total), 4),
        "removed_by_reason": dict(reasons),
    }
    return kept, stats


def convert_code_contests_row(row: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """Convert one HuggingFace `deepmind/code_contests` dataset row into SSPO problem format."""
    name = str(row.get("name") or f"{split}_{idx}")
    pid = "cc-" + re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:64]
    pub_tests = row.get("public_tests") or {}
    priv_tests = row.get("private_tests") or {}
    gen_tests = row.get("generated_tests") or {}

    public = list(zip(pub_tests.get("input", []), pub_tests.get("output", [])))
    hidden = list(zip(priv_tests.get("input", []), priv_tests.get("output", [])))
    if len(hidden) < 40:
        hidden += list(zip(gen_tests.get("input", []), gen_tests.get("output", [])))
    hidden = hidden[:40]

    # Extract Python 3 reference solutions (language == 3 in CodeContests)
    sols = row.get("solutions") or {}
    langs = sols.get("language", [])
    codes = sols.get("solution", [])
    py_sols = [c for lang, c in zip(langs, codes) if lang in (3, "3", "python", "python3") and c]

    cf_contest = row.get("cf_contest_id")
    contest_id = f"cf_{cf_contest}" if cf_contest else f"cc_{hashlib.sha1(pid.encode()).hexdigest()[:6]}"

    return {
        "pid": pid,
        "source": "code_contests",
        "split": split,
        "contest_id": contest_id,
        "difficulty": str(row.get("cf_rating") or row.get("difficulty") or "unknown"),
        "timeout": 6.0,
        "question": str(row.get("description") or ""),
        "public": [[str(i), str(o)] for i, o in public],
        "hidden": [[str(i), str(o)] for i, o in hidden],
        "reference_solution": py_sols[0] if py_sols else None,
        "reference_solutions": py_sols[:3],
    }


def convert_taco_row(row: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """Convert one HuggingFace `BAAI/TACO` row into SSPO problem format."""
    pid = f"taco-{split}-{idx:05d}"
    io_raw = row.get("input_output")
    inputs, outputs = [], []
    if isinstance(io_raw, str):
        try:
            parsed = json.loads(io_raw)
            inputs = [str(x) for x in parsed.get("inputs", [])]
            outputs = [str(x) for x in parsed.get("outputs", [])]
        except Exception:
            pass
    pairs = list(zip(inputs, outputs))
    public = pairs[:2] if len(pairs) >= 7 else pairs[:1]
    hidden = pairs[len(public) : len(public) + 40]
    sols_raw = row.get("solutions")
    py_sols: List[str] = []
    if isinstance(sols_raw, str):
        try:
            py_sols = [str(s) for s in json.loads(sols_raw) if s]
        except Exception:
            pass
    return {
        "pid": pid,
        "source": "taco",
        "split": split,
        "contest_id": f"taco_{idx // 10}",
        "difficulty": str(row.get("difficulty") or "unknown"),
        "timeout": 6.0,
        "question": str(row.get("question") or ""),
        "public": public,
        "hidden": hidden,
        "reference_solution": py_sols[0] if py_sols else None,
        "reference_solutions": py_sols[:3],
    }


def run_data_pipeline_on_problems(
    raw_problems: List[Dict[str, Any]],
    trajectories_by_pid: Optional[Dict[str, List[List[Dict[str, Any]]]]] = None,
) -> Dict[str, Any]:
    """Run filtering, representative program extraction, and hidden-test labeling on a list of problems.

    Returns a summary dictionary with `filter_stats`, `kept_problems`, `extracted_programs`, and `labels`.
    """
    kept, filter_stats = filter_problems(raw_problems)
    extracted: Dict[str, List[Dict[str, Any]]] = {}
    labels: Dict[str, Dict[str, Any]] = {}

    for prob in kept:
        pid = prob["pid"]
        trajs = (trajectories_by_pid or {}).get(pid)
        if trajs is None:
            # Default trajectory uses the problem's reference_solution (if any) and wrong_solutions (if any)
            turns = []
            for w_code in prob.get("wrong_solutions", [])[:1]:
                turns.append({"text": f"```python\n{w_code}\n```"})
            if prob.get("reference_solution"):
                turns.append({"text": f"```python\n{prob['reference_solution']}\n```"})
            trajs = [turns] if turns else [[{"text": "```python\nprint(0)\n```"}]]

        extracted[pid] = []
        for t_idx, turns in enumerate(trajs):
            rep = extract_representative_program(turns, prob=prob)
            extracted[pid].append(rep)
            k = rep["key"]
            label_key = f"{pid}:{k}"
            if label_key not in labels:
                labels[label_key] = label_program_on_hidden(rep["code"], prob)

    return {
        "filter_stats": filter_stats,
        "kept_problems": kept,
        "extracted_programs": extracted,
        "labels": labels,
    }


def prepare_datasets(
    out_dir: str | Path,
    source: str = "sample",
    max_train: int = 1000,
    dev_size: int = 200,
    max_test: int = 300,
    include_taco: bool = False,
) -> Dict[str, Any]:
    """Prepare `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `filter_stats.json` in `out_dir`."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if source == "sample" or os.environ.get("SSPO_OFFLINE_SAMPLE", "0") == "1":
        sample_file = REPO_ROOT / "data" / "sample_problems.jsonl"
        raw_all = read_jsonl(sample_file)
        kept, stats = filter_problems(raw_all)
        # Split kept problems into train / dev / test deterministically
        n_total = len(kept)
        n_dev = min(dev_size, max(2, n_total // 5))
        n_test = min(max_test, max(4, n_total // 4))
        train_rows = [dict(p, split="train") for p in kept[: max(1, n_total - n_dev - n_test)]]
        dev_rows = [dict(p, split="dev") for p in kept[len(train_rows) : len(train_rows) + n_test]]
        test_rows = [dict(p, split="test") for p in kept[len(train_rows) + len(dev_rows) :]]
        if not test_rows:
            test_rows = [dict(p, split="test") for p in dev_rows]
    else:
        from datasets import load_dataset

        hf_token = os.environ.get("HF_TOKEN") or None
        raw_train: List[Dict[str, Any]] = []
        raw_test: List[Dict[str, Any]] = []

        ds_train = load_dataset("deepmind/code_contests", split="train", streaming=True, token=hf_token)
        for idx, row in enumerate(ds_train):
            if idx >= (max_train + dev_size) * 2:
                break
            raw_train.append(convert_code_contests_row(row, "train", idx))

        for sp in ("valid", "test"):
            ds_eval = load_dataset("deepmind/code_contests", split=sp, streaming=True, token=hf_token)
            for idx, row in enumerate(ds_eval):
                if len(raw_test) >= max_test * 2:
                    break
                raw_test.append(convert_code_contests_row(row, "test", idx))

        if include_taco:
            ds_taco = load_dataset("BAAI/TACO", split="train", streaming=True, token=hf_token)
            for idx, row in enumerate(ds_taco):
                if idx >= 500:
                    break
                raw_train.append(convert_taco_row(row, "train", idx))

        kept_train_all, stats_train = filter_problems(raw_train)
        kept_test_all, stats_test = filter_problems(raw_test)

        # Deterministic hash shuffle so dev_size problems are held out cleanly
        kept_train_all.sort(key=lambda x: hashlib.sha256(f"sspo-v1:{x['pid']}".encode()).hexdigest())
        dev_rows = [dict(p, split="dev") for p in kept_train_all[:dev_size]]
        train_rows = [dict(p, split="train") for p in kept_train_all[dev_size : dev_size + max_train]]
        test_rows = [dict(p, split="test") for p in kept_test_all[:max_test]]

        total_raw = stats_train["total_raw"] + stats_test["total_raw"]
        total_kept = stats_train["kept"] + stats_test["kept"]
        total_rem = total_raw - total_kept
        merged_reasons = collections.Counter(stats_train["removed_by_reason"]) + collections.Counter(
            stats_test["removed_by_reason"]
        )
        stats = {
            "total_raw": total_raw,
            "kept": total_kept,
            "removed_total": total_rem,
            "removed_fraction": round(total_rem / max(1, total_raw), 4),
            "removed_by_reason": dict(merged_reasons),
        }

    stats["counts_by_split"] = {
        "train": len(train_rows),
        "dev": len(dev_rows),
        "test": len(test_rows),
    }
    atomic_write_jsonl(out_path / "train.jsonl", train_rows)
    atomic_write_jsonl(out_path / "dev.jsonl", dev_rows)
    atomic_write_jsonl(out_path / "test.jsonl", test_rows)
    atomic_write_json(out_path / "filter_stats.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare and filter SSPO datasets (CodeContests / TACO / Sample).")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "data"), help="Output data directory")
    parser.add_argument(
        "--source",
        type=str,
        choices=["code_contests", "sample"],
        default="code_contests",
        help="Dataset source ('code_contests' downloads from HF; 'sample' uses built-in offline dataset)",
    )
    parser.add_argument("--max-train", type=int, default=1000, help="Max training problems")
    parser.add_argument("--dev-size", type=int, default=200, help="Number of dev problems held out")
    parser.add_argument("--max-test", type=int, default=300, help="Max evaluation problems")
    parser.add_argument("--include-taco", action="store_true", help="Also include BAAI/TACO problems")
    args = parser.parse_args()

    stats = prepare_datasets(
        out_dir=args.out_dir,
        source=args.source,
        max_train=args.max_train,
        dev_size=args.dev_size,
        max_test=args.max_test,
        include_taco=args.include_taco,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
