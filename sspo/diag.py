"""Gate 1: Training-Free Diagnostics (`run_diag.sh` / `python -m sspo.diag`).

Workflow:
1. Samples `n = 16` (or `n_rollouts` from config) rollouts per problem on up to 300 training pool problems using the
   frozen base policy model.
2. Evaluates `(g(y), Chk(y), sig(y))` using the cached execution checker.
3. Computes:
   - Selector gain (`select@8 - pass@1`, gate criterion >= 2.0 percentage points)
   - Selection gap (`cov@8 - select@8`, gate criterion >= 5.0 percentage points)
   - Critical problem rate (`exists K : J_S(K) < J_cov(K)`, gate criterion >= 10%)
   - Cluster impurity rate (`cluster_impure`)
   - False rejection rate `eps_hat` (`Chk = 0` and `g = 1`)
   - Exact agreement between Dynamic Programming (`DIFF` / `UHAT`) and 6435-subset brute-force enumeration (`<= 1e-9`)
4. Filters the training pool to active problems with `0 < pass@8 < 1` and writes `data/train_active.jsonl`,
   `results/diag/diag_problems.jsonl`, `results/diag/summary.json`, `results/tables/diagnostics.csv`, and `results/diag/done`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sspo.advantage import compute_sspo_advantages
from sspo.atomic_io import (
    REPO_ROOT,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    has_done_marker,
    read_jsonl,
    to_rel_path,
    write_done_marker,
)
from sspo.checker import ExecutionCache, extract_representative_program
from sspo.config import load_config
from sspo.eval import evaluate_problem_rollouts, summarize_eval_rows


def default_mock_rollout_generator(prob: Dict[str, Any], n: int, turns: int = 1, seed: int = 0) -> List[List[Dict[str, Any]]]:
    """Generate realistic mock rollout trajectories for a problem (mixing C, W_1, W_2, T candidates)."""
    rng = random.Random(seed + abs(hash(str(prob.get("pid", "")))) % 10007)
    ref_sol = prob.get("reference_solution") or "print(0)\n"
    wrong_sols = prob.get("wrong_solutions") or [
        "a, b = map(int, input().split())\nprint(a + b + 999)\n",
        "print('WRONG')\n",
    ]
    tle_sol = prob.get("tle_solution") or "import sys\nsys.exit(1)\n"

    trajectories: List[List[Dict[str, Any]]] = []
    # Construct a realistic mix (e.g. ~20-35% C, ~45% dominant wrong cluster W_1, ~10% W_2, ~20% T)
    for i in range(n):
        u = (i + rng.random()) / float(max(1, n))
        if u < 0.25:
            code_1 = wrong_sols[0] if turns == 2 else ref_sol
            code_2 = ref_sol
        elif u < 0.70:
            # Dominant wrong cluster W_1 (passes public examples if wrong_sols[-1] is designed so, or wrong_sols[0])
            code_1 = wrong_sols[0]
            code_2 = wrong_sols[0]
        elif u < 0.80 and len(wrong_sols) > 1:
            code_1 = wrong_sols[1]
            code_2 = wrong_sols[1]
        else:
            code_1 = tle_sol
            code_2 = tle_sol

        if turns == 1:
            trajectories.append([{"text": f"```python\n{code_2}\n```", "code": code_2}])
        else:
            trajectories.append(
                [
                    {"text": f"```python\n{code_1}\n```", "code": code_1},
                    {"text": f"```python\n{code_2}\n```", "code": code_2},
                ]
            )
    return trajectories


def run_diagnostics(
    config_path: str | Path = "configs/pilot.yaml",
    out_dir: str | Path = REPO_ROOT / "results" / "diag",
    resume: bool = True,
    mock_rollout_fn: Optional[Callable[[Dict[str, Any], int, int, int], List[List[Dict[str, Any]]]]] = None,
) -> Dict[str, Any]:
    """Execute Gate 1 Training-Free Diagnostics."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if resume and has_done_marker(out_path) and (out_path / "summary.json").exists():
        return json.loads((out_path / "summary.json").read_text(encoding="utf-8"))

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    rollout_cfg = cfg.get("rollout", {})
    eval_cfg = cfg.get("eval", {})
    checker_cfg = cfg.get("checker", {})

    train_file = REPO_ROOT / data_cfg.get("train_file", "data/train.jsonl")
    if not train_file.exists():
        train_file = REPO_ROOT / "data" / "sample_problems.jsonl"
    problems = read_jsonl(train_file)
    diag_limit = int(data_cfg.get("diag_pool_size", data_cfg.get("max_train_problems", 300)))
    diag_problems = problems[:diag_limit]

    n_rollouts = int(train_cfg.get("n_rollouts", 16))
    k_deploy = int(train_cfg.get("k_deploy", 8))
    turns = int(rollout_cfg.get("turns", 1))
    n_subsets = int(eval_cfg.get("n_subsets", 200))
    subset_seed = int(eval_cfg.get("subset_seed", 42))

    cache_root = REPO_ROOT / checker_cfg.get("cache_dir", "cache")
    cache = ExecutionCache(cache_root=cache_root, checker_version=str(checker_cfg.get("version", "v1")))

    use_mock = (
        mock_rollout_fn is not None
        or cfg.get("mode") == "smoke"
        or os.environ.get("SSPO_MOCK_LLM", "0") == "1"
    )
    rollout_fn = mock_rollout_fn or default_mock_rollout_generator

    from sspo.trainer import RolloutEngine

    engine = RolloutEngine(
        cfg,
        mock=use_mock,
        mock_fn=rollout_fn if use_mock else None,
        enable_lora=False,
    )

    partial_cache_file = out_path / "diag_rollout_cache.json"
    cached_eval_by_pid: Dict[str, Dict[str, Any]] = {}
    if resume and partial_cache_file.exists():
        try:
            cached_eval_by_pid = json.loads(partial_cache_file.read_text(encoding="utf-8"))
        except Exception:
            cached_eval_by_pid = {}

    chunk_size = 32
    k_list = [k for k in (1, 2, 4, 8, 16) if k <= n_rollouts]
    for c_start in range(0, len(diag_problems), chunk_size):
        chunk_probs = diag_problems[c_start : c_start + chunk_size]
        todo_probs = [p for p in chunk_probs if p["pid"] not in cached_eval_by_pid]
        if todo_probs:
            todo_seeds = [c_start + i for i, p in enumerate(chunk_probs) if p["pid"] not in cached_eval_by_pid]
            traj_map = engine.sample_batch_trajectories(
                todo_probs, n=n_rollouts, turns=turns, seeds=todo_seeds
            )
            eval_items = []
            for p in todo_probs:
                for t in traj_map[p["pid"]]:
                    rep = extract_representative_program(t, prob=p)
                    eval_items.append((p["pid"], rep["code"], p))
            all_evs = cache.evaluate_batch_programs(eval_items)
            ev_cursor = 0
            for p in todo_probs:
                pid = p["pid"]
                trajs = traj_map[pid]
                g_list, chk_list, sig_list = [], [], []
                for _ in trajs:
                    ev = all_evs[ev_cursor]
                    ev_cursor += 1
                    g_list.append(ev["g"])
                    chk_list.append(ev["chk"])
                    sig_list.append(ev["sig"])
                cached_eval_by_pid[pid] = {
                    "g_list": g_list,
                    "chk_list": chk_list,
                    "sig_list": sig_list,
                }
            if not use_mock:
                atomic_write_json(partial_cache_file, cached_eval_by_pid)

    eval_rows: List[Dict[str, Any]] = []
    active_train_problems: List[Dict[str, Any]] = []
    dp_vs_bf_diffs: List[float] = []

    for idx, prob in enumerate(diag_problems):
        pid = prob["pid"]
        contest_id = str(prob.get("contest_id", f"c_{idx}"))
        lists = cached_eval_by_pid[pid]
        g_list = lists["g_list"]
        chk_list = lists["chk_list"]
        sig_list = lists["sig_list"]

        erow = evaluate_problem_rollouts(
            pid=pid,
            contest_id=contest_id,
            g_list=g_list,
            chk_list=chk_list,
            sig_list=sig_list,
            k_list=k_list,
            n_subsets=n_subsets,
            subset_seed=subset_seed,
        )
        eval_rows.append(erow)

        # Verify DP vs Brute-force agreement on every cluster-pure problem
        res_dp = compute_sspo_advantages(
            g_list, chk_list, sig_list, k=min(k_deploy, n_rollouts), force_brute_force=False
        )
        res_bf = compute_sspo_advantages(
            g_list, chk_list, sig_list, k=min(k_deploy, n_rollouts), force_brute_force=True
        )
        max_diff = float(np.max(np.abs(res_dp.advantages - res_bf.advantages))) if len(g_list) > 0 else 0.0
        dp_vs_bf_diffs.append(max_diff)

        # Filter training pool to 0 < pass@k < 1
        k_str = str(min(k_deploy, n_rollouts))
        p_k = float(erow["pass_at_k"].get(k_str, 0.0))
        if 0.0 < p_k < 1.0:
            active_train_problems.append(prob)

    if len(active_train_problems) < min(4, len(diag_problems)):
        # Keep full diagnostic pool if too few problems fall strictly in (0, 1) (e.g. tiny smoke sample)
        active_train_problems = list(diag_problems)

    active_file = REPO_ROOT / "data" / "train_active.jsonl"
    atomic_write_jsonl(active_file, active_train_problems)
    atomic_write_jsonl(out_path / "diag_problems.jsonl", eval_rows)

    agg = summarize_eval_rows(eval_rows, run_name="diag")
    k_key = str(min(k_deploy, n_rollouts))
    sel_k = float(agg.get("select_at_k", {}).get(k_key, 0.0))
    cov_k = float(agg.get("cov_at_k", {}).get(k_key, 0.0))
    pass_1 = float(agg.get("pass_at_k", {}).get("1", 0.0))
    gap_k = float(agg.get("gap_at_k", {}).get(k_key, 0.0))
    crit_rate = float(agg.get("critical_problem_rate", 0.0))
    impure_rate = float(agg.get("cluster_impure_rate", 0.0))
    eps_hat = float(agg.get("eps_hat_mean", 0.0))
    max_dp_bf_diff = float(max(dp_vs_bf_diffs, default=0.0))
    selector_gain = round(sel_k - pass_1, 6)

    gate1_passed = bool(
        max_dp_bf_diff <= 1e-9
        and selector_gain >= 0.02
        and gap_k >= 0.05
        and crit_rate >= 0.10
    )

    summary = {
        "stage": "diag",
        "config": to_rel_path(config_path),
        "n_problems": len(eval_rows),
        "n_rollouts": n_rollouts,
        "k_deploy": min(k_deploy, n_rollouts),
        "pass_at_1": pass_1,
        "cov_at_8": cov_k,
        "select_at_8": sel_k,
        "selector_gain": selector_gain,
        "selection_gap_at_8": gap_k,
        "critical_problem_rate": crit_rate,
        "cluster_impure_rate": impure_rate,
        "eps_hat": eps_hat,
        "chk_pass_rate": float(agg.get("chk_pass_rate_mean", 0.0)),
        "dp_vs_bruteforce_max_diff": max_dp_bf_diff,
        "active_train_pool_count": len(active_train_problems),
        "active_train_file": to_rel_path(active_file),
        "gate1_passed": gate1_passed,
    }
    atomic_write_json(out_path / "summary.json", summary)

    # Also write `results/tables/diagnostics.csv`
    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fields = list(summary.keys())
    csv_text = ",".join(fields) + "\n" + ",".join(str(summary[f]) for f in fields) + "\n"
    atomic_write_text(tables_dir / "diagnostics.csv", csv_text)

    write_done_marker(out_path, {"n_problems": len(eval_rows), "gate1_passed": gate1_passed})
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run SSPO Gate 1 Training-Free Diagnostics.")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "diag"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock rollouts (CPU/smoke mode)")
    args = parser.parse_args()

    if args.mock:
        os.environ["SSPO_MOCK_LLM"] = "1"
    summary = run_diagnostics(
        config_path=args.config,
        out_dir=args.out_dir,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
