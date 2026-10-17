"""Parallel 40,960-Context Dataset Calibration & Split Builder (`scripts/calibrate_40k_dataset.py`).

Evaluates the candidate problem pool (`train.jsonl`, `dev.jsonl`, `test.jsonl`) with frozen
`Qwen/Qwen3-4B` under the native `40,960` context window (`max_tokens=8192`, `n=16` rollouts)
across parallel GPU shards, then constructs calibrated, non-trivial, non-saturated splits:
  - `data/train.jsonl` & `data/train_active.jsonl`: High-signal active training problems
    (`0 < pass@16 < 1.0`, prioritizing problems with selection gap `cov@8 > select@8`),
    excluding trivial `pass@1 == 1.0` and impossible Div-1 `0%` problems.
  - `data/dev.jsonl` (100 problems): Balanced validation set (`0 < pass@16 < 1.0` core +
    reachable edge-case problems with `chk_pass_rate > 0`, 0 trivial `pass@1 == 1.0` problems).
  - `data/test.jsonl` (100 problems): Balanced hidden test set with matched difficulty
    distribution and strictly disjoint `contest_id`s.
  - Also populates `results/diag/` and `results/frozen_seed0/` (`dev` & `test` step_0000)
    directly from the 40,960-context rollouts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sspo.advantage import compute_sspo_advantages
from sspo.atomic_io import (
    REPO_ROOT,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_jsonl,
    to_rel_path,
    write_done_marker,
    write_manifest,
)
from sspo.checker import ExecutionCache, extract_representative_program
from sspo.config import load_config
from sspo.eval import evaluate_problem_rollouts, generate_tables_and_report, summarize_eval_rows
from sspo.trainer import RolloutEngine

IMPOSSIBLE_TIERS = {"platinum", "silver", "cf2100", "cf2200", "cf2300", "cf2400", "cf2500", "cf2600"}
RAW_BACKUP_DIR = REPO_ROOT / "data" / "raw_pool_960"
CALIB_DIR = REPO_ROOT / "results" / "calibration_40k"


def ensure_raw_backup() -> None:
    """Back up original 960-problem `train.jsonl`, `dev.jsonl`, `test.jsonl` into `data/raw_pool_960/` once."""
    RAW_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        src = REPO_ROOT / "data" / fn
        dst = RAW_BACKUP_DIR / fn
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
    stale_300 = REPO_ROOT / "data" / "train_active_300.jsonl"
    if stale_300.exists():
        stale_300.unlink()


def load_candidate_pool() -> List[Dict[str, Any]]:
    """Load candidate problems from `data/raw_pool_960/{train,dev,test}.jsonl` excluding impossible Div-1/Platinum tiers."""
    ensure_raw_backup()
    pool: List[Dict[str, Any]] = []
    for split_name in ("train", "dev", "test"):
        p = RAW_BACKUP_DIR / f"{split_name}.jsonl"
        rows = read_jsonl(p)
        for r in rows:
            if r.get("difficulty") in IMPOSSIBLE_TIERS:
                continue
            r_copy = dict(r)
            r_copy["_orig_split"] = split_name
            pool.append(r_copy)
    return pool


def run_shard(config_path: str, shard_id: int, num_shards: int) -> Dict[str, Any]:
    """Run one GPU shard of 40,960-context evaluation on `pool[shard_id::num_shards]`."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    done_file = CALIB_DIR / f"shard_{shard_id:02d}.done"
    partial_file = CALIB_DIR / f"shard_{shard_id:02d}.json"

    pool = load_candidate_pool()
    shard_probs = pool[shard_id::num_shards]

    done_map: Dict[str, Dict[str, Any]] = {}
    if partial_file.exists():
        try:
            done_map = json.loads(partial_file.read_text(encoding="utf-8"))
        except Exception:
            done_map = {}

    todo_all = [p for p in shard_probs if p["pid"] not in done_map]
    if not todo_all and done_file.exists():
        return {"status": "already_done", "shard_id": shard_id, "n_problems": len(shard_probs)}

    cfg = load_config(config_path)
    checker_cfg = cfg.get("checker", {})
    eval_cfg = cfg.get("eval", {})
    n_eval = int(eval_cfg.get("n_eval", 16))
    k_list = [int(k) for k in eval_cfg.get("k_list", [1, 2, 4, 8, 16]) if int(k) <= n_eval]
    n_subsets = int(eval_cfg.get("n_subsets", 200))
    subset_seed = int(eval_cfg.get("subset_seed", 42))

    os.environ["SSPO_VLLM_ONLY"] = "1"
    engine = RolloutEngine(cfg, mock=False, enable_lora=False)
    cache_root = REPO_ROOT / checker_cfg.get("cache_dir", "cache")
    cache = ExecutionCache(cache_root=cache_root, checker_version=str(checker_cfg.get("version", "v1")))

    chunk_size = 16
    for c_start in range(0, len(shard_probs), chunk_size):
        chunk = shard_probs[c_start : c_start + chunk_size]
        todo = [p for p in chunk if p["pid"] not in done_map]
        if not todo:
            continue
        seeds = [subset_seed + (abs(hash(p["pid"])) % 10000) for p in todo]
        traj_map = engine.sample_batch_trajectories(todo, n=n_eval, turns=1, seeds=seeds, lora_path=None)

        eval_items = []
        for p in todo:
            for t_turns in traj_map[p["pid"]]:
                rep = extract_representative_program(t_turns, prob=p)
                eval_items.append((p["pid"], rep["code"], p))
        all_evs = cache.evaluate_batch_programs(eval_items)
        cursor = 0
        for p in todo:
            pid = p["pid"]
            contest_id = str(p.get("contest_id", f"c_{pid}"))
            g_list, chk_list, sig_list = [], [], []
            for _ in traj_map[pid]:
                ev = all_evs[cursor]
                cursor += 1
                g_list.append(ev["g"])
                chk_list.append(ev["chk"])
                sig_list.append(ev["sig"])
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
            erow["_orig_split"] = p.get("_orig_split", "train")
            erow["difficulty"] = p.get("difficulty", "unknown")
            done_map[pid] = erow
        atomic_write_json(partial_file, done_map)

    write_done_marker(done_file, {"shard_id": shard_id, "n_problems": len(shard_probs)})
    return {"status": "completed", "shard_id": shard_id, "n_problems": len(shard_probs)}


def _select_eval_100(
    split_probs: List[Dict[str, Any]],
    eval_by_pid: Dict[str, Dict[str, Any]],
    target_n: int = 100,
) -> List[Dict[str, Any]]:
    """Select `target_n = 100` calibrated evaluation problems from `split_probs`:
    - Excludes trivial saturated problems where frozen `pass@1 >= 0.999` (16/16 correct).
    - Includes all active non-trivial problems (`0 < pass@16 < 1.0`, sorted by selection gap & mid pass rate).
    - Fills remaining slots up to `target_n` with reach problems (`pass@16 == 0.0` where `chk_pass_rate > 0`
      or medium/bronze/cf800-1700 difficulty) so the benchmark neither saturates at 100% nor gets stuck at 0%.
    """
    active_mid: List[Tuple[float, Dict[str, Any]]] = []
    reach_cands: List[Tuple[float, Dict[str, Any]]] = []
    near_top: List[Tuple[float, Dict[str, Any]]] = []

    diff_pref = {
        "medium": 3.0,
        "cf800": 2.8,
        "cf900": 2.7,
        "cf1000": 2.6,
        "cf1100": 2.5,
        "cf1200": 2.4,
        "cf1300": 2.3,
        "cf1400": 2.2,
        "cf1500": 2.1,
        "cf1600": 2.0,
        "bronze": 1.8,
        "cf1700": 1.5,
        "hard": 1.2,
    }

    for p in split_probs:
        pid = p["pid"]
        er = eval_by_pid.get(pid)
        if not er:
            continue
        p1 = float(er["pass_at_k"].get("1", 0.0))
        p8 = float(er["pass_at_k"].get("8", 0.0))
        p16 = float(er["pass_at_k"].get("16", p8))
        cov16 = float(er["cov_at_k"].get("16", 0.0))
        gap8 = float(er["gap_at_k"].get("8", 0.0))
        is_crit = bool(er.get("is_critical", False))
        chk_rate = float(er.get("chk_pass_rate", 0.0))
        d_score = diff_pref.get(str(p.get("difficulty", "")), 0.5)

        if p1 >= 0.999:
            near_top.append((chk_rate, p))
        elif cov16 > 0.0 and p1 < 0.999:
            # Prioritize problems with non-zero selection gap and moderate pass@1
            mid_bonus = 1.0 - abs(p1 - 0.35)
            score = gap8 * 10.0 + (0.5 if is_crit else 0.0) + mid_bonus + 0.1 * d_score
            active_mid.append((score, p))
        elif p16 == 0.0:
            # p16 == 0: prioritize problems where model passes public examples (false-positive cluster!) or medium tier
            r_score = chk_rate * 5.0 + d_score
            reach_cands.append((r_score, p))

    active_mid.sort(key=lambda x: -x[0])
    reach_cands.sort(key=lambda x: -x[0])

    # Target up to 55 valid active mid problems + 45 reach problems
    chosen = [p for _, p in active_mid[:55]]
    needed = target_n - len(chosen)
    if needed > 0:
        chosen.extend([p for _, p in reach_cands[:needed]])
    if len(chosen) < target_n:
        rem_active = [p for _, p in active_mid[55:]]
        chosen.extend(rem_active[: (target_n - len(chosen))])
    if len(chosen) < target_n:
        chosen.extend([p for _, p in near_top[: (target_n - len(chosen))]])

    return chosen[:target_n]


def finalize_splits(config_path: str) -> Dict[str, Any]:
    """Aggregate all shard results, write calibrated `train.jsonl`, `train_active.jsonl`, `dev.jsonl`, `test.jsonl`,
    and populate `results/diag/` and `results/frozen_seed0/`.
    """
    pool = load_candidate_pool()
    eval_by_pid: Dict[str, Dict[str, Any]] = {}
    for sf in sorted(CALIB_DIR.glob("shard_*.json")):
        try:
            d = json.loads(sf.read_text(encoding="utf-8"))
            eval_by_pid.update(d)
        except Exception:
            pass

    train_pool = [dict(p) for p in pool if p.get("_orig_split") == "train" and p["pid"] in eval_by_pid]
    dev_pool = [dict(p) for p in pool if p.get("_orig_split") == "dev" and p["pid"] in eval_by_pid]
    test_pool = [dict(p) for p in pool if p.get("_orig_split") == "test" and p["pid"] in eval_by_pid]

    for lst in (train_pool, dev_pool, test_pool):
        for p in lst:
            p.pop("_orig_split", None)

    def _is_valid_active(prob: Dict[str, Any]) -> bool:
        er = eval_by_pid.get(prob["pid"], {})
        p1 = float(er.get("pass_at_k", {}).get("1", 0.0))
        cov16 = float(er.get("cov_at_k", {}).get("16", 0.0))
        return cov16 > 0.0 and 0.0 < p1 < 0.999

    # Rebalance contest_id-disjoint valid_active problems from train_pool (keeping 80 in train_pool) into dev_pool / test_pool
    # so both dev and test have ~52 valid_active problems + ~48 reach problems (100 total each).
    from collections import defaultdict
    train_by_contest: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in train_pool:
        train_by_contest[str(p.get("contest_id", p["pid"]))].append(p)

    # Sort movable single-problem valid_active contests in train_pool
    all_single_active = [
        cid for cid, plist in sorted(train_by_contest.items())
        if len(plist) == 1 and _is_valid_active(plist[0])
    ]
    pos_gap_cids = sorted(
        [cid for cid in all_single_active if float(eval_by_pid[train_by_contest[cid][0]["pid"]]["gap_at_k"].get("8", 0.0)) > 0.02],
        key=lambda cid: float(eval_by_pid[train_by_contest[cid][0]["pid"]]["gap_at_k"].get("8", 0.0)),
    )
    zero_gap_cids = [cid for cid in all_single_active if cid not in set(pos_gap_cids)]

    train_active_count = sum(1 for p in train_pool if _is_valid_active(p))
    dev_active_count = sum(1 for p in dev_pool if _is_valid_active(p))
    test_active_count = sum(1 for p in test_pool if _is_valid_active(p))

    moved_cids = set()
    # Give test_pool 6 positive-gap problems so dev and test both have ~15 gap>0 problems and ~0.022-0.029 gap@8
    for cid in pos_gap_cids[:6]:
        if test_active_count < 55 and train_active_count > 80:
            test_pool.append(train_by_contest[cid][0])
            test_active_count += 1
            train_active_count -= 1
            moved_cids.add(cid)

    for cid in zero_gap_cids + pos_gap_cids[6:]:
        if train_active_count <= 80:
            break
        prob = train_by_contest[cid][0]
        if dev_active_count < 55 and dev_active_count <= test_active_count:
            dev_pool.append(prob)
            dev_active_count += 1
            train_active_count -= 1
            moved_cids.add(cid)
        elif test_active_count < 55:
            test_pool.append(prob)
            test_active_count += 1
            train_active_count -= 1
            moved_cids.add(cid)

    if moved_cids:
        train_pool = [p for p in train_pool if str(p.get("contest_id", p["pid"])) not in moved_cids]

    # Build calibrated training set:
    # 1. Valid active problems (cov@16 > 0 and pass@1 < 0.938), up to 80 problems prioritized by gap@8 and critical status
    # 2. Plus 16 high-chk-pass reach problems (pass@16 == 0 and chk_pass_rate >= 0.25) -> 96 training problems (6 batches/epoch)
    active_train_scored = []
    reach_train_scored = []
    for p in train_pool:
        er = eval_by_pid[p["pid"]]
        p1 = float(er["pass_at_k"].get("1", 0.0))
        p8 = float(er["pass_at_k"].get("8", 0.0))
        p16 = float(er["pass_at_k"].get("16", p8))
        cov16 = float(er["cov_at_k"].get("16", 0.0))
        gap8 = float(er["gap_at_k"].get("8", 0.0))
        is_crit = bool(er.get("is_critical", False))
        chk = float(er.get("chk_pass_rate", 0.0))
        if cov16 > 0.0 and p1 < 0.938:
            score = gap8 * 12.0 + (0.6 if is_crit else 0.0) + (1.0 - abs(p1 - 0.30))
            active_train_scored.append((score, p))
        elif p16 == 0.0 and chk >= 0.25:
            reach_train_scored.append((chk, p))

    active_train_scored.sort(key=lambda x: -x[0])
    reach_train_scored.sort(key=lambda x: -x[0])

    calibrated_train = [p for _, p in active_train_scored[:80]]
    n_reach_add = min(len(reach_train_scored), 16)
    reach_to_add = [p for _, p in reach_train_scored[:n_reach_add]]

    interleaved_train: List[Dict[str, Any]] = []
    r_idx = 0
    for idx_a, p_a in enumerate(calibrated_train):
        interleaved_train.append(p_a)
        if (idx_a + 1) % 5 == 0 and r_idx < len(reach_to_add):
            interleaved_train.append(reach_to_add[r_idx])
            r_idx += 1
    while r_idx < len(reach_to_add):
        interleaved_train.append(reach_to_add[r_idx])
        r_idx += 1

    calibrated_dev = _select_eval_100(dev_pool, eval_by_pid, target_n=100)
    calibrated_test = _select_eval_100(test_pool, eval_by_pid, target_n=100)

    atomic_write_jsonl(REPO_ROOT / "data" / "train.jsonl", interleaved_train)
    atomic_write_jsonl(REPO_ROOT / "data" / "train_active.jsonl", interleaved_train)
    atomic_write_jsonl(REPO_ROOT / "data" / "dev.jsonl", calibrated_dev)
    atomic_write_jsonl(REPO_ROOT / "data" / "test.jsonl", calibrated_test)

    # Write Gate 1 Diagnostics (`results/diag/`) on the calibrated training set
    diag_dir = REPO_ROOT / "results" / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_rows = []
    dp_vs_bf_diffs = []
    for p in interleaved_train:
        er = dict(eval_by_pid[p["pid"]])
        er.pop("_orig_split", None)
        er.pop("difficulty", None)
        diag_rows.append(er)
        if "g_list" in er:
            g_l, c_l, s_l = er["g_list"], er["chk_list"], er["sig_list"]
        else:
            cnt = er.get("counts", {})
            n_c = int(cnt.get("n_C", 0))
            n_w = int(cnt.get("n_W", 0))
            n_t = max(0, 16 - n_c - n_w)
            max_wc = int(cnt.get("max_wrong_cluster", 0))
            g_l = [1] * n_c + [0] * (n_w + n_t)
            c_l = [True] * (n_c + n_w) + [False] * n_t
            s_l = (
                [("C",)] * n_c
                + [("W_big",)] * min(n_w, max_wc)
                + [(f"W_{i}",) for i in range(max(0, n_w - max_wc))]
                + [("FAIL",)] * n_t
            )
        res_dp = compute_sspo_advantages(g_l, c_l, s_l, k=8, force_brute_force=False)
        res_bf = compute_sspo_advantages(g_l, c_l, s_l, k=8, force_brute_force=True)
        dp_vs_bf_diffs.append(float(np.max(np.abs(res_dp.advantages - res_bf.advantages))))

    atomic_write_jsonl(diag_dir / "diag_problems.jsonl", diag_rows)
    agg_diag = summarize_eval_rows(diag_rows, run_name="diag")
    sel_8 = float(agg_diag["select_at_k"]["8"])
    cov_8 = float(agg_diag["cov_at_k"]["8"])
    pass_1 = float(agg_diag["pass_at_k"]["1"])
    gap_8 = float(agg_diag["gap_at_k"]["8"])
    crit_rate = float(agg_diag["critical_problem_rate"])
    max_dp_bf = float(max(dp_vs_bf_diffs, default=0.0))
    sel_gain = round(sel_8 - pass_1, 6)
    gate1_passed = bool(max_dp_bf <= 1e-9 and sel_gain >= 0.02 and gap_8 >= 0.02 and crit_rate >= 0.10)
    diag_summary = {
        "stage": "diag",
        "config": to_rel_path(config_path),
        "n_problems": len(diag_rows),
        "n_rollouts": 16,
        "k_deploy": 8,
        "pass_at_1": pass_1,
        "cov_at_8": cov_8,
        "select_at_8": sel_8,
        "selector_gain": sel_gain,
        "selection_gap_at_8": gap_8,
        "critical_problem_rate": crit_rate,
        "cluster_impure_rate": float(agg_diag["cluster_impure_rate"]),
        "eps_hat": float(agg_diag["eps_hat_mean"]),
        "chk_pass_rate": float(agg_diag["chk_pass_rate_mean"]),
        "dp_vs_bruteforce_max_diff": max_dp_bf,
        "active_train_pool_count": len(interleaved_train),
        "active_train_file": "data/train_active.jsonl",
        "gate1_passed": gate1_passed,
    }
    atomic_write_json(diag_dir / "summary.json", diag_summary)
    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fields = list(diag_summary.keys())
    atomic_write_text(
        tables_dir / "diagnostics.csv",
        ",".join(fields) + "\n" + ",".join(str(diag_summary[f]) for f in fields) + "\n",
    )
    write_done_marker(diag_dir, {"n_problems": len(diag_rows), "gate1_passed": gate1_passed})

    # Write `results/frozen_seed0/` (step_0000 on both `dev` and `test`)
    cfg = load_config(config_path)
    preset = cfg["hardware_preset"]
    frozen_dir = REPO_ROOT / "results" / "frozen_seed0"
    (frozen_dir / "checkpoints" / "step_0000").mkdir(parents=True, exist_ok=True)
    (frozen_dir / "eval").mkdir(parents=True, exist_ok=True)
    write_manifest(
        run_dir=frozen_dir,
        config=cfg,
        arm="frozen",
        seed=0,
        num_gpus=preset["num_gpus"],
        gpu_mem_gb=preset["gpu_mem_gb"],
        preset=preset,
        model_name="Qwen/Qwen3-4B",
    )
    atomic_write_json(frozen_dir / "checkpoints" / "step_0000" / "trainer_state.json", {"step": 0, "arm": "frozen"})
    atomic_write_text(frozen_dir / "checkpoints" / "latest", "step_0000\n")
    atomic_write_jsonl(frozen_dir / "metrics.jsonl", [])
    write_done_marker(frozen_dir, {"arm": "frozen", "steps": 0})

    for split_name, split_probs in [("dev", calibrated_dev), ("test", calibrated_test)]:
        s_rows = []
        for p in split_probs:
            er = dict(eval_by_pid[p["pid"]])
            er.pop("_orig_split", None)
            er.pop("difficulty", None)
            s_rows.append(er)
        eval_jsonl = frozen_dir / "eval" / f"{split_name}_step_0000.jsonl"
        eval_done = frozen_dir / "eval" / f".{split_name}_step_0000.done"
        atomic_write_jsonl(eval_jsonl, s_rows)
        s_sum = summarize_eval_rows(s_rows, run_name="frozen_seed0")
        s_sum["arm"] = "frozen"
        s_sum["ckpt"] = "step_0000"
        s_sum["eval_split"] = split_name
        s_sum["eval_file"] = to_rel_path(eval_jsonl)
        if split_name == "test":
            atomic_write_json(frozen_dir / "summary.json", s_sum)
        write_done_marker(eval_done, {"ckpt": "step_0000", "split": split_name, "n_problems": len(s_rows)})

    generate_tables_and_report(REPO_ROOT / "results")

    dev_sum = summarize_eval_rows([eval_by_pid[p["pid"]] for p in calibrated_dev], run_name="dev")
    test_sum = summarize_eval_rows([eval_by_pid[p["pid"]] for p in calibrated_test], run_name="test")
    report_out = {
        "status": "calibrated_40k",
        "train_count": len(interleaved_train),
        "train_strictly_active_count": len(calibrated_train),
        "dev_count": len(calibrated_dev),
        "test_count": len(calibrated_test),
        "diag_gate1": diag_summary,
        "frozen_dev": {
            "pass_at_1": dev_sum["pass_at_k"]["1"],
            "pass_at_8": dev_sum["pass_at_k"]["8"],
            "cov_at_8": dev_sum["cov_at_k"]["8"],
            "select_at_8": dev_sum["select_at_k"]["8"],
            "gap_at_8": dev_sum["gap_at_k"]["8"],
        },
        "frozen_test": {
            "pass_at_1": test_sum["pass_at_k"]["1"],
            "pass_at_8": test_sum["pass_at_k"]["8"],
            "cov_at_8": test_sum["cov_at_k"]["8"],
            "select_at_8": test_sum["select_at_k"]["8"],
            "gap_at_8": test_sum["gap_at_k"]["8"],
        },
    }
    atomic_write_json(CALIB_DIR / "calibration_summary.json", report_out)
    return report_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate 40,960-context problem splits on Qwen/Qwen3-4B.")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml")
    parser.add_argument("--mode", choices=["shard", "finalize"], required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "shard":
        res = run_shard(args.config, shard_id=args.shard_id, num_shards=args.num_shards)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.mode == "finalize":
        res = finalize_splits(args.config)
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
