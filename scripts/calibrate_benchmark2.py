"""Parallel 40,960-Context Dataset Calibration & Split Builder for Benchmark 2 (`scripts/calibrate_benchmark2.py`).

Constructs a second, strictly disjoint benchmark (`data_b2/` and `results_b2/`) with ZERO problem overlap
with Benchmark 1 (`data/{train,dev,test}.jsonl`):
- Draws candidate problems from:
  1. `exec-select-20261009` (`lcb-confirm-v1-atcoder-*`, 160 AtCoder LiveCodeBench problems: 80 medium, 80 hard)
  2. `exec-select-trainjudge-20261010` (`100000..100180`, 155 AtCoder LiveCodeBench problems: 49 medium, 106 hard)
  3. Disjoint USACO (`bronze`, `silver`, `gold`), LeetCode (`medium`, `hard`), Codeforces (`cfo`), and APPS-CF (`cf2000-2100`)
     problems not included in Benchmark 1.
- Evaluates uncalibrated candidate problems with frozen `Qwen/Qwen3-4B` under the native `40,960` context window
  (`max_tokens=8192`, `n=16` rollouts) across parallel GPU shards, reusing existing 40,960-context calibration
  records where available.
- Constructs `contest_id`-disjoint splits in `data_b2/`:
  - `data_b2/train.jsonl` & `data_b2/train_active.jsonl`
  - `data_b2/dev.jsonl` (100 validation problems)
  - `data_b2/test.jsonl` + `data_b2/test_hidden.enc` (100 sealed test problems)
- Populates `results_b2/diag/` and `results_b2/frozen_seed0/` (`dev` & `test` step_0000).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

DATA_B2_DIR = REPO_ROOT / "data_b2"
RAW_POOL_B2_FILE = DATA_B2_DIR / "raw_pool_b2.jsonl"
CALIB_B1_DIR = REPO_ROOT / "results" / "calibration_40k"
CALIB_B2_DIR = REPO_ROOT / "results_b2" / "calibration_40k"

EXCLUDED_TIERS_B2 = {"platinum", "cf2200", "cf2300", "cf2400", "cf2500", "cf2600"}


def _canonical_qid(prob: Dict[str, Any]) -> str:
    pid = str(prob.get("pid", ""))
    qid = prob.get("qid")
    if qid:
        return str(qid)
    if "atcoder-" in pid:
        return pid.split("atcoder-")[-1]
    if "leetcode-" in pid:
        return pid.split("leetcode-")[-1]
    return pid


def load_b2_candidate_pool() -> List[Dict[str, Any]]:
    """Build or load the strictly disjoint Benchmark 2 candidate pool (`data_b2/raw_pool_b2.jsonl`)."""
    DATA_B2_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_POOL_B2_FILE.exists():
        rows = read_jsonl(RAW_POOL_B2_FILE)
        if rows:
            return rows

    b1_pids = set()
    b1_qids = set()
    for s in ("train", "dev", "test"):
        p = REPO_ROOT / "data" / f"{s}.jsonl"
        if p.exists():
            for r in read_jsonl(p):
                b1_pids.add(str(r["pid"]))
                b1_qids.add(_canonical_qid(r))

    sources = [
        ("lcb_confirm", "/gscratch/simondu/xuanqi26/exec-select-20261009/data/lcb_main.json"),
        ("lcb_judge", "/gscratch/simondu/xuanqi26/exec-select-trainjudge-20261010/data/lcb_main.json"),
        ("usaco", "/gscratch/simondu/xuanqi26/exec-select-usaco-20261010/data/lcb_main.json"),
        ("leetcode", "/gscratch/simondu/xuanqi26/exec-select-lc-20261009/data/lcb_main.json"),
        ("cfo", "/gscratch/simondu/xuanqi26/exec-select-cfo-20261010/data/lcb_main.json"),
        ("lcb_n92", "/gscratch/simondu/xuanqi26/exec-select-n92-20261009/data/lcb_main.json"),
        ("apps_cf", "/gscratch/simondu/xuanqi26/exec-select-cf-20261010/data/lcb_main.json"),
    ]

    cases_dir = REPO_ROOT / "cache" / "checkers" / "cases"
    seen_pids = set()
    seen_qids = set()
    pool: List[Dict[str, Any]] = []

    for src_tag, json_path in sources:
        fp = Path(json_path)
        if not fp.exists():
            continue
        raw_list = json.loads(fp.read_text(encoding="utf-8"))
        for r in raw_list:
            pid = str(r["pid"])
            qid = _canonical_qid(r)
            if pid in b1_pids or qid in b1_qids or pid in seen_pids or qid in seen_qids:
                continue
            diff = str(r.get("difficulty") or "unknown").replace("lcb-", "")
            if diff in EXCLUDED_TIERS_B2:
                continue
            if not (cases_dir / f"{pid}.json").exists():
                continue

            cid = r.get("contest_id")
            if not cid:
                if "_" in qid:
                    cid = qid.split("_")[0]
                else:
                    cid = f"{src_tag}_{pid}"
            cid = f"{src_tag}:{cid}"

            norm = {
                "pid": pid,
                "qid": qid,
                "contest_id": str(cid),
                "difficulty": diff,
                "platform": str(r.get("platform") or src_tag),
                "timeout": float(r.get("timeout", 6.0)),
                "question": str(r.get("question", "")),
                "public": r.get("public", []),
                "hidden_file": r.get("hidden_file"),
                "reference_solution": r.get("reference_solution"),
            }
            if "prompt" in r and r["prompt"]:
                norm["prompt"] = r["prompt"]
            seen_pids.add(pid)
            seen_qids.add(qid)
            pool.append(norm)

    atomic_write_jsonl(RAW_POOL_B2_FILE, pool)
    return pool


def _load_existing_b1_calib() -> Dict[str, Dict[str, Any]]:
    calib: Dict[str, Dict[str, Any]] = {}
    if CALIB_B1_DIR.exists():
        for sf in sorted(CALIB_B1_DIR.glob("shard_*.json")):
            try:
                calib.update(json.loads(sf.read_text(encoding="utf-8")))
            except Exception:
                pass
    return calib


def run_shard(config_path: str, shard_id: int, num_shards: int) -> Dict[str, Any]:
    """Run one GPU shard of 40,960-context evaluation on `uncalibrated_pool[shard_id::num_shards]`."""
    from sspo.checker import ExecutionCache, extract_representative_program
    from sspo.config import load_config
    from sspo.eval import evaluate_problem_rollouts
    from sspo.trainer import RolloutEngine

    CALIB_B2_DIR.mkdir(parents=True, exist_ok=True)
    done_file = CALIB_B2_DIR / f"shard_{shard_id:02d}.done"
    partial_file = CALIB_B2_DIR / f"shard_{shard_id:02d}.json"

    pool = load_b2_candidate_pool()
    b1_calib = _load_existing_b1_calib()

    # Only shard the problems not already calibrated in CALIB_B1_DIR so GPU shards are evenly balanced!
    uncalib_pool = [p for p in pool if p["pid"] not in b1_calib]
    shard_probs = uncalib_pool[shard_id::num_shards]

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
    active_mid: List[Tuple[float, Dict[str, Any]]] = []
    reach_cands: List[Tuple[float, Dict[str, Any]]] = []
    near_top: List[Tuple[float, Dict[str, Any]]] = []

    diff_pref = {
        "medium": 3.0,
        "silver": 2.9,
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
        "hard": 1.4,
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
            mid_bonus = 1.0 - abs(p1 - 0.35)
            score = gap8 * 10.0 + (0.5 if is_crit else 0.0) + mid_bonus + 0.1 * d_score
            active_mid.append((score, p))
        elif p16 == 0.0:
            r_score = chk_rate * 5.0 + d_score
            reach_cands.append((r_score, p))

    active_mid.sort(key=lambda x: -x[0])
    reach_cands.sort(key=lambda x: -x[0])

    max_active = min(len(active_mid), max(32, int(target_n * 0.55)))
    chosen = [p for _, p in active_mid[:max_active]]
    needed = target_n - len(chosen)
    if needed > 0:
        chosen.extend([p for _, p in reach_cands[:needed]])
    if len(chosen) < target_n:
        rem_active = [p for _, p in active_mid[max_active:]]
        chosen.extend(rem_active[: (target_n - len(chosen))])
    if len(chosen) < target_n:
        chosen.extend([p for _, p in near_top[: (target_n - len(chosen))]])

    return chosen[:target_n]


def finalize_splits(config_path: str) -> Dict[str, Any]:
    """Aggregate all calibrated Benchmark 2 problems, build `contest_id`-disjoint `train`, `dev`, `test` splits
    in `data_b2/`, seal `data_b2/test_hidden.enc`, and write `results_b2/diag/` and `results_b2/frozen_seed0/`.
    """
    import numpy as np
    from benchmark.hidden_test_vault import Seal_test_split
    from sspo.advantage import compute_sspo_advantages
    from sspo.config import load_config
    from sspo.eval import generate_tables_and_report, summarize_eval_rows

    pool = load_b2_candidate_pool()
    eval_by_pid = _load_existing_b1_calib()
    for sf in sorted(CALIB_B2_DIR.glob("shard_*.json")):
        try:
            eval_by_pid.update(json.loads(sf.read_text(encoding="utf-8")))
        except Exception:
            pass

    # Update contest_id inside eval_by_pid to match pool's normalized contest_id
    calibrated_pool = []
    for p in pool:
        pid = p["pid"]
        if pid in eval_by_pid:
            er = dict(eval_by_pid[pid])
            er["contest_id"] = str(p["contest_id"])
            eval_by_pid[pid] = er
            calibrated_pool.append(p)

    # Group by contest_id and partition contests across train / dev / test so all three splits have balanced
    # active problems and selection gap, with zero shared contest_ids
    by_cid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in calibrated_pool:
        by_cid[str(p["contest_id"])].append(p)

    def _is_active(prob: Dict[str, Any]) -> bool:
        er = eval_by_pid[prob["pid"]]
        p1 = float(er["pass_at_k"].get("1", 0.0))
        cov16 = float(er["cov_at_k"].get("16", 0.0))
        return cov16 > 0.0 and 0.0 < p1 < 0.999

    # Sort contests by (n_active, total_gap8, deterministic hash)
    cid_records = []
    for cid, plist in by_cid.items():
        n_act = sum(1 for p in plist if _is_active(p))
        tot_gap = sum(float(eval_by_pid[p["pid"]]["gap_at_k"].get("8", 0.0)) for p in plist)
        h = hashlib.sha256(f"b2-split:{cid}".encode()).hexdigest()
        cid_records.append((n_act > 0, round(tot_gap, 4), n_act, h, cid, plist))

    cid_records.sort(key=lambda x: (-int(x[0]), -x[1], -x[2], x[3]))

    train_pool: List[Dict[str, Any]] = []
    dev_pool: List[Dict[str, Any]] = []
    test_pool: List[Dict[str, Any]] = []
    act_counts = [0, 0, 0]  # [train, dev, test] target ~ 49 : 32 : 32
    gap_sums = [0.0, 0.0, 0.0]

    for has_act, tot_gap, n_act, _, cid, plist in cid_records:
        if has_act:
            # Target shares: train=49/113, dev=32/113, test=32/113; balance both active count and gap sum!
            targets = [49.0, 32.0, 32.0]
            scores = []
            for i in range(3):
                if act_counts[i] >= targets[i] and any(act_counts[j] < targets[j] for j in range(3)):
                    scores.append(-1e9)
                else:
                    act_ratio = act_counts[i] / targets[i]
                    gap_ratio = gap_sums[i] / targets[i]
                    scores.append(-(act_ratio + 2.0 * gap_ratio))
            best_idx = int(max(range(3), key=lambda i: (scores[i], -i)))
            [train_pool, dev_pool, test_pool][best_idx].extend(plist)
            act_counts[best_idx] += n_act
            gap_sums[best_idx] += tot_gap
        else:
            smallest = min(range(3), key=lambda i: len([train_pool, dev_pool, test_pool][i]))
            [train_pool, dev_pool, test_pool][smallest].extend(plist)

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
        if cov16 > 0.0 and p1 < 0.999:
            score = gap8 * 12.0 + (0.6 if is_crit else 0.0) + (1.0 - abs(p1 - 0.30))
            active_train_scored.append((score, p))
        elif p16 == 0.0:
            reach_train_scored.append((chk, p))

    active_train_scored.sort(key=lambda x: -x[0])
    reach_train_scored.sort(key=lambda x: -x[0])

    calibrated_train = [p for _, p in active_train_scored[:48]]
    needed_train = max(0, 64 - len(calibrated_train))
    reach_to_add = [p for _, p in reach_train_scored[:needed_train]]

    interleaved_train: List[Dict[str, Any]] = []
    r_idx = 0
    for idx_a, p_a in enumerate(calibrated_train):
        interleaved_train.append(p_a)
        if (idx_a + 1) % 3 == 0 and r_idx < len(reach_to_add):
            interleaved_train.append(reach_to_add[r_idx])
            r_idx += 1
    while r_idx < len(reach_to_add):
        interleaved_train.append(reach_to_add[r_idx])
        r_idx += 1

    calibrated_dev = _select_eval_100(dev_pool, eval_by_pid, target_n=64)
    calibrated_test = _select_eval_100(test_pool, eval_by_pid, target_n=64)

    DATA_B2_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(DATA_B2_DIR / "train.jsonl", interleaved_train)
    atomic_write_jsonl(DATA_B2_DIR / "train_active.jsonl", interleaved_train)
    atomic_write_jsonl(DATA_B2_DIR / "dev.jsonl", calibrated_dev)
    atomic_write_jsonl(DATA_B2_DIR / "test.jsonl", calibrated_test)

    # Seal `data_b2/test.jsonl` into `data_b2/test_hidden.enc`
    Seal_test_split(
        test_jsonl_path=DATA_B2_DIR / "test.jsonl",
        vault_path=DATA_B2_DIR / "test_hidden.enc",
    )

    results_b2_root = REPO_ROOT / "results_b2"
    diag_dir = results_b2_root / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_rows = []
    dp_vs_bf_diffs = []
    for p in interleaved_train:
        er = dict(eval_by_pid[p["pid"]])
        er.pop("difficulty", None)
        diag_rows.append(er)
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
    gate1_passed = bool(max_dp_bf <= 1e-9 and sel_gain >= 0.01 and gap_8 >= 0.01)
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
        "active_train_file": "data_b2/train_active.jsonl",
        "gate1_passed": gate1_passed,
    }
    atomic_write_json(diag_dir / "summary.json", diag_summary)
    tables_dir = results_b2_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fields = list(diag_summary.keys())
    atomic_write_text(
        tables_dir / "diagnostics.csv",
        ",".join(fields) + "\n" + ",".join(str(diag_summary[f]) for f in fields) + "\n",
    )
    write_done_marker(diag_dir, {"n_problems": len(diag_rows), "gate1_passed": gate1_passed})

    cfg = load_config(config_path)
    preset = cfg["hardware_preset"]
    frozen_dir = results_b2_root / "frozen_seed0"
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

    # Note: Do not copy calibration-conditioned rollouts into `frozen_seed0/eval/`.
    # Leaving `frozen_seed0/eval/` empty ensures `scripts/eval_worker_pool.py` evaluates
    # `step_0000` using the exact same paired evaluation rollout seeds (`subset_seed + c_start + i`)
    # as all trained checkpoints (`step_0005..0030`), avoiding Winner's Curse selection bias.

    dev_sum = summarize_eval_rows([eval_by_pid[p["pid"]] for p in calibrated_dev], run_name="dev")
    test_sum = summarize_eval_rows([eval_by_pid[p["pid"]] for p in calibrated_test], run_name="test")
    report_out = {
        "status": "calibrated_b2_40k",
        "total_disjoint_pool": len(calibrated_pool),
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
    atomic_write_json(CALIB_B2_DIR / "calibration_summary.json", report_out)
    return report_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate 40,960-context Benchmark 2 splits on Qwen/Qwen3-4B.")
    parser.add_argument("--config", type=str, default="configs/pilot_b2.yaml")
    parser.add_argument("--mode", choices=["prepare", "shard", "finalize"], required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "prepare":
        pool = load_b2_candidate_pool()
        b1_calib = _load_existing_b1_calib()
        uncalib = [p for p in pool if p["pid"] not in b1_calib]
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "total_b2_candidates": len(pool),
                    "already_calibrated": len(pool) - len(uncalib),
                    "uncalibrated_to_run": len(uncalib),
                },
                indent=2,
            )
        )
    elif args.mode == "shard":
        res = run_shard(args.config, shard_id=args.shard_id, num_shards=args.num_shards)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.mode == "finalize":
        res = finalize_splits(args.config)
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
