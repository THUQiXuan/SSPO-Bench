"""Parallel Multi-GPU Checkpoint Evaluation Worker Pool (`scripts/eval_worker_pool.py`).

Continuously scans `results/<run>/checkpoints/step_*` across all active 40,960-context runs
and evaluates any completed checkpoint (`step_0005` .. `step_0030`) on both `dev` and `test`
splits using atomic lock files (`eval/.{split}_{ckpt}.lock`) so multiple GPU workers
never duplicate work and automatically recover if a preempted worker leaves a stale lock.
Exits automatically once all expected training runs have `done` markers and all checkpoints
are evaluated on both `dev` and `test`.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sspo.atomic_io import REPO_ROOT, atomic_write_json, has_done_marker
from sspo.config import load_config
from sspo.eval import generate_tables_and_report, summarize_eval_rows
from sspo.atomic_io import read_jsonl, to_rel_path, write_done_marker
from sspo.checker import ExecutionCache, extract_representative_program
from sspo.eval import evaluate_problem_rollouts
from sspo.trainer import RolloutEngine

EXPECTED_RUNS = [
    "frozen_seed0",
    "sspo_seed0",
    "sspo_seed1",
    "sspo_seed2",
    "sspo_seed3",
    "sspo_seed4",
    "sspo_seed5",
    "sspo_seed6",
    "drgrpo_pass1_seed0",
    "drgrpo_pass1_seed1",
    "pkpo_pass8_seed0",
    "pkpo_pass8_seed1",
    "tang_maj_loo_seed0",
    "tang_maj_loo_seed1",
    "tang_maj_loo_seed2",
    "checker_reward_seed0",
    "sspo_turns2_seed0",
    "sspo_turns2_seed1",
    "sspo_turns2_seed2",
    "sspo_turns2_seed3",
    "sspo_turns2_seed4",
]

LOCK_STALE_SECONDS = 1800  # 30 minutes


def list_candidate_tasks(results_root: Path, expected_runs: List[str] | None = None) -> Tuple[List[Tuple[Path, str, str]], bool]:
    """Return `(pending_tasks, all_runs_training_done)`."""
    pending: List[Tuple[Path, str, str]] = []
    all_train_done = True
    run_list = expected_runs if expected_runs is not None else EXPECTED_RUNS

    for rname in run_list:
        rdir = results_root / rname
        if not has_done_marker(rdir):
            all_train_done = False
        ckpt_root = rdir / "checkpoints"
        if not ckpt_root.exists():
            continue
        if rname.startswith("frozen"):
            steps = ["step_0000"] if (ckpt_root / "step_0000").exists() else []
        else:
            steps = sorted(
                d.name
                for d in ckpt_root.iterdir()
                if d.is_dir() and d.name.startswith("step_") and (d / "lora_adapter" / "adapter_config.json").exists()
            )
        # Prioritize final checkpoint (step_0030 / checker_reward step_0020 / frozen step_0000) and test split first
        is_final = lambda s, rn=rname: s in ("step_0030", "step_0000") or (s == "step_0020" and rn.startswith("checker_reward"))
        steps_ordered = sorted(steps, key=lambda s: (0 if is_final(s) else 1, s))
        for split in ("test", "dev"):
            for ckpt in steps_ordered:
                eval_dir = rdir / "eval"
                done_m = eval_dir / f".{split}_{ckpt}.done"
                eval_f = eval_dir / f"{split}_{ckpt}.jsonl"
                if done_m.exists() and eval_f.exists():
                    continue
                pending.append((rdir, split, ckpt))

    pending.sort(
        key=lambda item: (
            0 if (item[2] in ("step_0030", "step_0000") or (item[2] == "step_0020" and item[0].name.startswith("checker_reward"))) else 1,
            0 if item[1] == "test" else 1,
            -int(item[2].split("_")[-1]) if item[2].split("_")[-1].isdigit() else 0,
            item[0].name,
        )
    )
    return pending, all_train_done


def try_acquire_lock(eval_dir: Path, split: str, ckpt: str, worker_id: str) -> bool:
    eval_dir.mkdir(parents=True, exist_ok=True)
    lock_file = eval_dir / f".{split}_{ckpt}.lock"
    now = time.time()
    if lock_file.exists():
        try:
            age = now - lock_file.stat().st_mtime
            if age < LOCK_STALE_SECONDS:
                return False
            lock_file.unlink()
        except OSError:
            return False
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"worker": worker_id, "host": socket.gethostname(), "ts": now}, f)
        return True
    except OSError:
        return False


def release_lock(eval_dir: Path, split: str, ckpt: str) -> None:
    lock_file = eval_dir / f".{split}_{ckpt}.lock"
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass


def evaluate_task_with_shared_engine(
    engine: RolloutEngine,
    cache: ExecutionCache,
    cfg: dict,
    rdir: Path,
    split: str,
    ckpt_name: str,
    lora_seq_id: int,
) -> dict:
    """Evaluate `(rdir, split, ckpt_name)` reusing the already-initialized vLLM `engine`."""
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    rollout_cfg = cfg.get("rollout", {})

    eval_out_dir = rdir / "eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl_path = eval_out_dir / f"{split}_{ckpt_name}.jsonl"
    eval_done_marker = eval_out_dir / f".{split}_{ckpt_name}.done"

    split_file = REPO_ROOT / data_cfg.get(f"{split}_file", f"data/{split}.jsonl")
    problems = [p for p in read_jsonl(split_file) if not p["pid"].startswith("sample-bad")]
    max_eval = int(data_cfg.get("max_eval_problems", 100))
    problems = problems[:max_eval]
    if split == "test" and problems and not problems[0].get("hidden"):
        from benchmark.hidden_test_vault import inject_hidden_tests_in_memory

        vault_rel = data_cfg.get("test_vault_file")
        vault_path = (REPO_ROOT / vault_rel) if vault_rel else (split_file.parent / "test_hidden.enc")
        problems = inject_hidden_tests_in_memory(problems, vault_path=vault_path)

    arm = rdir.name.split("_seed")[0]
    if "turns2" in rdir.name and not arm.endswith("_turns2"):
        arm = f"{arm}_turns2"

    n_eval = int(eval_cfg.get("n_eval", 16))
    k_list = [int(k) for k in eval_cfg.get("k_list", [1, 2, 4, 8, 16]) if int(k) <= n_eval]
    n_subsets = int(eval_cfg.get("n_subsets", 200))
    subset_seed = int(eval_cfg.get("subset_seed", 42))
    turns = 2 if "turns2" in rdir.name else int(rollout_cfg.get("turns", 1))

    ckpt_lora_dir = rdir / "checkpoints" / ckpt_name / "lora_adapter"
    eval_lora_path = str(ckpt_lora_dir) if (ckpt_lora_dir.exists() and arm != "frozen") else None

    partial_file = eval_out_dir / f".{split}_{ckpt_name}.partial.json"
    done_by_pid = {}
    if partial_file.exists():
        try:
            done_by_pid = json.loads(partial_file.read_text(encoding="utf-8"))
        except Exception:
            done_by_pid = {}

    chunk_size = 16
    for c_start in range(0, len(problems), chunk_size):
        chunk_probs = problems[c_start : c_start + chunk_size]
        todo_probs = [p for p in chunk_probs if p["pid"] not in done_by_pid]
        if todo_probs:
            todo_seeds = [subset_seed + c_start + i for i, p in enumerate(chunk_probs) if p["pid"] not in done_by_pid]
            traj_map = engine.sample_batch_trajectories(
                todo_probs, n=n_eval, turns=turns, seeds=todo_seeds, lora_path=eval_lora_path, lora_id=lora_seq_id
            )
            eval_items = []
            for p in todo_probs:
                for t_turns in traj_map[p["pid"]]:
                    rep = extract_representative_program(t_turns, prob=p)
                    eval_items.append((p["pid"], rep["code"], p))
            all_evs = cache.evaluate_batch_programs(eval_items)
            ev_cursor = 0
            for p in todo_probs:
                pid = p["pid"]
                contest_id = str(p.get("contest_id", f"c_{pid}"))
                trajs = traj_map[pid]
                g_list, chk_list, sig_list = [], [], []
                for _ in trajs:
                    ev = all_evs[ev_cursor]
                    ev_cursor += 1
                    g_list.append(ev["g"])
                    chk_list.append(ev["chk"])
                    sig_list.append(ev["sig"])
                done_by_pid[pid] = evaluate_problem_rollouts(
                    pid=pid,
                    contest_id=contest_id,
                    g_list=g_list,
                    chk_list=chk_list,
                    sig_list=sig_list,
                    k_list=k_list,
                    n_subsets=n_subsets,
                    subset_seed=subset_seed,
                )
            atomic_write_json(partial_file, done_by_pid)
            # Refresh lock mtime
            lock_f = eval_out_dir / f".{split}_{ckpt_name}.lock"
            if lock_f.exists():
                try:
                    os.utime(str(lock_f), None)
                except OSError:
                    pass

    eval_rows = [done_by_pid[p["pid"]] for p in problems if p["pid"] in done_by_pid]
    from sspo.atomic_io import atomic_write_jsonl

    atomic_write_jsonl(eval_jsonl_path, eval_rows)
    summary = summarize_eval_rows(eval_rows, run_name=rdir.name)
    summary["arm"] = arm
    summary["ckpt"] = ckpt_name
    summary["eval_split"] = split
    summary["eval_file"] = to_rel_path(eval_jsonl_path)

    # Update summary.json if this is the latest checkpoint on test split
    latest_ptr = rdir / "checkpoints" / "latest"
    latest_ckpt = latest_ptr.read_text(encoding="utf-8").strip() if latest_ptr.exists() else ckpt_name
    if split == "test" and ckpt_name == latest_ckpt:
        atomic_write_json(rdir / "summary.json", summary)

    write_done_marker(eval_done_marker, {"ckpt": ckpt_name, "split": split, "n_problems": len(eval_rows)})
    if partial_file.exists():
        try:
            partial_file.unlink()
        except OSError:
            pass
    try:
        generate_tables_and_report(rdir.parent)
    except Exception:
        pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel checkpoint evaluation worker.")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--expected-runs", type=str, default=None, help="Comma-separated list of expected run names")
    parser.add_argument("--worker-id", type=str, default="w0")
    args = parser.parse_args()

    os.environ["SSPO_VLLM_ONLY"] = "1"
    cfg = load_config(args.config)
    checker_cfg = cfg.get("checker", {})
    cache_root = REPO_ROOT / checker_cfg.get("cache_dir", "cache")
    cache = ExecutionCache(cache_root=cache_root, checker_version=str(checker_cfg.get("version", "v1")))

    engine: RolloutEngine | None = None
    lora_counter = 1000
    results_root = REPO_ROOT / args.results_root
    expected_runs = [x.strip() for x in args.expected_runs.split(",") if x.strip()] if args.expected_runs else None

    while True:
        pending, all_train_done = list_candidate_tasks(results_root, expected_runs=expected_runs)
        claimed = None
        for rdir, split, ckpt in pending:
            if try_acquire_lock(rdir / "eval", split, ckpt, args.worker_id):
                claimed = (rdir, split, ckpt)
                break

        if claimed is not None:
            rdir, split, ckpt = claimed
            try:
                if engine is None:
                    engine = RolloutEngine(cfg, mock=False, enable_lora=True)
                lora_counter += 1
                print(
                    f"[{args.worker_id}] Evaluating {rdir.name} split={split} ckpt={ckpt}...",
                    flush=True,
                )
                s = evaluate_task_with_shared_engine(
                    engine=engine,
                    cache=cache,
                    cfg=cfg,
                    rdir=rdir,
                    split=split,
                    ckpt_name=ckpt,
                    lora_seq_id=lora_counter,
                )
                print(
                    f"[{args.worker_id}] Done {rdir.name} {split}_{ckpt}: Select@8={s.get('select_at_k', {}).get('8')}",
                    flush=True,
                )
            finally:
                release_lock(rdir / "eval", split, ckpt)
            continue

        if all_train_done and not pending:
            print(f"[{args.worker_id}] All training runs and checkpoint evaluations completed. Exiting.", flush=True)
            break

        time.sleep(20)


if __name__ == "__main__":
    main()
