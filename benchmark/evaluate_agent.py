"""Unified Agent Benchmark Runner & Scorecard Generator (`benchmark/evaluate_agent.py`).

Commands:
  1. `python -m benchmark.evaluate_agent verify`
     Runs 3-layer anti-hacking & mathematical verification on `benchmark/agent_algorithm.py` (< 3s).
  2. `python -m benchmark.evaluate_agent smoke`
     Runs verification + fast 4-step end-to-end training & evaluation smoke test (< 5s on CPU).
  3. `python -m benchmark.evaluate_agent train [--seed 0] [--run-dir results/agent_seed0]`
     Verifies locks & algorithm, then runs the locked 30-step RL training on `data/train.jsonl`.
  4. `python -m benchmark.evaluate_agent eval-dev [--run-dir results/agent_seed0]`
     Evaluates saved checkpoints on the held-out Validation Set (`data/dev.jsonl`, 100 problems)
     so the agent can iterate on its RL algorithm and select checkpoints without touching the test set.
  5. `python -m benchmark.evaluate_agent eval-test [--run-dir results/agent_seed0]`
     Evaluates the trained run on the encrypted Hidden Test Set (`data/test.jsonl` + `data/test_hidden.enc`)
     and generates `results/agent_seed0/benchmark_scorecard.json` comparing against all baselines.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from benchmark.verifier import run_full_verification
from sspo.atomic_io import REPO_ROOT, atomic_write_json, read_jsonl
from sspo.eval import summarize_eval_rows
from sspo.trainer import run_evaluation, run_training

LOCKED_CONFIG = REPO_ROOT / "configs" / "benchmark_locked.yaml"
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
BASELINES_JSON = REPO_ROOT / "leaderboard" / "baselines.json"


def _list_saved_checkpoints(run_dir: Path) -> List[str]:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return []
    steps = sorted(
        d.name for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("step_")
    )
    return steps


def evaluate_all_checkpoints_on_split(
    run_dir: Path,
    config_path: Path,
    split: str = "dev",
    mock: bool = False,
) -> Dict[str, Any]:
    """Evaluate all saved `step_*` checkpoints in `run_dir` on `split` (`dev` or `test`)."""
    steps = _list_saved_checkpoints(run_dir)
    if not steps:
        steps = ["latest"]

    per_ckpt: Dict[str, Dict[str, Any]] = {}
    for ckpt in steps:
        summary = run_evaluation(
            run_dir=run_dir,
            config_path=config_path,
            ckpt=ckpt,
            eval_split=split,
            resume=True,
            mock=mock,
        )
        per_ckpt[ckpt] = {
            "ckpt": ckpt,
            "pass_at_1": summary.get("pass_at_k", {}).get("1", 0.0),
            "pass_at_8": summary.get("pass_at_k", {}).get("8", 0.0),
            "cov_at_8": summary.get("cov_at_k", {}).get("8", 0.0),
            "select_at_8": summary.get("select_at_k", {}).get("8", 0.0),
            "gap_at_8": summary.get("gap_at_k", {}).get("8", 0.0),
            "chk_pass_rate": summary.get("chk_pass_rate_mean", 0.0),
            "cluster_impure_rate": summary.get("cluster_impure_rate", 0.0),
        }

    best_by_select8 = max(per_ckpt.values(), key=lambda x: x["select_at_8"])
    best_by_chk = max(per_ckpt.values(), key=lambda x: x["chk_pass_rate"])
    final_ckpt = per_ckpt[steps[-1]]

    report = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
        "split": split,
        "best_by_select_at_8": best_by_select8,
        "best_by_chk_pass_rate": best_by_chk,
        "final_checkpoint": final_ckpt,
        "checkpoints": per_ckpt,
    }
    atomic_write_json(run_dir / f"{split}_progression.json", report)
    return report


def build_benchmark_scorecard(
    run_dir: Path,
    verification: Dict[str, Any],
    test_report: Dict[str, Any],
    dev_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compare agent's test performance against official baselines in `leaderboard/baselines.json`."""
    baselines_data = json.loads(BASELINES_JSON.read_text(encoding="utf-8")) if BASELINES_JSON.exists() else {}
    baselines_list = baselines_data.get("baselines", [])

    test_ckpts = test_report.get("checkpoints", {})
    oracle_best = test_report["best_by_select_at_8"]
    chk_selected_step = test_report["best_by_chk_pass_rate"]["ckpt"]
    chk_selected_test = test_ckpts.get(chk_selected_step, oracle_best)
    final_test = test_report["final_checkpoint"]

    if dev_report and dev_report.get("best_by_select_at_8"):
        val_best_step = dev_report["best_by_select_at_8"]["ckpt"]
        val_selected_test = test_ckpts.get(val_best_step, chk_selected_test)
    else:
        val_selected_test = chk_selected_test

    comparisons = {}
    for b in baselines_list:
        b_arm = b["arm"]
        b_best_sel8 = b["best_ckpt"]["select_at_8"]
        b_final_sel8 = b["final_ckpt"]["select_at_8"]
        comparisons[b_arm] = {
            "baseline_best_select_at_8": b_best_sel8,
            "baseline_final_select_at_8": b_final_sel8,
            "delta_vs_baseline_best_pt": round((oracle_best["select_at_8"] - b_best_sel8) * 100.0, 2),
            "delta_vs_baseline_final_pt": round((final_test["select_at_8"] - b_final_sel8) * 100.0, 2),
            "beats_baseline_best": bool(oracle_best["select_at_8"] > b_best_sel8),
            "beats_baseline_final": bool(final_test["select_at_8"] > b_final_sel8),
        }

    scorecard = {
        "benchmark": "SSPO-Bench v1.0",
        "verification_passed": verification["passed"],
        "algorithm_sha256": verification.get("algorithm_sha256"),
        "agent_results": {
            "val_selected_test": val_selected_test,
            "chk_selected_test": chk_selected_test,
            "final_checkpoint_test": final_test,
            "oracle_best_test": oracle_best,
        },
        "vs_baselines": comparisons,
    }
    atomic_write_json(run_dir / "benchmark_scorecard.json", scorecard)
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="SSPO-Bench Agent Evaluation & Verification CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify", help="Run 3-layer verification on benchmark/agent_algorithm.py")
    p_ver.add_argument("--skip-locks", action="store_true", help="Skip Layer 1 repo lock check")

    p_smoke = sub.add_parser("smoke", help="Run verification + fast 4-step CPU smoke test of agent algorithm")
    p_smoke.add_argument("--run-dir", type=str, default=str(REPO_ROOT / "results" / "agent_smoke"))

    p_train = sub.add_parser("train", help="Run locked 30-step RL training with benchmark/agent_algorithm.py")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--run-dir", type=str, default=None)
    p_train.add_argument("--turns", type=int, choices=[1, 2], default=1)
    p_train.add_argument("--max-steps", type=int, default=None)

    p_dev = sub.add_parser("eval-dev", help="Evaluate agent checkpoints on Validation Set (data/dev.jsonl)")
    p_dev.add_argument("--run-dir", type=str, default=str(REPO_ROOT / "results" / "agent_seed0"))
    p_dev.add_argument("--mock", action="store_true")

    p_test = sub.add_parser("eval-test", help="Evaluate agent checkpoints on Hidden Test Set and produce scorecard")
    p_test.add_argument("--run-dir", type=str, default=str(REPO_ROOT / "results" / "agent_seed0"))
    p_test.add_argument("--mock", action="store_true")

    args = parser.parse_args()

    if args.cmd == "verify":
        ver = run_full_verification(check_locks=not args.skip_locks)
        print(json.dumps(ver, indent=2, ensure_ascii=False))
        if not ver["passed"]:
            raise SystemExit(1)

    elif args.cmd == "smoke":
        ver = run_full_verification(check_locks=True)
        if not ver["passed"]:
            print(json.dumps(ver, indent=2, ensure_ascii=False))
            raise SystemExit(1)
        rdir = Path(args.run_dir)
        if rdir.exists():
            shutil.rmtree(rdir)
        train_res = run_training(
            config_path=SMOKE_CONFIG,
            arm="agent",
            seed=0,
            run_dir=rdir,
            max_steps=4,
            resume=False,
            mock=True,
        )
        dev_rep = evaluate_all_checkpoints_on_split(rdir, SMOKE_CONFIG, split="dev", mock=True)
        test_rep = evaluate_all_checkpoints_on_split(rdir, SMOKE_CONFIG, split="test", mock=True)
        scorecard = build_benchmark_scorecard(rdir, ver, test_rep, dev_rep)
        print(
            json.dumps(
                {
                    "status": "smoke_passed",
                    "verification": ver["passed"],
                    "train": train_res,
                    "scorecard": scorecard,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    elif args.cmd == "train":
        ver = run_full_verification(check_locks=True)
        if not ver["passed"]:
            print(json.dumps(ver, indent=2, ensure_ascii=False))
            raise SystemExit(1)
        rdir = Path(args.run_dir) if args.run_dir else (REPO_ROOT / "results" / f"agent_seed{args.seed}")
        res = run_training(
            config_path=LOCKED_CONFIG,
            arm="agent",
            seed=args.seed,
            run_dir=rdir,
            turns=args.turns,
            max_steps=args.max_steps,
            resume=True,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.cmd == "eval-dev":
        ver = run_full_verification(check_locks=True)
        if not ver["passed"]:
            print(json.dumps(ver, indent=2, ensure_ascii=False))
            raise SystemExit(1)
        rdir = Path(args.run_dir)
        rep = evaluate_all_checkpoints_on_split(rdir, LOCKED_CONFIG, split="dev", mock=args.mock)
        print(json.dumps(rep, indent=2, ensure_ascii=False))

    elif args.cmd == "eval-test":
        ver = run_full_verification(check_locks=True)
        if not ver["passed"]:
            print(json.dumps(ver, indent=2, ensure_ascii=False))
            raise SystemExit(1)
        rdir = Path(args.run_dir)
        dev_file = rdir / "dev_progression.json"
        dev_rep = json.loads(dev_file.read_text(encoding="utf-8")) if dev_file.exists() else None
        test_rep = evaluate_all_checkpoints_on_split(rdir, LOCKED_CONFIG, split="test", mock=args.mock)
        scorecard = build_benchmark_scorecard(rdir, ver, test_rep, dev_rep)
        print(json.dumps(scorecard, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
