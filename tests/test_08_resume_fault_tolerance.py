"""Test 08: Resume & Fault-Tolerance Verification.

Section 5.8 requirement:
Run step 1 and force an interruption; restart the exact same command (`resume=True`), and assert:
1. It resumes from step 2 (does not re-run step 1).
2. Execution cache hits (`cache_hit_rate > 0`).
3. `metrics.jsonl` has no duplicate or corrupted step entries (even if a stale step-2 line was left behind before crash).
4. Re-running after completion (`done` marker exists) is an immediate no-op (`status == "already_done"`).
"""
from __future__ import annotations

import json
from pathlib import Path

from sspo.atomic_io import read_jsonl
from sspo.trainer import run_training


def test_kill_after_step1_and_resume_step2(tmp_path: Path):
    run_dir = tmp_path / "sspo_resume_test"

    # 1. Run Step 1 and force interruption after Step 1 (`stop_after_step=1` out of `max_steps=2`)
    res1 = run_training(
        config_path="configs/smoke.yaml",
        arm="sspo",
        seed=0,
        run_dir=run_dir,
        max_steps=2,
        stop_after_step=1,
        resume=True,
        mock=True,
    )
    assert res1["status"] == "interrupted_for_resume_test"
    assert res1["last_step"] == 1
    assert res1["steps_run"] == 1
    assert not (run_dir / "done").exists()
    assert (run_dir / "checkpoints" / "latest").read_text(encoding="utf-8").strip() == "step_0001"

    rows_after_1 = read_jsonl(run_dir / "metrics.jsonl")
    assert len(rows_after_1) == 1
    assert rows_after_1[0]["step"] == 1

    # Simulate a crash that left a partial/stale step 2 row in metrics.jsonl before checkpointing
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 2, "loss": 999.0, "stale": True}) + "\n")
        f.write("{truncated_json_line_from_kill")

    # 2. Restart the same command (`max_steps=2, resume=True`)
    res2 = run_training(
        config_path="configs/smoke.yaml",
        arm="sspo",
        seed=0,
        run_dir=run_dir,
        max_steps=2,
        resume=True,
        mock=True,
    )
    assert res2["status"] == "completed"
    assert res2["last_step"] == 2
    assert res2["steps_run"] == 1, "Must only run Step 2 upon resume!"
    assert res2["cache_hit_rate"] > 0.5, "Execution cache must hit on resumed run!"
    assert (run_dir / "done").exists()
    assert (run_dir / "checkpoints" / "latest").read_text(encoding="utf-8").strip() == "step_0002"

    # Verify metrics.jsonl has exactly 2 clean rows (step 1 and step 2, with stale row replaced)
    rows_after_2 = read_jsonl(run_dir / "metrics.jsonl")
    assert [r["step"] for r in rows_after_2] == [1, 2]
    assert "stale" not in rows_after_2[1]

    # 3. Re-running a completed run is a safe no-op
    res3 = run_training(
        config_path="configs/smoke.yaml",
        arm="sspo",
        seed=0,
        run_dir=run_dir,
        max_steps=2,
        resume=True,
        mock=True,
    )
    assert res3["status"] == "already_done"
