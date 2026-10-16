"""Test 07: Training Loop Dry-Run on CPU (2 steps, mock rollouts, advantage -> loss -> backward -> checkpoint).

Section 5.7 requirement:
Run 2 training steps on CPU using a tiny transformer (`TinyCausalLM`), replacing rollouts & execution
with mock/cached evaluation, and assert advantages, loss, parameter updates, checkpoints, `manifest.json`,
and `metrics.jsonl` schema; also test the `verl` outcome advantage tensor wrapper (`sspo/verl_patch.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from sspo.atomic_io import read_jsonl
from sspo.trainer import run_evaluation, run_training
from sspo.verl_patch import compute_verl_sspo_outcome_advantage


def test_training_loop_dryrun_all_arms_and_2turns(tmp_path: Path):
    for arm in ("sspo", "drgrpo_pass1", "pkpo_pass8", "checker_reward", "frozen"):
        run_dir = tmp_path / f"{arm}_seed0"
        res = run_training(
            config_path="configs/smoke.yaml",
            arm=arm,
            seed=0,
            run_dir=run_dir,
            turns=2 if arm == "sspo" else 1,
            max_steps=2,
            resume=False,
            mock=True,
        )
        assert res["status"] == "completed"
        assert (run_dir / "done").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "checkpoints" / "latest").exists()

        # Verify no absolute paths in manifest.json
        manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
        assert "/Users/" not in manifest_text

        if arm != "frozen":
            rows = read_jsonl(run_dir / "metrics.jsonl")
            assert len(rows) == 2
            assert rows[0]["step"] == 1
            assert rows[1]["step"] == 2
            for r in rows:
                assert isinstance(r["loss"], float)
                assert r["grad_norm"] >= 0.0
                assert "counts" in r and "adv_stats" in r
                if arm == "sspo":
                    assert r["adv_stats"]["neg_wrong_viol"] == 0

            ckpt_latest = (run_dir / "checkpoints" / "latest").read_text(encoding="utf-8").strip()
            assert ckpt_latest == "step_0002"
            assert (run_dir / "checkpoints" / "step_0002" / "model.pt").exists()
            assert (run_dir / "checkpoints" / "step_0002" / "optimizer.pt").exists()

        # Also run evaluation on the run_dir
        eval_summary = run_evaluation(
            run_dir=run_dir,
            config_path="configs/smoke.yaml",
            ckpt="latest",
            mock=True,
        )
        assert eval_summary["n_problems"] == 8
        assert (run_dir / "summary.json").exists()


def test_verl_patch_tensor_wrapper():
    bs, seq_len = 8, 12
    token_rewards = torch.zeros((bs, seq_len), dtype=torch.float32)
    response_mask = torch.ones((bs, seq_len), dtype=torch.float32)
    response_mask[:, :3] = 0.0  # first 3 tokens are prompt
    index = ["prob_1"] * 4 + ["prob_2"] * 4
    g_list = [1, 0, 0, 0, 1, 1, 0, 0]
    chk_list = [1, 1, 1, 0, 1, 1, 1, 0]
    sig_list = ["sC", "sW", "sW", "sT", "sC", "sC", "sW", "sT"]

    adv_tensor, ret_tensor, updated_sigma = compute_verl_sspo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=index,
        g_list=g_list,
        chk_list=chk_list,
        sig_list=sig_list,
        arm="sspo",
        k=2,
        alpha=0.7,
    )
    assert adv_tensor.shape == (bs, seq_len)
    assert torch.all(adv_tensor[:, :3] == 0.0)
    assert torch.all(adv_tensor[0, 3:] > 0.0)  # C candidate gets positive advantage
    assert torch.all(adv_tensor[1, 3:] <= 0.0)  # g=0 candidate gets non-positive advantage
    assert "sC" in updated_sigma["prob_1"]
