"""Reference Baseline 1: Vanilla Pass@1 Dr. GRPO (`drgrpo_pass1`)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from benchmark.agent_algorithm import GroupAdvantageOutput, GroupContext, PolicyLossBatch
from sspo.selector import classify_candidates


def compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput:
    n = ctx.n
    if n == 0:
        return GroupAdvantageOutput(advantages=np.zeros(0, dtype=np.float64), loss_scale=0.0, drop_problem=True)

    g_arr = np.array([1.0 if int(g) == 1 else 0.0 for g in ctx.g_list], dtype=np.float64)
    total_g = float(np.sum(g_arr))
    advantages = g_arr - (total_g - g_arr) / float(n - 1) if n > 1 else g_arr.copy()
    drop_problem = bool(np.all(np.abs(advantages) < 1e-12))

    sigma_plus = set(ctx.history_state.get("sigma_plus", ()))
    group = classify_candidates(ctx.g_list, ctx.chk_list, ctx.sig_list, sigma_plus=sigma_plus)
    return GroupAdvantageOutput(
        advantages=advantages,
        loss_scale=1.0 / float(n),
        drop_problem=drop_problem,
        updated_state={"sigma_plus": sorted(sigma_plus | group.c_sigs)},
        metrics={"adv_cov_mean": float(np.mean(advantages)), "adv_sel_mean": 0.0},
    )


def compute_policy_loss(batch: PolicyLossBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_ratio = torch.clamp(batch.policy_logprobs - batch.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    adv = batch.advantages.unsqueeze(-1)
    pg_token = -torch.minimum(ratio * adv, torch.clamp(ratio, 1.0 - batch.clip_eps, 1.0 + batch.clip_eps) * adv)

    log_ref_diff = torch.clamp(batch.ref_logprobs - batch.policy_logprobs, min=-20.0, max=20.0)
    kl_token = torch.exp(log_ref_diff) - log_ref_diff - 1.0

    scale = batch.loss_scales / float(max(1, batch.num_problems_in_batch) * max(1, batch.l_max))
    masked_pg = (pg_token * batch.action_mask).sum(dim=-1) * scale
    masked_kl = (kl_token * batch.action_mask).sum(dim=-1) * scale
    total_loss = (masked_pg + batch.kl_coef * masked_kl).sum()

    with torch.no_grad():
        active_tok = torch.clamp(batch.action_mask.sum(), min=1.0)
        mean_kl = float(((kl_token * batch.action_mask).sum() / active_tok).item())
        clipped = ((ratio < (1.0 - batch.clip_eps)) | (ratio > (1.0 + batch.clip_eps))).float()
        clip_frac = float(((clipped * batch.action_mask).sum() / active_tok).item())

    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "pg_loss": float(masked_pg.sum().detach().item()),
        "kl": mean_kl,
        "clip_frac": clip_frac,
    }
