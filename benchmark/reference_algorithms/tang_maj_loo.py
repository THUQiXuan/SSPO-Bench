"""Reference Baseline 5: Tang et al. (Meta, 2503.19595) Majority-Vote Leave-One-Out with Mean Centering (`tang_maj_loo`).

Implements the best-performing biased leave-one-out (LOO) estimator from:
  Tang et al., "Optimizing Language Models for Inference Time Objectives using Reinforcement Learning" (arXiv:2503.19595)
applied to the test-time selector `S` (public example filter + execution signature majority vote):
  1. Reward scale in `{-1, +1}`: `f(Y) = 2 * J_S(Y) - 1`
  2. Single-set leave-one-out marginal credit (Eq. 2):
       `A_i = f(Y) - f(Y_{-i}) = 2 * (J_S(Y) - J_S(Y \\ {i}))`
  3. Mean-centered (biased) advantage variant (Eq. 4, best reported variant in Tang et al.):
       `A'_i = A_i - (1 / n) * sum_{j=1}^n A_j`
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from benchmark.agent_algorithm import GroupAdvantageOutput, GroupContext, PolicyLossBatch
from sspo.selector import classify_candidates, j_s_subset


def compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput:
    n, k = ctx.n, ctx.k
    if n <= 1:
        return GroupAdvantageOutput(advantages=np.zeros(n, dtype=np.float64), loss_scale=0.0, drop_problem=True)

    sigma_plus = set(ctx.history_state.get("sigma_plus", ()))
    group = classify_candidates(ctx.g_list, ctx.chk_list, ctx.sig_list, sigma_plus=sigma_plus)

    full_indices = list(range(n))
    f_full = 2.0 * j_s_subset(full_indices, ctx.g_list, ctx.chk_list, ctx.sig_list) - 1.0

    a_raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        minus_i = [j for j in range(n) if j != i]
        f_minus_i = 2.0 * j_s_subset(minus_i, ctx.g_list, ctx.chk_list, ctx.sig_list) - 1.0
        a_raw[i] = f_full - f_minus_i

    # Biased variant Eq. (4) in Tang et al. (2503.19595): A'_i = A_i - mean_j A_j
    advantages = a_raw - float(np.mean(a_raw))
    drop_problem = bool(np.all(np.abs(advantages) < 1e-12))

    return GroupAdvantageOutput(
        advantages=advantages,
        loss_scale=float(k) / float(n),
        drop_problem=drop_problem,
        updated_state={"sigma_plus": sorted(sigma_plus | group.c_sigs)},
        metrics={
            "tang_f_full": float(f_full),
            "tang_raw_l1": float(np.mean(np.abs(a_raw))),
        },
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
