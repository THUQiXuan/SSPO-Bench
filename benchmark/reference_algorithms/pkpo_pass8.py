"""Reference Baseline 2: PKPO Pass@8 Coverage-Aware Policy Optimization (`pkpo_pass8`)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from benchmark.agent_algorithm import GroupAdvantageOutput, GroupContext, PolicyLossBatch
from sspo.dp import comb_exact
from sspo.selector import classify_candidates


def compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput:
    n, k = ctx.n, ctx.k
    if n == 0 or k < 1 or k > n:
        return GroupAdvantageOutput(advantages=np.zeros(n, dtype=np.float64), loss_scale=0.0, drop_problem=True)

    flags = [1 if int(g) == 1 else 0 for g in ctx.g_list]
    n_c = sum(flags)
    denom_km1 = comb_exact(n - 1, k - 1)
    denom_k = comb_exact(n - 1, k)

    adv_cov = np.zeros(n, dtype=np.float64)
    b_cov = np.zeros(n, dtype=np.float64)
    for i in range(n):
        f_i = flags[i]
        if f_i == 1 and denom_km1 > 0:
            adv_cov[i] = float(comb_exact(n - n_c, k - 1)) / float(denom_km1)
        n_prime_c = n_c - f_i
        m_wrong = (n - 1) - n_prime_c
        v_km1 = 1.0 - (float(comb_exact(m_wrong, k - 1)) / float(denom_km1)) if denom_km1 > 0 else 0.0
        v_k = 1.0 - (float(comb_exact(m_wrong, k)) / float(denom_k)) if denom_k > 0 else 1.0
        b_cov[i] = v_k - v_km1

    advantages = adv_cov - b_cov
    drop_problem = bool(np.all(np.abs(advantages) < 1e-12))

    sigma_plus = set(ctx.history_state.get("sigma_plus", ()))
    group = classify_candidates(ctx.g_list, ctx.chk_list, ctx.sig_list, sigma_plus=sigma_plus)
    return GroupAdvantageOutput(
        advantages=advantages,
        loss_scale=float(k) / float(n),
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
