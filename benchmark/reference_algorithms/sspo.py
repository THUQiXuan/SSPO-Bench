"""Reference Baseline 4: Selection-Aware Policy Optimization (`sspo`, alpha = 0.7)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from benchmark.agent_algorithm import GroupAdvantageOutput, GroupContext, PolicyLossBatch
from sspo.dp import brute_force_selection_advantages, comb_exact, dp_selection_advantages
from sspo.selector import classify_candidates


def compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput:
    n, k = ctx.n, ctx.k
    alpha = 0.7
    if n == 0 or k < 1 or k > n:
        return GroupAdvantageOutput(advantages=np.zeros(n, dtype=np.float64), loss_scale=0.0, drop_problem=True)

    sigma_plus = set(ctx.history_state.get("sigma_plus", ()))
    group = classify_candidates(ctx.g_list, ctx.chk_list, ctx.sig_list, sigma_plus=sigma_plus)

    if group.cluster_impure:
        adv_sel, _, _ = brute_force_selection_advantages(
            ctx.g_list, ctx.chk_list, ctx.sig_list, k=k, sigma_plus=sigma_plus
        )
    else:
        adv_sel = dp_selection_advantages(group, k=k)

    is_c = [1 if (int(ctx.g_list[i]) == 1 and int(ctx.chk_list[i]) == 1) else 0 for i in range(n)]
    n_c = sum(is_c)
    denom_km1 = comb_exact(n - 1, k - 1)
    denom_k = comb_exact(n - 1, k)

    adv_cov_centered = np.zeros(n, dtype=np.float64)
    for i in range(n):
        f_i = is_c[i]
        a_cov = (float(comb_exact(n - n_c, k - 1)) / float(denom_km1)) if (f_i == 1 and denom_km1 > 0) else 0.0
        n_prime_c = n_c - f_i
        m_wrong = (n - 1) - n_prime_c
        v_km1 = 1.0 - (float(comb_exact(m_wrong, k - 1)) / float(denom_km1)) if denom_km1 > 0 else 0.0
        v_k = 1.0 - (float(comb_exact(m_wrong, k)) / float(denom_k)) if denom_k > 0 else 1.0
        adv_cov_centered[i] = a_cov - (v_k - v_km1)

    advantages = (1.0 - alpha) * adv_cov_centered + alpha * adv_sel
    drop_problem = bool(np.all(np.abs(advantages) < 1e-12))

    return GroupAdvantageOutput(
        advantages=advantages,
        loss_scale=float(k) / float(n),
        drop_problem=drop_problem,
        updated_state={"sigma_plus": sorted(sigma_plus | group.c_sigs)},
        metrics={
            "adv_cov_mean": float(np.mean(adv_cov_centered)),
            "adv_sel_mean": float(np.mean(adv_sel)),
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
