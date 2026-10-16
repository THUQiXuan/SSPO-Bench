"""Integration module registering SSPO advantage estimators with `verl.trainer.ppo.core_algos`.

Provides:
1. `compute_verl_sspo_outcome_advantage(...)`: tensor-level wrapper compatible with `verl`'s
   `(token_level_rewards, response_mask, index)` signature, broadcasting `(k / n_x) * A_i` onto every
   action token `t` where `response_mask[i, t] == 1`.
2. `register_sspo_with_verl()`: registers `"sspo"`, `"drgrpo_pass1"`, `"pkpo_pass8"`, and `"checker_reward"`
   into `verl.trainer.ppo.core_algos` when `verl` is installed on the GPU machine.
"""
from __future__ import annotations

import collections
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from sspo.advantage import compute_arm_advantages


def compute_verl_sspo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[str | int],
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    sigma_plus_by_uid: Optional[Dict[str | int, Set[str]]] = None,
    arm: str = "sspo",
    k: int = 8,
    alpha: float = 0.7,
    center_select: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str | int, Set[str]]]:
    """Compute token-level advantages and returns in `verl` format `(bs, response_length)`.

    Each trajectory `i` belonging to problem group `index[i]` receives scalar advantage
    `(k / n_x) * A_i` on all action tokens (`response_mask[i, t] == 1`).
    Problems where all `A_i == 0` receive `0.0` everywhere.
    """
    bs, seq_len = response_mask.shape
    assert len(index) == bs and len(g_list) == bs and len(chk_list) == bs and len(sig_list) == bs

    groups: Dict[str | int, List[int]] = collections.defaultdict(list)
    for row_idx, uid in enumerate(index):
        groups[uid].append(row_idx)

    updated_sigma: Dict[str | int, Set[str]] = {}
    if sigma_plus_by_uid:
        for uid, sset in sigma_plus_by_uid.items():
            updated_sigma[uid] = set(sset)

    scalar_adv = np.zeros(bs, dtype=np.float32)
    for uid, rows in groups.items():
        grp_g = [int(g_list[r]) for r in rows]
        grp_chk = [int(chk_list[r]) for r in rows]
        grp_sig = [str(sig_list[r]) for r in rows]
        res = compute_arm_advantages(
            arm=arm,
            g_list=grp_g,
            chk_list=grp_chk,
            sig_list=grp_sig,
            k=k,
            alpha=alpha,
            sigma_plus=updated_sigma.get(uid, set()),
            center_select=center_select,
        )
        updated_sigma[uid] = res.updated_sigma_plus
        if not res.all_zero:
            for local_i, r_idx in enumerate(rows):
                scalar_adv[r_idx] = float(res.loss_scale * res.advantages[local_i])

    adv_tensor = (
        torch.from_numpy(scalar_adv).to(device=response_mask.device, dtype=response_mask.dtype).unsqueeze(-1)
        * response_mask
    )
    return adv_tensor, adv_tensor, updated_sigma


def register_sspo_with_verl() -> bool:
    """Register SSPO advantage estimators in `verl.trainer.ppo.core_algos` if `verl` is installed."""
    try:
        import verl.trainer.ppo.core_algos as core_algos  # type: ignore[import-not-found]
    except ImportError:
        return False

    for arm_name in ("sspo", "drgrpo_pass1", "pkpo_pass8", "checker_reward"):
        def _make_fn(a_name: str):
            def _estimator(token_level_rewards, response_mask, index, **kwargs):
                adv, ret, _ = compute_verl_sspo_outcome_advantage(
                    token_level_rewards=token_level_rewards,
                    response_mask=response_mask,
                    index=index,
                    g_list=kwargs["g_list"],
                    chk_list=kwargs["chk_list"],
                    sig_list=kwargs["sig_list"],
                    sigma_plus_by_uid=kwargs.get("sigma_plus_by_uid"),
                    arm=a_name,
                    k=int(kwargs.get("k", 8)),
                    alpha=float(kwargs.get("alpha", 0.7)),
                )
                return adv, ret

            return _estimator

        fn = _make_fn(arm_name)
        if hasattr(core_algos, "register_adv_est"):
            core_algos.register_adv_est(arm_name)(fn)
        elif hasattr(core_algos, "ADV_ESTIMATOR_REGISTRY"):
            core_algos.ADV_ESTIMATOR_REGISTRY[arm_name] = fn
        setattr(core_algos, f"compute_{arm_name}_outcome_advantage", fn)

    return True
