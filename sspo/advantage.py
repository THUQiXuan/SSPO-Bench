"""Advantage estimators for SSPO and baseline arms (`sspo`, `drgrpo_pass1`, `pkpo_pass8`, `checker_reward`).

Pure Python / NumPy module (completely independent of training frameworks):
- Coverage leave-one-out advantage & baseline:
    `A_i^cov = 1[i in C] * C(n - n_C, k - 1) / C(n - 1, k - 1)`
    `b_i^cov = V(k) - V(k - 1)`, `V(j) = 1 - C(n - 1 - n'_C, j) / C(n - 1, j)`
- PKPO binary closed form (cross-tested with `(k / n) * A_i^cov`):
    `A_i^PKPO = f_i * C(n - c, k - 1) / C(n, k)`
- Selector leave-one-out advantage `A_i^S` (uncentered by default to preserve Theorem 2(d1) anti-hacking):
    `A_i^S = DIFF(C)` for `i in C`, `DIFF(W_m)` for `i in W_m`, `-UHAT(W*, k - 1)` for `i in W*`, `0` for `i in T`.
- Combined SSPO advantage:
    `A_i = (1 - alpha) * (A_i^cov - b_i^cov) + alpha * A_i^S`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from sspo.dp import (
    UHAT,
    brute_force_selection_advantages,
    comb_exact,
    dp_selection_advantages,
)
from sspo.selector import ClassifiedGroup, classify_candidates


@dataclass
class AdvantageResult:
    """Complete advantage computation result for one problem's `n` rollouts."""

    advantages: np.ndarray  # shape (n,), per-sample advantage A_i
    loss_scale: float  # problem-level factor k / n_x in Dr. GRPO loss
    adv_cov_raw: np.ndarray  # A_i^(cov, U)
    b_cov: np.ndarray  # b_i^cov = V(k) - V(k-1)
    adv_cov_centered: np.ndarray  # A_i^(cov, U) - b_i^cov
    adv_sel_raw: np.ndarray  # uncentered A_i^(S, U)
    b_sel: np.ndarray  # b_i^S = UHAT(X, k) - UHAT(X, k-1) (0 when uncentered)
    adv_sel_used: np.ndarray  # A_i^S used in total advantage
    pkpo_closed_form: np.ndarray  # f_i * C(n - c, k - 1) / C(n, k)
    group: ClassifiedGroup
    used_brute_force: bool
    all_zero: bool
    updated_sigma_plus: Set[str]


def coverage_advantages(
    is_c_list: Sequence[int],
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute `(A_i^cov, b_i^cov, A_i^cov - b_i^cov)` for binary indicators `is_c_list`."""
    n = len(is_c_list)
    adv_cov = np.zeros(n, dtype=np.float64)
    b_cov = np.zeros(n, dtype=np.float64)
    if n == 0 or k < 1 or k > n:
        return adv_cov, b_cov, adv_cov - b_cov

    flags = [1 if int(x) == 1 else 0 for x in is_c_list]
    n_c = sum(flags)
    denom_km1 = comb_exact(n - 1, k - 1)
    denom_k = comb_exact(n - 1, k)

    # Precompute for i in C (where n'_C = n_c - 1) and i not in C (where n'_C = n_c)
    for i in range(n):
        f_i = flags[i]
        if f_i == 1 and denom_km1 > 0:
            adv_cov[i] = float(comb_exact(n - n_c, k - 1)) / float(denom_km1)
        else:
            adv_cov[i] = 0.0

        n_prime_c = n_c - f_i
        m_wrong = (n - 1) - n_prime_c
        # V(j) = 1 - C(n - 1 - n'_C, j) / C(n - 1, j)
        v_km1 = 1.0 - (float(comb_exact(m_wrong, k - 1)) / float(denom_km1)) if denom_km1 > 0 else 0.0
        v_k = 1.0 - (float(comb_exact(m_wrong, k)) / float(denom_k)) if denom_k > 0 else 1.0
        b_cov[i] = v_k - v_km1

    return adv_cov, b_cov, adv_cov - b_cov


def pkpo_binary_advantages(
    is_correct: Sequence[int],
    k: int,
) -> np.ndarray:
    """PKPO binary closed-form advantage:
    `A_i^PKPO = f_i * C(n - c, k - 1) / C(n, k)` for `f_i = 1`, and `0` for `f_i = 0`.
    Note: `(k / n) * A_i^cov == A_i^PKPO` identically because `C(n, k) = (n / k) * C(n - 1, k - 1)`.
    """
    n = len(is_correct)
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or k < 1 or k > n:
        return out
    flags = [1 if int(x) == 1 else 0 for x in is_correct]
    c = sum(flags)
    denom = comb_exact(n, k)
    if denom == 0:
        return out
    val = float(comb_exact(n - c, k - 1)) / float(denom)
    for i, f_i in enumerate(flags):
        if f_i == 1:
            out[i] = val
    return out


def selection_loo_baselines(group: ClassifiedGroup, k: int) -> np.ndarray:
    """Compute leave-one-out selection baselines `b_i^S = UHAT(X_i, k) - UHAT(X_i, k-1)`."""
    n = group.n
    b_s = np.zeros(n, dtype=np.float64)
    if n == 0 or k < 1 or k >= n:
        return b_s
    n_C = group.n_C
    n_T = group.n_T
    n_W_list = group.n_W_clusters
    cache: Dict[Tuple[str, int], float] = {}

    for i in range(n):
        cls_name, m_idx = group.candidate_class[i]
        key = (cls_name, n_W_list[m_idx] if cls_name == "W" else 0)
        if key not in cache:
            x_spec = ("W", m_idx) if cls_name == "W" else cls_name
            uk = UHAT(x_spec, k, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_list)
            ukm1 = UHAT(x_spec, k - 1, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_list)
            cache[key] = uk - ukm1
        b_s[i] = cache[key]
    return b_s


def compute_sspo_advantages(
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k: int = 8,
    alpha: float = 0.7,
    sigma_plus: Optional[Iterable[str]] = None,
    center_select: str = "none",
    force_brute_force: bool = False,
) -> AdvantageResult:
    """Compute SSPO advantages for one problem's `n` rollouts."""
    n = len(g_list)
    sigma_plus_set = set(sigma_plus) if sigma_plus is not None else set()
    group = classify_candidates(g_list, chk_list, sig_list, sigma_plus=sigma_plus_set)

    # Step 6-8: Selection advantage A^S
    use_bf = bool(group.cluster_impure or force_brute_force)
    if use_bf:
        adv_sel_raw, uhat_km1, uhat_k = brute_force_selection_advantages(
            g_list, chk_list, sig_list, k=k, sigma_plus=sigma_plus_set
        )
        b_sel_loo = uhat_k - uhat_km1
    else:
        adv_sel_raw = dp_selection_advantages(group, k=k)
        b_sel_loo = (
            selection_loo_baselines(group, k=k)
            if center_select == "loo"
            else np.zeros(n, dtype=np.float64)
        )

    if center_select == "none":
        b_sel = np.zeros(n, dtype=np.float64)
        adv_sel_used = adv_sel_raw.copy()
    elif center_select == "loo":
        b_sel = b_sel_loo
        adv_sel_used = adv_sel_raw - b_sel
    elif center_select == "mean":
        b_sel = np.full(n, float(np.mean(adv_sel_raw)) if n > 0 else 0.0, dtype=np.float64)
        adv_sel_used = adv_sel_raw - b_sel
    else:
        raise ValueError(f"Unsupported center_select={center_select!r}")

    # Step 9-10: Coverage advantage A^cov and LOO baseline b^cov (where C = {g=1, Chk=1})
    is_c = [1 if (int(g_list[i]) == 1 and int(chk_list[i]) == 1) else 0 for i in range(n)]
    adv_cov_raw, b_cov, adv_cov_centered = coverage_advantages(is_c, k=k)
    pkpo_cf = pkpo_binary_advantages(is_c, k=k)

    # Step 11: Total advantage A_i = (1 - alpha) * (A_i^cov - b_i^cov) + alpha * A_i^S
    advantages = (1.0 - alpha) * adv_cov_centered + alpha * adv_sel_used

    # Step 12: Update Sigma^+_x for future steps (current step used pre-existing sigma_plus_set)
    updated_sigma_plus = set(sigma_plus_set) | group.c_sigs

    # Step 13: Drop problem if all A_i == 0
    all_zero = bool(n == 0 or np.all(np.abs(advantages) < 1e-12))
    loss_scale = (float(k) / float(n)) if n > 0 else 0.0

    return AdvantageResult(
        advantages=advantages,
        loss_scale=loss_scale,
        adv_cov_raw=adv_cov_raw,
        b_cov=b_cov,
        adv_cov_centered=adv_cov_centered,
        adv_sel_raw=adv_sel_raw,
        b_sel=b_sel,
        adv_sel_used=adv_sel_used,
        pkpo_closed_form=pkpo_cf,
        group=group,
        used_brute_force=use_bf,
        all_zero=all_zero,
        updated_sigma_plus=updated_sigma_plus,
    )


def load_policy_module_for_arm(arm: str):
    """Load the benchmark policy module (`compute_group_advantages`, `compute_policy_loss`) for `arm`."""
    import importlib
    import importlib.util
    import os
    from pathlib import Path
    from sspo.atomic_io import REPO_ROOT

    custom_file = os.environ.get("SSPO_AGENT_ALGO_FILE")
    if arm == "agent" and custom_file:
        p = Path(custom_file)
        if not p.is_absolute():
            p = REPO_ROOT / p
        spec = importlib.util.spec_from_file_location("sspo_custom_agent_algo", str(p))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    if arm in ("sspo", "drgrpo_pass1", "pkpo_pass8", "checker_reward", "tang_maj_loo"):
        return importlib.import_module(f"benchmark.reference_algorithms.{arm}")

    return importlib.import_module("benchmark.agent_algorithm")


def get_policy_loss_fn_for_arm(arm: str):
    """Return `compute_policy_loss(batch: PolicyLossBatch)` from the corresponding benchmark policy module."""
    mod = load_policy_module_for_arm(arm)
    return mod.compute_policy_loss


def compute_arm_advantages(
    arm: str,
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k: int = 8,
    alpha: float = 0.7,
    sigma_plus: Optional[Iterable[str]] = None,
    center_select: str = "none",
    pid: str = "",
    step: int = 1,
    total_steps: int = 30,
    response_lengths: Sequence[int] = (),
    sequence_logprobs: Sequence[float] = (),
    ref_sequence_logprobs: Sequence[float] = (),
    turn_count: int = 1,
) -> AdvantageResult:
    """Compute advantages for any of the training arms:
    - `agent`: Agent-editable RL algorithm in `benchmark/agent_algorithm.py` (or `SSPO_AGENT_ALGO_FILE`)
    - `sspo`: full SSPO objective (`alpha = 0.7`, `J^-`, `benchmark/reference_algorithms/sspo.py`)
    - `drgrpo_pass1`: Dr. GRPO on gold hidden test reward `g(y)` (`k = 1`, `benchmark/reference_algorithms/drgrpo_pass1.py`)
    - `pkpo_pass8`: PKPO pass@8 objective on `g(y)` (`k = 8`, `alpha = 0`, `benchmark/reference_algorithms/pkpo_pass8.py`)
    - `checker_reward`: Dr. GRPO on checker reward `Chk(y)` (`benchmark/reference_algorithms/checker_reward.py`)
    - `tang_maj_loo`: Tang et al. (2503.19595) majority-vote LOO with mean centering (`benchmark/reference_algorithms/tang_maj_loo.py`)
    """
    n = len(g_list)
    if arm in ("agent", "tang_maj_loo"):
        from benchmark.agent_algorithm import GroupContext

        mod = load_policy_module_for_arm(arm)
        group = classify_candidates(g_list, chk_list, sig_list, sigma_plus=sigma_plus)
        ctx = GroupContext(
            pid=str(pid),
            step=int(step),
            total_steps=int(total_steps),
            n=n,
            k=int(k),
            g_list=tuple(int(x) for x in g_list),
            chk_list=tuple(int(x) for x in chk_list),
            sig_list=tuple(str(x) for x in sig_list),
            response_lengths=tuple(int(x) for x in response_lengths) if response_lengths else tuple([0] * n),
            sequence_logprobs=tuple(float(x) for x in sequence_logprobs) if sequence_logprobs else tuple([0.0] * n),
            ref_sequence_logprobs=(
                tuple(float(x) for x in ref_sequence_logprobs) if ref_sequence_logprobs else tuple([0.0] * n)
            ),
            turn_count=int(turn_count),
            history_state={"sigma_plus": sorted(sigma_plus or ())},
        )
        out = mod.compute_group_advantages(ctx)
        adv = np.asarray(out.advantages, dtype=np.float64).reshape(-1)
        if adv.shape[0] != n:
            raise ValueError(f"Agent compute_group_advantages returned shape {adv.shape}, expected ({n},)")
        adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        adv = np.clip(adv, -50.0, 50.0)
        loss_scale = float(np.clip(float(out.loss_scale), 0.0, 2.0))
        all_zero = bool(out.drop_problem or n == 0 or np.all(np.abs(adv) < 1e-12))
        upd_sig = set((out.updated_state or {}).get("sigma_plus", sigma_plus or ())) | group.c_sigs
        zeros = np.zeros(n, dtype=np.float64)
        return AdvantageResult(
            advantages=adv,
            loss_scale=loss_scale,
            adv_cov_raw=adv,
            b_cov=zeros,
            adv_cov_centered=adv,
            adv_sel_raw=zeros,
            b_sel=zeros,
            adv_sel_used=zeros,
            pkpo_closed_form=zeros,
            group=group,
            used_brute_force=False,
            all_zero=all_zero,
            updated_sigma_plus=upd_sig,
        )

    if arm == "sspo":
        return compute_sspo_advantages(
            g_list=g_list,
            chk_list=chk_list,
            sig_list=sig_list,
            k=k,
            alpha=alpha,
            sigma_plus=sigma_plus,
            center_select=center_select,
        )

    group = classify_candidates(g_list, chk_list, sig_list, sigma_plus=sigma_plus)
    updated_sigma_plus = set(sigma_plus or ()) | group.c_sigs
    zeros = np.zeros(n, dtype=np.float64)

    if arm == "drgrpo_pass1":
        # k = 1 pass@1 Dr. GRPO (leave-one-out centered reward, no std normalization)
        adv_cov_raw, b_cov, adv_cov_centered = coverage_advantages(g_list, k=1)
        pkpo_cf = pkpo_binary_advantages(g_list, k=1)
        all_zero = bool(n == 0 or np.all(np.abs(adv_cov_centered) < 1e-12))
        return AdvantageResult(
            advantages=adv_cov_centered,
            loss_scale=(1.0 / float(n)) if n > 0 else 0.0,
            adv_cov_raw=adv_cov_raw,
            b_cov=b_cov,
            adv_cov_centered=adv_cov_centered,
            adv_sel_raw=zeros,
            b_sel=zeros,
            adv_sel_used=zeros,
            pkpo_closed_form=pkpo_cf,
            group=group,
            used_brute_force=False,
            all_zero=all_zero,
            updated_sigma_plus=updated_sigma_plus,
        )

    if arm == "pkpo_pass8":
        # Pass@k target (k = 8): PKPO binary closed form with LOO baseline
        # Note: (k/n) * adv_cov_raw == pkpo_binary_advantages(g_list, k)
        adv_cov_raw, b_cov, adv_cov_centered = coverage_advantages(g_list, k=k)
        pkpo_cf = pkpo_binary_advantages(g_list, k=k)
        all_zero = bool(n == 0 or np.all(np.abs(adv_cov_centered) < 1e-12))
        return AdvantageResult(
            advantages=adv_cov_centered,
            loss_scale=(float(k) / float(n)) if n > 0 else 0.0,
            adv_cov_raw=adv_cov_raw,
            b_cov=b_cov,
            adv_cov_centered=adv_cov_centered,
            adv_sel_raw=zeros,
            b_sel=zeros,
            adv_sel_used=zeros,
            pkpo_closed_form=pkpo_cf,
            group=group,
            used_brute_force=False,
            all_zero=all_zero,
            updated_sigma_plus=updated_sigma_plus,
        )

    if arm == "checker_reward":
        # Reward = Chk(y) in {0, 1}
        adv_cov_raw, b_cov, adv_cov_centered = coverage_advantages(chk_list, k=1)
        pkpo_cf = pkpo_binary_advantages(chk_list, k=1)
        all_zero = bool(n == 0 or np.all(np.abs(adv_cov_centered) < 1e-12))
        return AdvantageResult(
            advantages=adv_cov_centered,
            loss_scale=(1.0 / float(n)) if n > 0 else 0.0,
            adv_cov_raw=adv_cov_raw,
            b_cov=b_cov,
            adv_cov_centered=adv_cov_centered,
            adv_sel_raw=zeros,
            b_sel=zeros,
            adv_sel_used=zeros,
            pkpo_closed_form=pkpo_cf,
            group=group,
            used_brute_force=False,
            all_zero=all_zero,
            updated_sigma_plus=updated_sigma_plus,
        )

    raise ValueError(f"Unknown training arm: {arm!r}")


def compute_budget_mixture_advantages(
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    omega: Dict[int, float],
    alpha: float = 0.7,
    sigma_plus: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Proposition B budget-mixture advantage:
    `A_i^mu = sum_k mu(k) * A_i^(U, k)` with `mu(k) = omega(k) * k / n`.
    """
    n = len(g_list)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    total = np.zeros(n, dtype=np.float64)
    for k_val, w_k in omega.items():
        if w_k <= 0 or k_val < 1 or k_val > n:
            continue
        res_k = compute_sspo_advantages(
            g_list=g_list,
            chk_list=chk_list,
            sig_list=sig_list,
            k=k_val,
            alpha=alpha,
            sigma_plus=sigma_plus,
        )
        mu_k = float(w_k) * (float(k_val) / float(n))
        total += mu_k * res_k.advantages
    return total


def neyman_sample_allocation(
    sigma_1_estimates: Dict[str, float],
    total_budget: int,
    n_min: int = 12,
    n_max: int = 24,
) -> Dict[str, int]:
    """Proposition A Neyman sample allocation `n_x propto sigma_{1, x}` clamped to `[n_min, n_max]`."""
    pids = list(sigma_1_estimates.keys())
    m = len(pids)
    if m == 0:
        return {}
    weights = np.array([max(1e-4, float(sigma_1_estimates[p])) for p in pids], dtype=np.float64)
    raw = weights / weights.sum() * float(total_budget)
    alloc = np.clip(np.round(raw).astype(int), n_min, n_max)

    # Adjust to match total_budget within [n_min * m, n_max * m]
    target = int(np.clip(total_budget, n_min * m, n_max * m))
    order = np.argsort(-weights)
    while int(alloc.sum()) < target:
        for idx in order:
            if alloc[idx] < n_max and int(alloc.sum()) < target:
                alloc[idx] += 1
    while int(alloc.sum()) > target:
        for idx in reversed(order):
            if alloc[idx] > n_min and int(alloc.sum()) > target:
                alloc[idx] -= 1
    return {p: int(alloc[i]) for i, p in enumerate(pids)}
