"""Agent-Editable RL Algorithm Module for SSPO-Bench (`benchmark/agent_algorithm.py`).

================================================================================
BENCHMARK RULES & INTERFACE SPECIFICATION
================================================================================
This is the **ONLY** file you (the AI agent) are permitted to modify.
All other files in the repository (`sspo/*`, `configs/*`, `data/*`, `benchmark/verifier.py`,
`benchmark/evaluate_agent.py`) are cryptographically locked via SHA-256 hashes.

### Problem Setting
We train a code generation policy (`Qwen/Qwen2.5-Coder-3B-Instruct` with LoRA r=16)
that will be deployed at test time with **Test-Time Selection (`Select@K`)**:
- At deployment time, for each problem `x`, `K = 8` candidate programs are sampled
  and passed through a deterministic selector `S(Y_K)`:
  1. **Public Example Filter (`Chk(y) in {0, 1}`)**: Candidates are executed on the
     problem's public example inputs. If any candidate passes (`Chk(y) = 1`),
     failing candidates (`Chk(y) = 0`) are filtered out.
  2. **Execution Signature Majority Vote (`sig(y)`)**: Surviving candidates are
     clustered by their execution output signature `sig(y)` on public/probe inputs.
     The selector picks a candidate from the largest output cluster (breaking ties
     uniformly at random).
- **Goal**: Maximize test-time **`Select@8`** (the probability that the single program
  chosen by `S` from `K = 8` candidates passes all hidden tests `g(y) = 1`).

### Why Standard RL (`Pass@1` / `Pass@K`) Can Be Suboptimal for `Select@K`
During training, for each problem `x`, `n = 16` candidate trajectories are sampled
and evaluated by the training environment, yielding three signals per candidate `i`:
  - `g_list[i] in {0, 1}`   : Hidden-test correctness (`1` = passes all hidden tests)
  - `chk_list[i] in {0, 1}` : Public-example checker outcome (`1` = passes public examples)
  - `sig_list[i] : str`     : Canonical execution output signature (or `"TLE"`/`"RE"`)

 Candidates naturally partition into:
  - `C = {i : g_i = 1 and chk_i = 1}` : Correct candidates that pass the public filter
  - `T = {i : chk_i = 0}`             : Candidates rejected by the public filter
  - `W_m = {i : g_i = 0, chk_i = 1, sig_i = s_m}` : **False-positive wrong clusters**
    that pass the public examples (`chk_i = 1`) and collude on the same wrong output `s_m`!
    Whenever a wrong cluster `|W_m| >= |C|`, majority voting picks the wrong answer!

### Your Degrees of Freedom
You may design and implement:
1. **`compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput`**:
   - Assigns per-candidate advantages `advantages: np.ndarray` of shape `(n,)` and
     problem-level `loss_scale: float` (`in [0.0, 2.0]`) for each problem's `n` rollouts.
   - You have access to `g_list`, `chk_list`, `sig_list`, `n`, `k`, `step`, `total_steps`,
     `response_lengths`, `sequence_logprobs`, `ref_sequence_logprobs`, and a persistent
     per-problem dictionary `history_state` (e.g., to track historically verified signatures).
   - You may also use read-only combinatorial/DP helpers from `sspo.dp` and `sspo.selector`.
2. **`compute_policy_loss(batch: PolicyLossBatch) -> Tuple[torch.Tensor, Dict[str, float]]`**:
   - Computes the differentiable token-level policy gradient + regularization loss
     from `policy_logprobs`, `old_logprobs`, `ref_logprobs`, `action_mask`, `advantages`,
     and `loss_scales`.

### Verification & Sandbox Constraints (`python -m benchmark.evaluate_agent verify`)
- **Allowed imports ONLY**: `math`, `itertools`, `collections`, `functools`, `typing`,
  `dataclasses`, `random`, `numpy`, `torch`, `torch.nn.functional`, `sspo.dp`, `sspo.selector`.
- **Forbidden**: Any filesystem/network/process/introspection access (`os`, `sys`, `open`,
  `eval`, `exec`, `getattr`, `subprocess`, `pathlib`, etc.).
- **Required Mathematical Invariants**:
  1. **Permutation Equivariance**: Permuting the `n` candidates in `ctx` must permute
     `output.advantages` identically.
  2. **Finiteness & Bounds**: No `NaN`/`Inf`; `|advantages[i]| <= 50.0`; `0.0 <= loss_scale <= 2.0`.
  3. **Differentiability**: `compute_policy_loss` must return a scalar `torch.Tensor` with
     `requires_grad=True` and valid finite gradients w.r.t. `policy_logprobs`.
  4. **Efficiency**: `< 50 ms` per call to `compute_group_advantages` on CPU.
================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from sspo.dp import (
    UHAT,
    brute_force_selection_advantages,
    comb_exact,
    dp_selection_advantages,
)
from sspo.selector import ClassifiedGroup, classify_candidates


@dataclass(frozen=True)
class GroupContext:
    """Read-only rollout outcomes and metadata for a single training problem `x`."""

    pid: str
    step: int
    total_steps: int
    n: int  # Number of training rollouts (default: 16)
    k: int  # Target deployment selection budget K (default: 8)
    g_list: Tuple[int, ...]  # Hidden test correctness in {0, 1}, shape (n,)
    chk_list: Tuple[int, ...]  # Public example checker pass in {0, 1}, shape (n,)
    sig_list: Tuple[str, ...]  # Canonical execution output signature, shape (n,)
    response_lengths: Tuple[int, ...] = ()  # Generated token counts per rollout, shape (n,)
    sequence_logprobs: Tuple[float, ...] = ()  # Sum of old policy token logprobs, shape (n,)
    ref_sequence_logprobs: Tuple[float, ...] = ()  # Sum of ref policy token logprobs, shape (n,)
    turn_count: int = 1
    history_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupAdvantageOutput:
    """Output returned by `compute_group_advantages` for a single problem `x`."""

    advantages: np.ndarray  # 1D float64 array of shape (n,)
    loss_scale: float = 0.5  # Problem-level weight in [0.0, 2.0] (e.g. k / n or 1 / n)
    drop_problem: bool = False  # If True, skip gradient update for this problem
    updated_state: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyLossBatch:
    """Read-only micro-batch of token log-probabilities for `compute_policy_loss`."""

    policy_logprobs: torch.Tensor  # (B, T) current policy token logprobs (requires_grad=True)
    old_logprobs: torch.Tensor  # (B, T) rollout behavior policy token logprobs (detached)
    ref_logprobs: torch.Tensor  # (B, T) frozen reference policy token logprobs (detached)
    action_mask: torch.Tensor  # (B, T) float32 mask: 1.0 on generated tokens, 0.0 on prompt/pad
    advantages: torch.Tensor  # (B,) sequence-level advantage A_i from compute_group_advantages
    loss_scales: torch.Tensor  # (B,) problem-level scale factor from compute_group_advantages
    num_problems_in_batch: int
    l_max: int = 1536
    clip_eps: float = 0.2
    kl_coef: float = 0.001
    step: int = 1
    total_steps: int = 30


# ==============================================================================
# PART 1: GROUP-LEVEL ADVANTAGE / CREDIT ASSIGNMENT (AGENT EDITABLE)
# ==============================================================================

def compute_group_advantages(ctx: GroupContext) -> GroupAdvantageOutput:
    """Compute per-rollout advantages `A_1, ..., A_n` for one problem's `n` candidates.

    Default Starting Baseline: **Vanilla Pass@1 Dr. GRPO (`drgrpo_pass1`)**
    - Uses leave-one-out mean-centered binary hidden-test reward `g_i in {0, 1}`:
        `A_i = g_i - (1 / (n - 1)) * sum_{j != i} g_j`
    - Does NOT model the public checker `chk_list`, output signature clusters `sig_list`,
      or the deployment selection budget `k = 8`.

     Agents should modify this function (and/or `compute_policy_loss` below) to design
    a stronger selection-aware RL algorithm!
    """
    n = ctx.n
    if n == 0:
        return GroupAdvantageOutput(
            advantages=np.zeros(0, dtype=np.float64),
            loss_scale=0.0,
            drop_problem=True,
            updated_state=dict(ctx.history_state),
        )

    g_arr = np.array([1.0 if int(g) == 1 else 0.0 for g in ctx.g_list], dtype=np.float64)
    total_g = float(np.sum(g_arr))

    if n > 1:
        loo_baseline = (total_g - g_arr) / float(n - 1)
        advantages = g_arr - loo_baseline
    else:
        advantages = g_arr.copy()

    drop_problem = bool(np.all(np.abs(advantages) < 1e-12))
    loss_scale = 1.0 / float(n)

    # Track verified signatures in history_state (optional utility for stateful algorithms)
    sigma_plus = set(ctx.history_state.get("sigma_plus", ()))
    group = classify_candidates(ctx.g_list, ctx.chk_list, ctx.sig_list, sigma_plus=sigma_plus)
    updated_sigma_plus = sorted(sigma_plus | group.c_sigs)

    return GroupAdvantageOutput(
        advantages=advantages,
        loss_scale=loss_scale,
        drop_problem=drop_problem,
        updated_state={"sigma_plus": updated_sigma_plus},
        metrics={
            "adv_cov_mean": float(np.mean(advantages)),
            "adv_sel_mean": 0.0,
        },
    )


# ==============================================================================
# PART 2: TOKEN-LEVEL POLICY LOSS FUNCTION (AGENT EDITABLE)
# ==============================================================================

def compute_policy_loss(batch: PolicyLossBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the differentiable token-level policy gradient + KL loss for a micro-batch.

    Default Starting Baseline: **Dr. GRPO Clipped Surrogate + Unbiased KL Penalty**
    - Fixed length normalizer `1 / l_max` (avoids length bias across variable-length solutions).
    - Clipped importance sampling ratio `r_{i,t} = exp(log pi_theta - log pi_old)`.
    - Unbiased nonnegative KL estimator `exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1`.
    """
    log_ratio = torch.clamp(batch.policy_logprobs - batch.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    adv = batch.advantages.unsqueeze(-1)  # (B, 1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - batch.clip_eps, 1.0 + batch.clip_eps) * adv
    pg_token = -torch.minimum(surr1, surr2)

    # Unbiased per-token KL(pi_theta || pi_ref) >= 0
    log_ref_diff = torch.clamp(batch.ref_logprobs - batch.policy_logprobs, min=-20.0, max=20.0)
    kl_token = torch.exp(log_ref_diff) - log_ref_diff - 1.0

    per_seq_Scale = batch.loss_scales / float(max(1, batch.num_problems_in_batch) * max(1, batch.l_max))
    masked_pg = (pg_token * batch.action_mask).sum(dim=-1) * per_seq_Scale
    masked_kl = (kl_token * batch.action_mask).sum(dim=-1) * per_seq_Scale

    total_loss = (masked_pg + batch.kl_coef * masked_kl).sum()

    with torch.no_grad():
        active_tokens = torch.clamp(batch.action_mask.sum(), min=1.0)
        mean_kl = float(((kl_token * batch.action_mask).sum() / active_tokens).item())
        clipped = ((ratio < (1.0 - batch.clip_eps)) | (ratio > (1.0 + batch.clip_eps))).float()
        clip_frac = float(((clipped * batch.action_mask).sum() / active_tokens).item())

    stats = {
        "loss": float(total_loss.detach().item()),
        "pg_loss": float(masked_pg.sum().detach().item()),
        "kl": mean_kl,
        "clip_frac": clip_frac,
    }
    return total_loss, stats
