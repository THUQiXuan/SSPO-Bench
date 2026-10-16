"""Test 03: Unbiasedness Monte Carlo & Exact Finite-Difference Verification (Theorem 2(a)(e)).

Section 5.3 requirement:
Construct a small tabular policy (e.g., 3-class discrete distribution), compute the gradient of
`J_S^(k)` and `J_cov^(k)` via finite differences, and compare with the Monte Carlo mean of
`(k / n) * sum_i A_i * s_i`, asserting relative error is within sampling noise.
"""
from __future__ import annotations

import numpy as np

from sspo.advantage import compute_sspo_advantages
from sspo.dp import comb_exact


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _exact_objectives_3class(logits: np.ndarray, k: int) -> tuple[float, float]:
    """Exact `(J_cov^(k), J_S^(k))` for 3-class tabular policy `(C, W_1, T)`."""
    probs = _softmax(logits)
    p_c, p_w, p_t = float(probs[0]), float(probs[1]), float(probs[2])
    j_cov = 1.0 - ((1.0 - p_c) ** k)

    j_s = 0.0
    for c in range(1, k + 1):
        for w in range(0, min(c, k - c) + 1):
            t = k - c - w
            multinom = comb_exact(k, c) * comb_exact(k - c, w)
            prob = multinom * (p_c ** c) * (p_w ** w) * (p_t ** t)
            j_val = 1.0 if c > w else 0.5
            j_s += prob * j_val
    return j_cov, j_s


def _finite_diff_gradients(logits: np.ndarray, k: int, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    grad_cov = np.zeros_like(logits)
    grad_sel = np.zeros_like(logits)
    for d in range(len(logits)):
        l_plus = logits.copy()
        l_minus = logits.copy()
        l_plus[d] += eps
        l_minus[d] -= eps
        cov_p, sel_p = _exact_objectives_3class(l_plus, k=k)
        cov_m, sel_m = _exact_objectives_3class(l_minus, k=k)
        grad_cov[d] = (cov_p - cov_m) / (2.0 * eps)
        grad_sel[d] = (sel_p - sel_m) / (2.0 * eps)
    return grad_cov, grad_sel


def test_unbiased_gradient_mc_and_exact():
    # 3-class discrete policy: class 0 = C, class 1 = W_1, class 2 = T
    logits = np.array([-0.2, 0.4, -0.1], dtype=np.float64)
    probs = _softmax(logits)
    n = 16
    k = 8
    alpha = 0.7

    fd_grad_cov, fd_grad_sel = _finite_diff_gradients(logits, k=k)
    fd_grad_total = (1.0 - alpha) * fd_grad_cov + alpha * fd_grad_sel

    # Score function s(y) = nabla_theta log pi_theta(y) = e_y - probs for each of the 3 classes
    score_vectors = np.eye(3, dtype=np.float64) - probs[None, :]

    # Monte Carlo estimation over M = 25,000 independent groups of size n = 16
    rng = np.random.default_rng(20261015)
    num_trials = 25000
    samples = rng.choice(3, size=(num_trials, n), p=probs)

    est_cov_list = np.zeros((num_trials, 3), dtype=np.float64)
    est_sel_list = np.zeros((num_trials, 3), dtype=np.float64)
    est_tot_list = np.zeros((num_trials, 3), dtype=np.float64)

    # Cache advantages by count tuple (n_c, n_w) since n=16 has only 153 count states!
    adv_cache = {}
    for t_idx in range(num_trials):
        row = samples[t_idx]
        n_c = int(np.sum(row == 0))
        n_w = int(np.sum(row == 1))
        n_t = n - n_c - n_w
        key = (n_c, n_w)
        if key not in adv_cache:
            g_canonical = [1] * n_c + [0] * n_w + [0] * n_t
            chk_canonical = [1] * n_c + [1] * n_w + [0] * n_t
            sig_canonical = ["C"] * n_c + ["W1"] * n_w + ["T"] * n_t
            res = compute_sspo_advantages(g_canonical, chk_canonical, sig_canonical, k=k, alpha=alpha)
            a_cov_c = res.adv_cov_centered[0] if n_c > 0 else 0.0
            a_cov_notc = res.adv_cov_centered[-1] if (n_w + n_t) > 0 else 0.0
            a_sel_c = res.adv_sel_raw[0] if n_c > 0 else 0.0
            a_sel_w = res.adv_sel_raw[n_c] if n_w > 0 else 0.0
            a_sel_t = 0.0
            adv_cache[key] = (a_cov_c, a_cov_notc, a_sel_c, a_sel_w, a_sel_t)

        a_cov_c, a_cov_notc, a_sel_c, a_sel_w, a_sel_t = adv_cache[key]
        scale = float(k) / float(n)
        # Sum (k/n) * A_i * s_i over the n samples
        g_cov = scale * (
            n_c * a_cov_c * score_vectors[0]
            + n_w * a_cov_notc * score_vectors[1]
            + n_t * a_cov_notc * score_vectors[2]
        )
        g_sel = scale * (
            n_c * a_sel_c * score_vectors[0]
            + n_w * a_sel_w * score_vectors[1]
            + n_t * a_sel_t * score_vectors[2]
        )
        est_cov_list[t_idx] = g_cov
        est_sel_list[t_idx] = g_sel
        est_tot_list[t_idx] = (1.0 - alpha) * g_cov + alpha * g_sel

    mc_grad_cov = np.mean(est_cov_list, axis=0)
    mc_grad_sel = np.mean(est_sel_list, axis=0)
    mc_grad_tot = np.mean(est_tot_list, axis=0)

    se_cov = np.std(est_cov_list, axis=0, ddof=1) / np.sqrt(num_trials)
    se_sel = np.std(est_sel_list, axis=0, ddof=1) / np.sqrt(num_trials)
    se_tot = np.std(est_tot_list, axis=0, ddof=1) / np.sqrt(num_trials)

    # Assert MC mean agrees with finite-difference gradient within 3.5 standard errors (and < 2% relative error)
    assert np.all(np.abs(mc_grad_cov - fd_grad_cov) <= 3.5 * se_cov + 1e-5)
    assert np.all(np.abs(mc_grad_sel - fd_grad_sel) <= 3.5 * se_sel + 1e-5)
    assert np.all(np.abs(mc_grad_tot - fd_grad_total) <= 3.5 * se_tot + 1e-5)
    rel_err = np.linalg.norm(mc_grad_tot - fd_grad_total) / np.linalg.norm(fd_grad_total)
    assert rel_err < 0.025, f"Relative error {rel_err:.4f} exceeds sampling noise threshold"
