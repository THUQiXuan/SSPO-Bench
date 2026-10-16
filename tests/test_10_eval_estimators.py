"""Test 10: Evaluation Estimators (`pass@k`, `select@k`, Paired Two-Level Bootstrap, Holm Correction).

Section 5.10 requirement:
Assert that the unbiased `pass@k` estimator and the `select@k` subset estimator match analytical values
on synthetic data, and verify `is_critical_pool` and paired two-level bootstrap intervals.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from sspo.dp import comb_exact
from sspo.eval import (
    estimate_pass_at_k,
    estimate_select_at_k,
    evaluate_problem_rollouts,
    holm_bonferroni,
    is_critical_pool,
    paired_two_level_bootstrap,
)
from sspo.selector import j_s_subset


def test_pass_at_k_unbiased_against_analytical():
    # 1. Exact hypergeometric check for fixed pool (n = 16, c = 4, k = 8)
    n, c, k = 16, 4, 8
    expected_hypergeom = 1.0 - (math.comb(12, 8) / math.comb(16, 8))
    assert abs(estimate_pass_at_k(n, c, k) - expected_hypergeom) < 1e-15

    # 2. Exact expectation over Binomial(n, p) equals analytical 1 - (1 - p)^k
    for p in (0.1, 0.25, 0.5, 0.7):
        for k_val in (1, 2, 4, 8, 16):
            analytical = 1.0 - ((1.0 - p) ** k_val)
            expected_est = 0.0
            for c_val in range(0, n + 1):
                binom_prob = comb_exact(n, c_val) * (p ** c_val) * ((1.0 - p) ** (n - c_val))
                expected_est += binom_prob * estimate_pass_at_k(n, c_val, k_val)
            assert abs(expected_est - analytical) < 1e-14


def test_select_at_k_subset_estimator_matches_analytical_and_bruteforce():
    # Synthetic pool: n = 12 candidates: 3 C, 5 W_1, 2 W_2, 2 T
    g_list = [1] * 3 + [0] * 5 + [0] * 2 + [0] * 2
    chk_list = [1] * 3 + [1] * 5 + [1] * 2 + [0] * 2
    sig_list = ["sigC"] * 3 + ["sigW1"] * 5 + ["sigW2"] * 2 + ["sigT"] * 2

    for k in (1, 2, 4, 8):
        # True analytical value on the finite pool by enumerating all C(12, k) subsets
        all_subs = list(itertools.combinations(range(12), k))
        analytical_pool_val = float(np.mean([j_s_subset(s, g_list, chk_list, sig_list) for s in all_subs]))

        res = estimate_select_at_k(
            g_list, chk_list, sig_list, k=k, n_subsets=2000, subset_seed=12345
        )
        # Exact DP value must equal full subset average to machine precision
        assert abs(res["select_at_k_exact"] - analytical_pool_val) < 1e-12
        # Random-subset Monte Carlo estimator (with n_subsets=2000) must be within 0.02 of analytical
        assert abs(res["select_at_k"] - analytical_pool_val) < 0.02
        assert res["n_subsets"] == 2000
        assert res["subset_seed"] == 12345

    assert is_critical_pool(g_list, chk_list, sig_list, k=8) is True
    # Pure C + T pool (no wrong passed candidates) is NOT critical because J_S(K) == J_cov(K) for every subset!
    assert is_critical_pool([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], ["C"] * 2 + ["T"] * 6, k=4) is False


def test_paired_two_level_bootstrap_and_holm():
    run_a = [
        {"pid": f"p{i}", "contest_id": f"c{i // 2}", "select_at_k": {"8": 0.65}}
        for i in range(20)
    ]
    run_b = [
        {"pid": f"p{i}", "contest_id": f"c{i // 2}", "select_at_k": {"8": 0.50}}
        for i in range(20)
    ]
    boot = paired_two_level_bootstrap([run_a], [run_b], metric="select_at_k", k_key="8", n_boot=500)
    assert abs(boot["mean_diff"] - 0.15) < 1e-9
    assert boot["ci_low"] > 0.10
    assert boot["win"] is True

    comps = holm_bonferroni([
        {"arm_a": "sspo", "arm_b": "drgrpo_pass1", "mean_diff": 0.04, "p_value": 0.005},
        {"arm_a": "sspo", "arm_b": "pkpo_pass8", "mean_diff": 0.03, "p_value": 0.020},
        {"arm_a": "sspo", "arm_b": "frozen", "mean_diff": 0.08, "p_value": 0.001},
    ])
    assert comps[2]["p_value_holm"] == 0.003
    assert comps[0]["p_value_holm"] == 0.010
    assert comps[1]["p_value_holm"] == 0.020
    assert all(c["significant_holm_005"] for c in comps)
