"""Test 02: Theorem 2 Properties & PKPO Cross-Check.

Section 5.2 requirement:
1. For T-class candidates (`Chk = 0`), selection advantage `A_i^S == 0`.
2. Under cluster purity (and with `J^-` on `W*`), every `g = 0` candidate has `A_i^S <= 0` and every `C` candidate has `A_i^S >= 0`.
3. Total SSPO advantage `A_i <= 0` for every `g = 0` candidate.
4. Coverage advantage `A_i^cov` multiplied by `k / n` is elementwise equal to the PKPO binary closed form
   `f_i * C(n - c, k - 1) / C(n, k)`.
5. Verify the exact counterexamples showing why selection centering is omitted.
"""
from __future__ import annotations

import numpy as np

from sspo.advantage import compute_sspo_advantages, coverage_advantages, pkpo_binary_advantages
from tests.test_01_dp_correctness import _build_candidates_from_counts, generate_200_configurations


def test_theorem2_sign_and_anti_hacking_properties():
    configs = generate_200_configurations(seed=42)
    for idx, (n, k, n_C, n_T, n_W_star, w_clusters) in enumerate(configs):
        g_list, chk_list, sig_list, sigma_plus = _build_candidates_from_counts(
            n_C, n_T, n_W_star, w_clusters
        )
        res = compute_sspo_advantages(
            g_list, chk_list, sig_list, k=k, alpha=0.7, sigma_plus=sigma_plus, center_select="none"
        )

        for i in range(n):
            cls_name, _ = res.group.candidate_class[i]
            # (1) T class has selection advantage identically 0
            if cls_name == "T":
                assert res.adv_sel_raw[i] == 0.0, f"T candidate must have A_i^S == 0 at config {idx}"
            # (2) C class has selection advantage >= 0
            if cls_name == "C":
                assert res.adv_sel_raw[i] >= -1e-12, f"C candidate must have A_i^S >= 0 at config {idx}"
            # (3) Every g = 0 candidate (whether Chk=0, W_m, or W*) has A_i^S <= 0 and total A_i <= 0
            if g_list[i] == 0:
                assert res.adv_sel_raw[i] <= 1e-12, (
                    f"g=0 candidate (class {cls_name}) got positive A_i^S={res.adv_sel_raw[i]} at config {idx}"
                )
                assert res.advantages[i] <= 1e-12, (
                    f"g=0 candidate (class {cls_name}) got positive total A_i={res.advantages[i]} at config {idx}"
                )


def test_coverage_times_k_over_n_matches_pkpo_closed_form():
    configs = generate_200_configurations(seed=99)
    for idx, (n, k, n_C, n_T, n_W_star, w_clusters) in enumerate(configs):
        g_list, chk_list, _, _ = _build_candidates_from_counts(n_C, n_T, n_W_star, w_clusters)
        is_c = [1 if (g_list[i] == 1 and chk_list[i] == 1) else 0 for i in range(n)]

        adv_cov_raw, b_cov, _ = coverage_advantages(is_c, k=k)
        pkpo_cf = pkpo_binary_advantages(is_c, k=k)

        scaled_cov = (float(k) / float(n)) * adv_cov_raw
        np.testing.assert_allclose(
            scaled_cov,
            pkpo_cf,
            atol=1e-15,
            rtol=1e-15,
            err_msg=f"(k/n)*A_i^cov != PKPO binary closed form at config {idx}",
        )
        # Also verify b_i^cov >= 0 everywhere
        assert np.all(b_cov >= -1e-15)


def test_centering_counterexamples():
    # Counterexample 1: n=16, k=8, 3 C, 9 W_1, 4 T
    # T candidate has b^S = -0.0210, so centered selection advantage becomes +0.0210 > 0!
    g1 = [1] * 3 + [0] * 9 + [0] * 4
    chk1 = [1] * 3 + [1] * 9 + [0] * 4
    sig1 = ["sigC"] * 3 + ["sigW1"] * 9 + ["sigT"] * 4
    res1_loo = compute_sspo_advantages(g1, chk1, sig1, k=8, alpha=0.7, center_select="loo")
    res1_uncentered = compute_sspo_advantages(g1, chk1, sig1, k=8, alpha=0.7, center_select="none")
    assert abs(res1_loo.b_sel[-1] - (-0.0210)) < 1e-4
    assert abs(res1_loo.adv_sel_used[-1] - (+0.0210)) < 1e-4
    assert res1_uncentered.adv_sel_used[-1] == 0.0

    # Counterexample 2: n=16, k=8, 3 C, 8 W_1, 1 W_2, 4 T
    # Singleton W_2 candidate has A^S = 0, b^S = -0.0272, centered +0.0272 > 0!
    g2 = [1] * 3 + [0] * 8 + [0] * 1 + [0] * 4
    chk2 = [1] * 3 + [1] * 8 + [1] * 1 + [0] * 4
    sig2 = ["sigC"] * 3 + ["sigW1"] * 8 + ["sigW2"] * 1 + ["sigT"] * 4
    res2_loo = compute_sspo_advantages(g2, chk2, sig2, k=8, alpha=0.7, center_select="loo")
    assert abs(res2_loo.adv_sel_raw[11]) < 1e-12
    assert abs(res2_loo.b_sel[11] - (-0.0272)) < 1e-4
    assert abs(res2_loo.adv_sel_used[11] - (+0.0272)) < 1e-4
