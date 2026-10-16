"""Test 01: Dynamic Programming Correctness (`COUNTS`, `TABLE`, `EXPECT`, `J`, `DIFF`, `UHAT` vs Brute-Force).

Section 5.1 requirement:
Randomly generate 200 class-count configurations (including tied wrong clusters, non-empty W*, only T, all C,
and other boundary cases), across n = 16, k = 8 and smaller (n, k) pairs; assert DIFF / UHAT / EXPECT match
full-subset brute-force enumeration elementwise within 1e-9; also test the cluster-impure fallback path.
"""
from __future__ import annotations

import random
from typing import List, Set, Tuple

import numpy as np

from sspo.advantage import compute_sspo_advantages
from sspo.dp import (
    DIFF,
    UHAT,
    brute_force_selection_advantages,
    brute_force_single_candidate,
    dp_selection_advantages,
)
from sspo.selector import classify_candidates


def _build_candidates_from_counts(
    n_C: int,
    n_T: int,
    n_W_star: int,
    w_clusters: List[int],
) -> Tuple[List[int], List[int], List[str], Set[str]]:
    g_list: List[int] = []
    chk_list: List[int] = []
    sig_list: List[str] = []
    sigma_plus: Set[str] = {"sig_C_main"}

    # C candidates
    for _ in range(n_C):
        g_list.append(1)
        chk_list.append(1)
        sig_list.append("sig_C_main")

    # T candidates (Chk = 0, mix of g = 0 and g = 1)
    for idx in range(n_T):
        g_list.append(1 if idx % 3 == 0 else 0)
        chk_list.append(0)
        sig_list.append(f"sig_T_{idx}")

    # W* candidates (g = 0, Chk = 1, sig in sigma_plus)
    for _ in range(n_W_star):
        g_list.append(0)
        chk_list.append(1)
        sig_list.append("sig_C_main")

    # W_1 .. W_M wrong clusters
    for m_idx, sz in enumerate(w_clusters):
        for _ in range(sz):
            g_list.append(0)
            chk_list.append(1)
            sig_list.append(f"sig_W_{m_idx}")

    return g_list, chk_list, sig_list, sigma_plus


def generate_200_configurations(seed: int = 20261015) -> List[Tuple[int, int, int, int, int, List[int]]]:
    """Return 200 diverse `(n, k, n_C, n_T, n_W_star, w_clusters)` configurations."""
    rng = random.Random(seed)
    configs: List[Tuple[int, int, int, int, int, List[int]]] = [
        # Boundary cases at n = 16, k = 8
        (16, 8, 0, 16, 0, []),  # only T
        (16, 8, 16, 0, 0, []),  # all C
        (16, 8, 0, 0, 16, []),  # all W*
        (16, 8, 0, 0, 0, [16]),  # single giant W_1
        (16, 8, 3, 4, 0, [9]),  # counterexample 1 (3 C, 9 W_1, 4 T)
        (16, 8, 3, 4, 0, [8, 1]),  # counterexample 2 (3 C, 8 W_1, 1 W_2, 4 T)
        (16, 8, 4, 2, 2, [4, 4]),  # tied W_1 == W_2 == C and non-empty W*
        (16, 8, 2, 2, 1, [3, 3, 3, 2]),  # 3-way tied wrong clusters + W*
        (16, 8, 1, 0, 0, [1] * 15),  # 15 singleton wrong clusters
        (16, 8, 5, 3, 3, [2, 2, 1]),  # heavy W*
        # Small (n, k) boundary cases
        (6, 3, 2, 1, 1, [2]),
        (8, 4, 2, 2, 0, [2, 2]),
        (10, 4, 3, 2, 1, [2, 1, 1]),
        (12, 6, 3, 3, 1, [3, 2]),
    ]

    nk_choices = [(16, 8)] * 120 + [(12, 6)] * 25 + [(10, 5)] * 20 + [(8, 4)] * 21
    while len(configs) < 200:
        n, k = nk_choices[len(configs) - 14]
        rem = n
        n_C = rng.randint(0, rem)
        rem -= n_C
        n_T = rng.randint(0, rem)
        rem -= n_T
        # Include W* > 0 in ~40% of random configs
        n_W_star = rng.randint(0, min(rem, 4)) if (rng.random() < 0.4 and rem > 0) else 0
        rem -= n_W_star
        w_clusters: List[int] = []
        while rem > 0:
            # Encourage ties frequently
            if w_clusters and rng.random() < 0.45 and rem >= w_clusters[-1]:
                sz = w_clusters[-1]
            else:
                sz = rng.randint(1, rem)
            w_clusters.append(sz)
            rem -= sz
        w_clusters.sort(reverse=True)
        configs.append((n, k, n_C, n_T, n_W_star, w_clusters))

    assert len(configs) == 200
    return configs


def test_200_random_configurations_dp_vs_bruteforce():
    configs = generate_200_configurations()
    assert len(configs) == 200

    for idx, (n, k, n_C, n_T, n_W_star, w_clusters) in enumerate(configs):
        g_list, chk_list, sig_list, sigma_plus = _build_candidates_from_counts(
            n_C, n_T, n_W_star, w_clusters
        )
        group = classify_candidates(g_list, chk_list, sig_list, sigma_plus=sigma_plus)
        assert not group.cluster_impure, f"Config {idx} should be cluster-pure after W* separation"

        adv_dp = dp_selection_advantages(group, k=k)
        adv_bf, uhat_km1_bf, uhat_k_bf = brute_force_selection_advantages(
            g_list, chk_list, sig_list, k=k, sigma_plus=sigma_plus
        )

        np.testing.assert_allclose(
            adv_dp,
            adv_bf,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"DP vs Brute-Force mismatch at config #{idx}: {(n, k, n_C, n_T, n_W_star, w_clusters)}",
        )

        # Also check UHAT(X, k-1) and UHAT(X, k) directly against brute-force uhat arrays
        for i in range(n):
            cls_name, m_idx = group.candidate_class[i]
            x_spec = ("W", m_idx) if cls_name == "W" else cls_name
            u_km1_dp = UHAT(
                x_spec, k - 1, n=n, n_C=group.n_C, n_T=group.n_T, n_W_clusters=group.n_W_clusters
            )
            assert abs(u_km1_dp - uhat_km1_bf[i]) <= 1e-9
            if k < n:
                u_k_dp = UHAT(
                    x_spec, k, n=n, n_C=group.n_C, n_T=group.n_T, n_W_clusters=group.n_W_clusters
                )
                assert abs(u_k_dp - uhat_k_bf[i]) <= 1e-9


def test_cluster_impure_fallback_path():
    # Case A: C spans multiple distinct signatures (sig_C1 and sig_C2) at n = 16, k = 8
    g_list = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    chk_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    sig_list = [
        "sig_C1",
        "sig_C1",
        "sig_C2",  # Second correct signature -> triggers fallback!
        "sig_W1",
        "sig_W1",
        "sig_W1",
        "sig_W1",
        "sig_W1",
        "sig_W2",
        "sig_W2",
        "sig_W3",
        "sig_W_star",
        "sig_T",
        "sig_T",
        "sig_T",
        "sig_T",
    ]
    sigma_plus = {"sig_W_star"}

    res = compute_sspo_advantages(g_list, chk_list, sig_list, k=8, alpha=0.7, sigma_plus=sigma_plus)
    assert res.group.cluster_impure is True
    assert res.used_brute_force is True

    # Verify each candidate i against explicit single-candidate C(15, 7) = 6435 subset enumeration
    for i in range(16):
        bf_i = brute_force_single_candidate(i, g_list, chk_list, sig_list, k=8, sigma_plus=sigma_plus)
        assert abs(res.adv_sel_raw[i] - bf_i) <= 1e-12

    # Case B: A wrong cluster shares a signature with C that was not yet in sigma_plus
    sig_list_overlap = [
        "sig_C_new",
        "sig_C_new",
        "sig_C_new",
        "sig_C_new",  # g=0 but shares sig_C_new with C!
        "sig_W1",
        "sig_W1",
        "sig_W1",
        "sig_W1",
        "sig_W1",
        "sig_W2",
        "sig_W2",
        "sig_W2",
        "sig_T",
        "sig_T",
        "sig_T",
        "sig_T",
    ]
    res2 = compute_sspo_advantages(g_list, chk_list, sig_list_overlap, k=8, alpha=0.7, sigma_plus=set())
    assert res2.group.cluster_impure is True
    assert res2.used_brute_force is True
    for i in range(16):
        bf_i = brute_force_single_candidate(i, g_list, chk_list, sig_list_overlap, k=8, sigma_plus=set())
        assert abs(res2.adv_sel_raw[i] - bf_i) <= 1e-12
    # And after this step, sig_C_new is added to updated_sigma_plus
    assert "sig_C_new" in res2.updated_sigma_plus
