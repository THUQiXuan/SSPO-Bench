"""Exact dynamic programming (`COUNTS`, `TABLE`, `EXPECT`, `J`, `DIFF`, `UHAT`) and 6435-subset brute-force fallback.

Definitions:
- `COUNTS(X)`: after removing one member of class `X`, returns remaining counts `(n'_C, n'_T, n'_d, n'_{W,1..M'})`;
  `W*` does not participate in valid draws (`J^- = 0` whenever a `W*` item is drawn), while remaining in the
  total denominator `C(n-1, kk)`.
- `TABLE(w)`:
    `D(0,0,0) = 1`;
    `D(m,s,t) = sum_{j=0}^{min(w, n'_{W,m}, s)} C(n'_{W,m}, j) * D(m-1, s-j, t - 1[j = w])`
    `N_w(s,t) = D(M',s,t)` (for `w >= 1`, taking `t >= 1`); `N_0(0,0) = 1`.
- `EXPECT(X, kk, phi)`:
    `sum_{w,t,c,a,s} C(n'_C, c) * C(n'_d, a) * N_w(s,t) * C(n'_T, kk-c-a-s) / C(n-1, kk) * phi(c,a,w,t)`
- `J(c,a,w,t)`:
    `V* = max(a,w)`, `R = t * 1[w = V* >= 1] + 1[a = V* >= 1]`;
    `0` if `c == 0` or `c < V*`, `1` if `c > V*`, `1 / (1 + R)` if `c == V* >= 1`.
- `DIFF(C) = EXPECT(C, k-1, J(c+1,a,w,t) - J(c,a,w,t))`
- `DIFF(W_{m*}) = EXPECT(W_{m*}, k-1, J(c,a+1,w,t) - J(c,a,w,t))`
- `UHAT(X, kk) = EXPECT(X, kk, J)`
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from sspo.selector import ClassifiedGroup, classify_candidates, j_minus_subset

ClassSpec = Union[str, Tuple[str, int]]


def comb_exact(n: int, k: int) -> int:
    """Exact integer binomial coefficient `C(n, k)` (returns 0 if k < 0 or k > n)."""
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def J(c: int, a: int, w: int, t: int) -> float:
    """Closed-form selector winning probability `J(c, a, w, t)` under cluster purity."""
    v_star = max(a, w)
    if c == 0 or c < v_star:
        return 0.0
    if c > v_star:
        return 1.0
    # Here c == v_star >= 1
    r = (t if (w == v_star and v_star >= 1) else 0) + (1 if (a == v_star and v_star >= 1) else 0)
    return 1.0 / (1.0 + r)


def COUNTS(
    X: ClassSpec,
    n_C: int,
    n_T: int,
    n_W_clusters: Sequence[int],
) -> Tuple[int, int, int, List[int]]:
    """Compute remaining counts `(n'_C, n'_T, n'_d, n'_{W,1..M'})` after removing one member of class `X`."""
    w_list = [int(x) for x in n_W_clusters]
    if X == "C":
        return n_C - 1, n_T, 0, w_list
    if X == "T":
        return n_C, n_T - 1, 0, w_list
    if X == "W*":
        return n_C, n_T, 0, w_list
    if isinstance(X, tuple) and len(X) == 2 and X[0] == "W":
        m_star = int(X[1])
        n_prime_d = w_list[m_star] - 1
        other_w = [w_list[m] for m in range(len(w_list)) if m != m_star]
        return n_C, n_T, n_prime_d, other_w
    raise ValueError(f"Unknown class specification: {X!r}")


def TABLE(w: int, n_prime_W: Sequence[int], max_s: int) -> List[List[int]]:
    """Compute `N_w(s, t)` for `0 <= s <= max_s` and `0 <= t <= M'` using exact integer arithmetic.

    `N_w(s, t)` is the number of ways to draw `s` items in total from the `M'` clusters `n_prime_W`
    such that every cluster has at most `w` items, and exactly `t` clusters have `w` items.
    For `w == 0`: `N_0(0, 0) = 1` and `0` elsewhere.
    For `w >= 1`: `N_w(s, t) = D(M', s, t)` for `t >= 1` and `0` for `t == 0`.
    """
    m_prime = len(n_prime_W)
    if w == 0:
        nw = [[0] * (m_prime + 1) for _ in range(max_s + 1)]
        nw[0][0] = 1
        return nw

    # D[s][t] for current prefix of clusters
    dp = [[0] * (m_prime + 1) for _ in range(max_s + 1)]
    dp[0][0] = 1

    s_upper = 0
    for m_idx, n_m in enumerate(n_prime_W, start=1):
        if n_m <= 0:
            continue
        max_j = min(w, n_m, max_s)
        new_s_upper = min(max_s, s_upper + max_j)
        new_dp = [[0] * (m_prime + 1) for _ in range(max_s + 1)]
        for j in range(max_j + 1):
            c_mj = comb_exact(n_m, j)
            if c_mj == 0:
                continue
            dt = 1 if j == w else 0
            for s_prev in range(0, min(s_upper, max_s - j) + 1):
                row_prev = dp[s_prev]
                row_new = new_dp[s_prev + j]
                for t_prev in range(0, m_idx):
                    val = row_prev[t_prev]
                    if val:
                        row_new[t_prev + dt] += c_mj * val
        dp = new_dp
        s_upper = new_s_upper

    # For w >= 1, t must be >= 1
    for s in range(max_s + 1):
        dp[s][0] = 0
    return dp


def EXPECT(
    X: ClassSpec,
    kk: int,
    phi: Callable[[int, int, int, int], float],
    n: int,
    n_C: int,
    n_T: int,
    n_W_clusters: Sequence[int],
) -> float:
    """Compute multivariate hypergeometric expectation `EXPECT(X, kk, phi)`."""
    if kk < 0 or n - 1 < kk:
        return 0.0
    denom = comb_exact(n - 1, kk)
    if denom == 0:
        return 0.0

    n_prime_C, n_prime_T, n_prime_d, n_prime_W = COUNTS(X, n_C, n_T, n_W_clusters)
    if n_prime_C < 0 or n_prime_T < 0 or n_prime_d < 0 or any(x < 0 for x in n_prime_W):
        return 0.0

    m_prime = len(n_prime_W)
    max_w_possible = min([kk, max(n_prime_W, default=0)])
    total_w_pool = sum(n_prime_W)

    # Precompute C(n'_C, c), C(n'_d, a), C(n'_T, r)
    max_c = min(kk, n_prime_C)
    max_a = min(kk, n_prime_d)
    comb_C = [comb_exact(n_prime_C, c) for c in range(max_c + 1)]
    comb_d = [comb_exact(n_prime_d, a) for a in range(max_a + 1)]
    comb_T = [comb_exact(n_prime_T, r) for r in range(kk + 1)]

    total_num = 0.0
    for w in range(0, max_w_possible + 1):
        max_s_w = min(kk, total_w_pool, w * m_prime)
        nw_table = TABLE(w, n_prime_W, max_s_w)
        t_range = [0] if w == 0 else list(range(1, m_prime + 1))

        for s in range(0, max_s_w + 1):
            rem_after_s = kk - s
            if rem_after_s < 0:
                break
            for t in t_range:
                nw_st = nw_table[s][t]
                if nw_st == 0:
                    continue
                for c in range(0, min(max_c, rem_after_s) + 1):
                    c_C = comb_C[c]
                    if c_C == 0:
                        continue
                    rem_after_sc = rem_after_s - c
                    # Check bounds on a: we need rem_after_sc - a <= n_prime_T => a >= rem_after_sc - n_prime_T
                    a_min = max(0, rem_after_sc - n_prime_T)
                    a_max = min(max_a, rem_after_sc)
                    if a_min > a_max:
                        continue
                    base_wt = float(nw_st * c_C)
                    for a in range(a_min, a_max + 1):
                        phi_val = phi(c, a, w, t)
                        if phi_val == 0.0:
                            continue
                        ways = base_wt * float(comb_d[a] * comb_T[rem_after_sc - a])
                        total_num += ways * phi_val

    return total_num / float(denom)


def DIFF(
    X: ClassSpec,
    n: int,
    k: int,
    n_C: int,
    n_T: int,
    n_W_clusters: Sequence[int],
) -> float:
    """Compute `DIFF(C)` or `DIFF(W_{m*})`."""
    if X == "C":
        return EXPECT(
            "C",
            k - 1,
            lambda c, a, w, t: J(c + 1, a, w, t) - J(c, a, w, t),
            n=n,
            n_C=n_C,
            n_T=n_T,
            n_W_clusters=n_W_clusters,
        )
    if isinstance(X, tuple) and len(X) == 2 and X[0] == "W":
        return EXPECT(
            X,
            k - 1,
            lambda c, a, w, t: J(c, a + 1, w, t) - J(c, a, w, t),
            n=n,
            n_C=n_C,
            n_T=n_T,
            n_W_clusters=n_W_clusters,
        )
    if X == "T":
        return 0.0
    if X == "W*":
        return -UHAT("W*", k - 1, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_clusters)
    raise ValueError(f"Unsupported class for DIFF: {X!r}")


def UHAT(
    X: ClassSpec,
    kk: int,
    n: int,
    n_C: int,
    n_T: int,
    n_W_clusters: Sequence[int],
) -> float:
    """Compute leave-one-out U-statistic `UHAT(X, kk) = EXPECT(X, kk, J)`."""
    return EXPECT(X, kk, J, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_clusters)


def dp_selection_advantages(group: ClassifiedGroup, k: int) -> np.ndarray:
    """Compute `A_i^(S, U)` for all `i in 0..n-1` via exact dynamic programming."""
    n = group.n
    adv_s = np.zeros(n, dtype=np.float64)
    if n == 0 or k < 1 or k > n:
        return adv_s

    n_C = group.n_C
    n_T = group.n_T
    n_W_list = group.n_W_clusters

    val_C: Optional[float] = None
    val_W_star: Optional[float] = None
    val_W_by_size: Dict[int, float] = {}

    for i in range(n):
        cls_name, m_idx = group.candidate_class[i]
        if cls_name == "T":
            adv_s[i] = 0.0
        elif cls_name == "C":
            if val_C is None:
                val_C = DIFF("C", n=n, k=k, n_C=n_C, n_T=n_T, n_W_clusters=n_W_list)
            adv_s[i] = val_C
        elif cls_name == "W*":
            if val_W_star is None:
                val_W_star = -UHAT("W*", k - 1, n=n, n_C=n_C, n_T=n_T, n_W_clusters=n_W_list)
            adv_s[i] = val_W_star
        elif cls_name == "W":
            c_size = n_W_list[m_idx]
            if c_size not in val_W_by_size:
                val_W_by_size[c_size] = DIFF(("W", m_idx), n=n, k=k, n_C=n_C, n_T=n_T, n_W_clusters=n_W_list)
            adv_s[i] = val_W_by_size[c_size]

    return adv_s


def brute_force_selection_advantages(
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k: int,
    sigma_plus: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force enumerate all `C(n-1, k-1)` subsets (e.g. `C(15, 7) = 6435` for `n=16, k=8`) for `J^-`.

    Returns:
      - `adv_s`: array of shape `(n,)` with `A_i^(S, U) = (1/C(n-1,k-1)) sum_{K'} [J^-(K' cup {i}) - J^-(K')]`
      - `uhat_k_minus_1`: array of shape `(n,)` with `U^-_{k-1}(Y_{-i})`
      - `uhat_k`: array of shape `(n,)` with `U^-_k(Y_{-i})`
    """
    n = len(g_list)
    adv_s = np.zeros(n, dtype=np.float64)
    uhat_km1 = np.zeros(n, dtype=np.float64)
    uhat_k = np.zeros(n, dtype=np.float64)
    if n == 0 or k < 1 or k > n:
        return adv_s, uhat_km1, uhat_k

    sigma_plus_set = set(sigma_plus) if sigma_plus is not None else set()
    w_star_set = {
        i
        for i in range(n)
        if int(g_list[i]) == 0 and int(chk_list[i]) == 1 and str(sig_list[i]) in sigma_plus_set
    }

    denom_km1 = comb_exact(n - 1, k - 1)
    denom_k = comb_exact(n - 1, k)

    # Sum J^-(K) over all subsets of size k
    sum_with_i = np.zeros(n, dtype=np.float64)
    total_size_k = 0.0
    for comb_k in itertools.combinations(range(n), k):
        val = j_minus_subset(comb_k, g_list, chk_list, sig_list, w_star_set)
        if val != 0.0:
            total_size_k += val
            for i in comb_k:
                sum_with_i[i] += val

    # Sum J^-(K') over all subsets of size k - 1
    sum_with_i_km1 = np.zeros(n, dtype=np.float64)
    total_size_km1 = 0.0
    if k - 1 > 0:
        for comb_km1 in itertools.combinations(range(n), k - 1):
            val = j_minus_subset(comb_km1, g_list, chk_list, sig_list, w_star_set)
            if val != 0.0:
                total_size_km1 += val
                for i in comb_km1:
                    sum_with_i_km1[i] += val

    # For each i:
    # sum_{K' subset Y_{-i}, |K'|=k-1} J^-(K') = total_size_km1 - sum_with_i_km1[i]
    # sum_{K subset Y_{-i}, |K|=k} J^-(K) = total_size_k - sum_with_i[i]
    for i in range(n):
        without_i_km1 = total_size_km1 - sum_with_i_km1[i]
        uhat_km1[i] = without_i_km1 / float(denom_km1) if denom_km1 > 0 else 0.0
        adv_s[i] = (sum_with_i[i] - without_i_km1) / float(denom_km1) if denom_km1 > 0 else 0.0
        if denom_k > 0:
            uhat_k[i] = (total_size_k - sum_with_i[i]) / float(denom_k)

    return adv_s, uhat_km1, uhat_k


def brute_force_single_candidate(
    i: int,
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k: int,
    sigma_plus: Optional[Set[str]] = None,
) -> float:
    """Explicitly enumerate all `C(n-1, k-1)` subsets `K' subset [n] \\ {i}` for candidate `i`."""
    n = len(g_list)
    denom = comb_exact(n - 1, k - 1)
    if denom == 0:
        return 0.0
    sigma_plus_set = set(sigma_plus) if sigma_plus is not None else set()
    w_star_set = {
        idx
        for idx in range(n)
        if int(g_list[idx]) == 0 and int(chk_list[idx]) == 1 and str(sig_list[idx]) in sigma_plus_set
    }
    others = [idx for idx in range(n) if idx != i]
    diff_sum = 0.0
    for k_prime in itertools.combinations(others, k - 1):
        v_without = j_minus_subset(k_prime, g_list, chk_list, sig_list, w_star_set)
        v_with = j_minus_subset((*k_prime, i), g_list, chk_list, sig_list, w_star_set)
        diff_sum += v_with - v_without
    return diff_sum / float(denom)
