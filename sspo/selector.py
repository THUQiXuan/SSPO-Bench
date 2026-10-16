"""Execution selector `S` (hard filter `Chk=1` + signature majority vote) and subset score functions.

Definitions:
- Classes:
  * `C = {g = 1, Chk = 1}`
  * `T = {Chk = 0}`
  * `W* = {g = 0, Chk = 1, sig in Sigma^+_x}`
  * `W_1 .. W_M` = remaining `{g = 0, Chk = 1, sig not in Sigma^+_x}` partitioned by `sig`
- Selector `S`:
  Filters out `T` (`Chk = 0`), partitions remaining candidates by `sig(y)`, chooses a largest cluster uniformly
  at random (breaking ties uniformly among largest clusters), and submits uniformly at random within the chosen cluster.
- Set scores:
  * `J_S(K)`: probability that selector `S` submits a `C` candidate (`g = 1, Chk = 1`) on subset `K`.
  * `J^-(K) = J_S(K) * 1[K cap W* == empty]`.
  * `J_cov(K) = 1[K cap C != empty]`.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class ClassifiedGroup:
    """Partition of `n` rollout candidates for a problem `x`."""

    n: int
    c_indices: List[int]
    t_indices: List[int]
    w_star_indices: List[int]
    w_clusters: List[List[int]]  # W_1 .. W_M (sorted by cluster size descending)
    w_cluster_sigs: List[str]
    candidate_class: List[Tuple[str, int]]  # per i: ('C', 0), ('T', 0), ('W*', 0), or ('W', m_idx)
    c_sigs: Set[str]
    cluster_impure: bool  # True if |c_sigs| > 1 or any W_m shares a signature with C
    eps_hat: float  # fraction of candidates with Chk=0 and g=1
    chk_pass_rate: float  # fraction of candidates with Chk=1

    @property
    def n_C(self) -> int:
        return len(self.c_indices)

    @property
    def n_T(self) -> int:
        return len(self.t_indices)

    @property
    def n_W_star(self) -> int:
        return len(self.w_star_indices)

    @property
    def n_W_clusters(self) -> List[int]:
        return [len(c) for c in self.w_clusters]

    @property
    def n_W_total(self) -> int:
        return self.n_W_star + sum(len(c) for c in self.w_clusters)


def classify_candidates(
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    sigma_plus: Optional[Iterable[str]] = None,
) -> ClassifiedGroup:
    """Partition `n` candidates into `C`, `T`, `W*`, and `W_1 .. W_M`."""
    n = len(g_list)
    assert len(chk_list) == n and len(sig_list) == n
    sigma_plus_set: Set[str] = set(sigma_plus) if sigma_plus is not None else set()

    c_indices: List[int] = []
    t_indices: List[int] = []
    w_star_indices: List[int] = []
    w_by_sig: Dict[str, List[int]] = collections.defaultdict(list)
    c_sigs: Set[str] = set()
    n_false_reject = 0

    for i in range(n):
        g_i = int(g_list[i])
        chk_i = int(chk_list[i])
        sig_i = str(sig_list[i])
        if chk_i == 0:
            t_indices.append(i)
            if g_i == 1:
                n_false_reject += 1
        elif g_i == 1:
            c_indices.append(i)
            c_sigs.add(sig_i)
        else:
            if sig_i in sigma_plus_set:
                w_star_indices.append(i)
            else:
                w_by_sig[sig_i].append(i)

    # Sort wrong clusters by size descending, then by signature for determinism
    sorted_w_items = sorted(w_by_sig.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    w_cluster_sigs = [kv[0] for kv in sorted_w_items]
    w_clusters = [kv[1] for kv in sorted_w_items]

    sig_to_m = {sig: m_idx for m_idx, sig in enumerate(w_cluster_sigs)}
    candidate_class: List[Tuple[str, int]] = [("", 0)] * n
    for i in c_indices:
        candidate_class[i] = ("C", 0)
    for i in t_indices:
        candidate_class[i] = ("T", 0)
    for i in w_star_indices:
        candidate_class[i] = ("W*", 0)
    for m_idx, cluster in enumerate(w_clusters):
        for i in cluster:
            candidate_class[i] = ("W", m_idx)

    # Cluster impurity occurs if C spans multiple signatures OR some W_m shares a signature with C
    c_multi_sig = len(c_sigs) > 1
    w_overlaps_c = any(sig in c_sigs for sig in sig_to_m)
    cluster_impure = c_multi_sig or w_overlaps_c

    eps_hat = (n_false_reject / n) if n > 0 else 0.0
    chk_pass_rate = ((len(c_indices) + len(w_star_indices) + sum(len(c) for c in w_clusters)) / n) if n > 0 else 0.0

    return ClassifiedGroup(
        n=n,
        c_indices=c_indices,
        t_indices=t_indices,
        w_star_indices=w_star_indices,
        w_clusters=w_clusters,
        w_cluster_sigs=w_cluster_sigs,
        candidate_class=candidate_class,
        c_sigs=c_sigs,
        cluster_impure=cluster_impure,
        eps_hat=eps_hat,
        chk_pass_rate=chk_pass_rate,
    )


def j_cov_subset(
    subset: Sequence[int],
    g_list: Sequence[int],
    chk_list: Sequence[int],
) -> float:
    """Checked coverage `J_cov(K) = 1[K contains at least one C candidate (g=1, Chk=1)]`."""
    for i in subset:
        if int(g_list[i]) == 1 and int(chk_list[i]) == 1:
            return 1.0
    return 0.0


def pass_subset(
    subset: Sequence[int],
    g_list: Sequence[int],
) -> float:
    """True pass indicator `1[K contains at least one g=1 candidate]`."""
    for i in subset:
        if int(g_list[i]) == 1:
            return 1.0
    return 0.0


def j_s_subset(
    subset: Sequence[int],
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
) -> float:
    """Exact probability `J_S(K)` that selector `S` submits a correct program (`C`) on subset `K`.

    Selector `S`:
    1. Filter out `T` (`Chk = 0`). If no candidate passes `Chk`, return 0.0.
    2. Partition remaining candidates by `sig`.
    3. Among clusters with maximum size `v_max`, choose one cluster uniformly at random,
       and choose one candidate inside that cluster uniformly at random.
    """
    clusters: Dict[str, List[int]] = {}
    for i in subset:
        if int(chk_list[i]) == 1:
            s = str(sig_list[i])
            if s not in clusters:
                clusters[s] = [i]
            else:
                clusters[s].append(i)

    if not clusters:
        return 0.0

    v_max = max(len(members) for members in clusters.values())
    winners = [members for members in clusters.values() if len(members) == v_max]
    total_prob = 0.0
    for members in winners:
        n_correct = sum(1 for i in members if int(g_list[i]) == 1)
        total_prob += n_correct / len(members)
    return total_prob / len(winners)


def j_minus_subset(
    subset: Sequence[int],
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    w_star_set: Set[int],
) -> float:
    """Conservative selector objective `J^-(K) = J_S(K) * 1[K cap W* == empty]` (Proposition D)."""
    if w_star_set:
        for i in subset:
            if i in w_star_set:
                return 0.0
    return j_s_subset(subset, g_list, chk_list, sig_list)


def select_candidate_index(
    subset: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    rng: Optional[random.Random] = None,
) -> int:
    """Run selector `S` on candidate indices `subset` to return one concrete selected index."""
    if not subset:
        raise ValueError("Cannot select from an empty candidate subset")
    r = rng or random.Random(0)
    passed = [i for i in subset if int(chk_list[i]) == 1]
    if not passed:
        # All candidates in T: uniform fallback among subset
        return r.choice(list(subset))
    clusters: Dict[str, List[int]] = collections.defaultdict(list)
    for i in passed:
        clusters[str(sig_list[i])].append(i)
    v_max = max(len(m) for m in clusters.values())
    winning_sigs = sorted(s for s, m in clusters.items() if len(m) == v_max)
    chosen_sig = r.choice(winning_sigs)
    return r.choice(clusters[chosen_sig])
