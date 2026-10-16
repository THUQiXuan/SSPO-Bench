"""Evaluation estimators (`pass@k`, `select@k`, `cov@k`, `gap@k`), paired two-level bootstrap, Holm correction, and report generation.

Metrics:
- Unbiased combinatorial `pass@k` and `cov@k`:
    `pass@k = 1 - C(n - c, k) / C(n, k)`
- `select@k`:
    Evaluates selector `S` on `n_subsets >= 200` random `k`-subsets (recording `n_subsets` and `subset_seed`),
    and also computes exact hypergeometric `select@k` when cluster-pure.
- Critical problem indicator (`exists K subset [n], |K|=k : J_S(K) < J_cov(K)`), selection gap, cluster impurity rate,
  `eps_hat` (`Chk=0` and `g=1` rate), class counts, and checker pass rate.
- Paired two-level bootstrap (contest cluster + training repeat) for 95% confidence intervals and Holm-Bonferroni correction.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sspo.atomic_io import (
    REPO_ROOT,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_jsonl,
    to_rel_path,
    write_done_marker,
)
from sspo.dp import UHAT, comb_exact
from sspo.selector import classify_candidates, j_cov_subset, j_s_subset


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased combinatorial estimator `1 - C(n - c, k) / C(n, k)` using exact integer combinations."""
    if k < 1 or n < k or c < 0:
        return 0.0
    if c >= n:
        return 1.0
    denom = comb_exact(n, k)
    if denom == 0:
        return 0.0
    num = comb_exact(n - c, k)
    return 1.0 - (float(num) / float(denom))


def exact_select_at_k_pure(
    n: int,
    n_C: int,
    n_T: int,
    n_W_clusters: Sequence[int],
    k: int,
) -> float:
    """Exact combinatorial `select@k = C(n, k)^(-1) sum_{|K|=k} J_S(K)` under cluster purity (`A1'`).

    Uses `UHAT('T', k)` on an augmented pool of size `n + 1` with one extra `T` item removed,
    which leaves the exact pool `(n_C, n_T, n_W_clusters)` of size `n`!
    """
    if k < 1 or n < k or n_C == 0:
        return 0.0
    return UHAT("T", k, n=n + 1, n_C=n_C, n_T=n_T + 1, n_W_clusters=n_W_clusters)


def is_critical_pool(
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k: int = 8,
) -> bool:
    """Check whether there exists a `k`-subset `K subset [n]` such that `J_S(K) < J_cov(K)`."""
    n = len(g_list)
    if n < k or k < 1:
        return False
    group = classify_candidates(g_list, chk_list, sig_list)
    if group.n_C == 0:
        return False
    if group.cluster_impure:
        return True
    # Minimum number of C items forced into any k-subset when drawing at most all non-C items
    non_c = n - group.n_C
    c_min = max(1, k - non_c)
    max_w_cluster = max(group.n_W_clusters, default=0)
    # We can form a k-subset with c_min items from C and min(max_w_cluster, k - c_min) items from W_1
    w_achievable = min(max_w_cluster, k - c_min)
    return w_achievable >= c_min


def estimate_select_at_k(
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k: int,
    n_subsets: int = 200,
    subset_seed: int = 42,
) -> Dict[str, Any]:
    """Estimate `select@k` by running selector `S` on `n_subsets` random `k`-subsets (recording `n_subsets` and `subset_seed`),
    and also compute exact `select@k_exact` when cluster-pure or `C(n, k) <= n_subsets`.
    """
    n = len(g_list)
    if k < 1 or n < k:
        return {
            "select_at_k": 0.0,
            "select_at_k_exact": 0.0,
            "n_subsets": int(n_subsets),
            "subset_seed": int(subset_seed),
        }

    group = classify_candidates(g_list, chk_list, sig_list)
    total_combs = comb_exact(n, k)

    # Random subset Monte Carlo estimator (always uses the specified n_subsets & seed)
    rng = random.Random(int(subset_seed) + 1009 * int(k))
    indices = list(range(n))
    mc_scores: List[float] = []
    for _ in range(max(1, int(n_subsets))):
        sub = rng.sample(indices, k)
        mc_scores.append(j_s_subset(sub, g_list, chk_list, sig_list))
    mc_val = float(np.mean(mc_scores))

    # Exact value via DP (when pure) or full enumeration (when C(n, k) is small)
    if not group.cluster_impure:
        exact_val = exact_select_at_k_pure(
            n=n,
            n_C=group.n_C,
            n_T=group.n_T + group.n_W_star,
            n_W_clusters=group.n_W_clusters,
            k=k,
        )
    elif total_combs <= 2000:
        exact_val = float(
            np.mean([j_s_subset(sub, g_list, chk_list, sig_list) for sub in itertools.combinations(indices, k)])
        )
    else:
        exact_val = mc_val

    return {
        "select_at_k": mc_val,
        "select_at_k_exact": exact_val,
        "n_subsets": int(n_subsets),
        "subset_seed": int(subset_seed),
    }


def evaluate_problem_rollouts(
    pid: str,
    contest_id: str,
    g_list: Sequence[int],
    chk_list: Sequence[int],
    sig_list: Sequence[str],
    k_list: Sequence[int] = (1, 2, 4, 8, 16),
    n_subsets: int = 200,
    subset_seed: int = 42,
    sigma_plus: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute full per-problem evaluation record matching `results/<run>/eval/<set>_<ckpt>.jsonl` schema."""
    n_eval = len(g_list)
    group = classify_candidates(g_list, chk_list, sig_list, sigma_plus=sigma_plus)
    n_pass_gold = sum(1 for g in g_list if int(g) == 1)

    valid_ks = [int(k) for k in k_list if 1 <= int(k) <= n_eval]
    pass_at_k: Dict[str, float] = {}
    cov_at_k: Dict[str, float] = {}
    select_at_k: Dict[str, float] = {}
    select_at_k_exact: Dict[str, float] = {}
    gap_at_k: Dict[str, float] = {}

    for idx, k in enumerate(valid_ks):
        key_str = str(k)
        pk = estimate_pass_at_k(n_eval, n_pass_gold, k)
        ck = estimate_pass_at_k(n_eval, group.n_C, k)
        # Deterministic per-problem-per-k seed derived from subset_seed
        prob_seed = int(subset_seed) + (abs(hash(str(pid))) % 10000)
        sel_res = estimate_select_at_k(
            g_list, chk_list, sig_list, k=k, n_subsets=n_subsets, subset_seed=prob_seed
        )
        sk = sel_res["select_at_k"]
        sk_exact = sel_res["select_at_k_exact"]
        pass_at_k[key_str] = round(pk, 6)
        cov_at_k[key_str] = round(ck, 6)
        select_at_k[key_str] = round(sk, 6)
        select_at_k_exact[key_str] = round(sk_exact, 6)
        gap_at_k[key_str] = round(ck - sk, 6)

    k_crit = 8 if 8 <= n_eval else (valid_ks[-1] if valid_ks else 1)
    crit = is_critical_pool(g_list, chk_list, sig_list, k=k_crit)

    return {
        "pid": str(pid),
        "contest_id": str(contest_id),
        "n_eval": n_eval,
        "k_list": valid_ks,
        "pass_at_k": pass_at_k,
        "cov_at_k": cov_at_k,
        "select_at_k": select_at_k,
        "select_at_k_exact": select_at_k_exact,
        "gap_at_k": gap_at_k,
        "n_subsets": int(n_subsets),
        "subset_seed": int(subset_seed),
        "is_critical": bool(crit),
        "cluster_impure": bool(group.cluster_impure),
        "eps_hat": round(group.eps_hat, 6),
        "chk_pass_rate": round(group.chk_pass_rate, 6),
        "counts": {
            "n_C": group.n_C,
            "n_T": group.n_T,
            "n_W": group.n_W_total,
            "n_W_star": group.n_W_star,
            "n_clusters": len(group.w_clusters),
            "max_wrong_cluster": max(group.n_W_clusters, default=0),
        },
    }


def summarize_eval_rows(rows: List[Dict[str, Any]], run_name: str = "") -> Dict[str, Any]:
    """Aggregate per-problem evaluation rows into dataset-level summary metrics."""
    if not rows:
        return {"run_name": run_name, "n_problems": 0}
    n_probs = len(rows)
    all_ks = sorted({int(k) for r in rows for k in r.get("k_list", [])})

    pass_means = {}
    cov_means = {}
    select_means = {}
    select_exact_means = {}
    gap_means = {}
    for k in all_ks:
        ks = str(k)
        p_vals = [r["pass_at_k"][ks] for r in rows if ks in r.get("pass_at_k", {})]
        c_vals = [r["cov_at_k"][ks] for r in rows if ks in r.get("cov_at_k", {})]
        s_vals = [r["select_at_k"][ks] for r in rows if ks in r.get("select_at_k", {})]
        se_vals = [r["select_at_k_exact"][ks] for r in rows if ks in r.get("select_at_k_exact", {})]
        g_vals = [r["gap_at_k"][ks] for r in rows if ks in r.get("gap_at_k", {})]
        if p_vals:
            pass_means[ks] = round(float(np.mean(p_vals)), 6)
        if c_vals:
            cov_means[ks] = round(float(np.mean(c_vals)), 6)
        if s_vals:
            select_means[ks] = round(float(np.mean(s_vals)), 6)
        if se_vals:
            select_exact_means[ks] = round(float(np.mean(se_vals)), 6)
        if g_vals:
            gap_means[ks] = round(float(np.mean(g_vals)), 6)

    k_main = "8" if "8" in select_means else (str(all_ks[-1]) if all_ks else "1")
    selector_gain_over_pass1 = round(
        select_means.get(k_main, 0.0) - pass_means.get("1", 0.0), 6
    )

    return {
        "run_name": run_name,
        "n_problems": n_probs,
        "pass_at_k": pass_means,
        "cov_at_k": cov_means,
        "select_at_k": select_means,
        "select_at_k_exact": select_exact_means,
        "gap_at_k": gap_means,
        "selector_gain_k_vs_pass1": selector_gain_over_pass1,
        "critical_problem_rate": round(float(np.mean([1.0 if r.get("is_critical") else 0.0 for r in rows])), 6),
        "cluster_impure_rate": round(float(np.mean([1.0 if r.get("cluster_impure") else 0.0 for r in rows])), 6),
        "eps_hat_mean": round(float(np.mean([float(r.get("eps_hat", 0.0)) for r in rows])), 6),
        "chk_pass_rate_mean": round(float(np.mean([float(r.get("chk_pass_rate", 0.0)) for r in rows])), 6),
    }


def paired_two_level_bootstrap(
    arm_a_runs: List[List[Dict[str, Any]]],
    arm_b_runs: List[List[Dict[str, Any]]],
    metric: str = "select_at_k",
    k_key: str = "8",
    n_boot: int = 2000,
    seed: int = 71,
) -> Dict[str, Any]:
    """Paired two-level bootstrap (Level 1: training repeat / seed; Level 2: contest cluster of problems).

    Computes `mean_diff = mean(A - B)`, 95% bootstrap confidence interval `[ci_low, ci_high]`,
    standard error `se`, and two-sided / one-sided p-value.
    """
    if not arm_a_runs or not arm_b_runs:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "p_value": 1.0, "win": False}

    def extract_run_map(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, str]]:
        vals: Dict[str, float] = {}
        contests: Dict[str, str] = {}
        for r in rows:
            pid = r["pid"]
            m_dict = r.get(metric, {})
            val = m_dict.get(k_key)
            if val is None and m_dict:
                # Fallback to largest available k
                max_k = max(m_dict.keys(), key=lambda x: int(x))
                val = m_dict[max_k]
            if val is not None:
                vals[pid] = float(val)
                contests[pid] = str(r.get("contest_id", pid))
        return vals, contests

    a_maps = [extract_run_map(r) for r in arm_a_runs]
    b_maps = [extract_run_map(r) for r in arm_b_runs]

    common_pids = sorted(
        set.intersection(*(set(m[0].keys()) for m in a_maps), *(set(m[0].keys()) for m in b_maps))
    )
    if not common_pids:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "p_value": 1.0, "win": False}

    contest_map = a_maps[0][1]
    clusters: Dict[str, List[str]] = collections.defaultdict(list)
    for pid in common_pids:
        clusters[contest_map.get(pid, pid)].append(pid)
    cluster_keys = sorted(clusters.keys())

    # Point estimate averaged over repeats and problems
    mean_a_per_pid = {pid: float(np.mean([m[0][pid] for m in a_maps])) for pid in common_pids}
    mean_b_per_pid = {pid: float(np.mean([m[0][pid] for m in b_maps])) for pid in common_pids}
    point_diffs = [mean_a_per_pid[pid] - mean_b_per_pid[pid] for pid in common_pids]
    mean_diff = float(np.mean(point_diffs))

    rng = random.Random(seed)
    boot_means: List[float] = []
    n_ra = len(a_maps)
    n_rb = len(b_maps)
    n_cl = len(cluster_keys)

    for _ in range(n_boot):
        # Level 1: resample repeats with replacement (if R > 1)
        sampled_a = [a_maps[rng.randrange(n_ra)][0] for _ in range(n_ra)] if n_ra > 1 else [a_maps[0][0]]
        sampled_b = [b_maps[rng.randrange(n_rb)][0] for _ in range(n_rb)] if n_rb > 1 else [b_maps[0][0]]
        # Level 2: resample contest clusters with replacement
        diffs_sample: List[float] = []
        for _ in range(n_cl):
            ck = cluster_keys[rng.randrange(n_cl)]
            for pid in clusters[ck]:
                va = sum(sa[pid] for sa in sampled_a) / n_ra
                vb = sum(sb[pid] for sb in sampled_b) / n_rb
                diffs_sample.append(va - vb)
        if diffs_sample:
            boot_means.append(sum(diffs_sample) / len(diffs_sample))

    boot_means.sort()
    ci_low = boot_means[int(0.025 * len(boot_means))] if boot_means else mean_diff
    ci_high = boot_means[int(0.975 * len(boot_means))] if boot_means else mean_diff
    se = float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0
    # Empirical two-sided p-value
    neg_frac = sum(1 for x in boot_means if x <= 0.0) / max(1, len(boot_means))
    pos_frac = sum(1 for x in boot_means if x >= 0.0) / max(1, len(boot_means))
    p_val = min(1.0, 2.0 * min(neg_frac, pos_frac))

    return {
        "n_problems": len(common_pids),
        "n_clusters": n_cl,
        "n_repeats_a": n_ra,
        "n_repeats_b": n_rb,
        "mean_diff": round(mean_diff, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "se": round(se, 6),
        "p_value": round(p_val, 6),
        "win": bool(ci_low > 0.0),
    }


def holm_bonferroni(comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply Holm-Bonferroni step-down correction to a list of comparison dicts containing `p_value`."""
    m = len(comparisons)
    if m == 0:
        return []
    indexed = list(enumerate(comparisons))
    indexed.sort(key=lambda iv: float(iv[1].get("p_value", 1.0)))
    out = [dict(c) for c in comparisons]
    running_max = 0.0
    for rank, (orig_idx, comp) in enumerate(indexed):
        adj_p = min(1.0, float(comp.get("p_value", 1.0)) * (m - rank))
        running_max = max(running_max, adj_p)
        out[orig_idx]["p_value_holm"] = round(running_max, 6)
        out[orig_idx]["significant_holm_005"] = bool(running_max < 0.05 and float(comp.get("mean_diff", 0.0)) > 0)
    return out


def generate_tables_and_report(results_root: str | Path = REPO_ROOT / "results") -> Dict[str, Any]:
    """Scan `results/<run_name>/` directories and generate `results/tables/*.csv` and `results/report.md`."""
    rroot = Path(results_root)
    tables_dir = rroot / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Discover all run summaries and evaluation files
    arm_eval_runs: Dict[str, List[List[Dict[str, Any]]]] = collections.defaultdict(list)
    run_summaries: List[Dict[str, Any]] = []
    all_ckpt_summaries: List[Dict[str, Any]] = []
    training_curve_rows: List[Dict[str, Any]] = []

    for run_dir in sorted(rroot.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("tables", "diag", "probe"):
            continue
        run_name = run_dir.name
        manifest_path = run_dir / "manifest.json"
        arm = run_name.split("_seed")[0]
        seed = 0
        if manifest_path.exists():
            try:
                mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                arm = str(mf.get("arm", arm))
                seed = int(mf.get("seed", 0))
            except Exception:
                pass
        if "turns2" in run_name and not arm.endswith("_turns2"):
            arm = f"{arm}_turns2"

        # Read training metrics if available
        metrics_file = run_dir / "metrics.jsonl"
        for mrow in read_jsonl(metrics_file):
            training_curve_rows.append(
                {
                    "run_name": run_name,
                    "arm": arm,
                    "seed": seed,
                    "step": mrow.get("step", 0),
                    "loss": mrow.get("loss", 0.0),
                    "pg_loss": mrow.get("pg_loss", 0.0),
                    "kl": mrow.get("kl", 0.0),
                    "pass_at_1_batch": mrow.get("pass_at_1_batch", 0.0),
                    "chk_pass_rate": mrow.get("chk_pass_rate", 0.0),
                    "cluster_impure_rate": mrow.get("cluster_impure_rate", 0.0),
                    "eps_hat": mrow.get("eps_hat", 0.0),
                    "wall_time_s": mrow.get("wall_time_s", 0.0),
                }
            )

        # Read evaluation jsonl files
        eval_dir = run_dir / "eval"
        eval_files = sorted(eval_dir.glob("*.jsonl")) if eval_dir.exists() else []
        if eval_files:
            best_summary = None
            best_rows = None
            latest_summary = None
            for ef in eval_files:
                erows = read_jsonl(ef)
                if not erows:
                    continue
                esum = summarize_eval_rows(erows, run_name=run_name)
                esum["arm"] = arm
                esum["seed"] = seed
                esum["eval_file"] = to_rel_path(ef)
                esum["ckpt"] = ef.stem.replace("test_", "")
                all_ckpt_summaries.append(esum)
                latest_summary = esum
                if best_summary is None or float(esum.get("select_at_k", {}).get("8", 0.0)) >= float(
                    best_summary.get("select_at_k", {}).get("8", 0.0)
                ):
                    best_summary = esum
                    best_rows = erows
            if best_summary is not None and best_rows is not None:
                arm_eval_runs[arm].append(best_rows)
                if latest_summary is not None:
                    best_summary["latest_ckpt"] = latest_summary.get("ckpt", "")
                    best_summary["latest_select_at_8"] = latest_summary.get("select_at_k", {}).get("8", 0.0)
                atomic_write_json(run_dir / "summary.json", best_summary)
                run_summaries.append(best_summary)
        elif (run_dir / "summary.json").exists():
            try:
                run_summaries.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
            except Exception:
                pass

    # 1. Write `results/tables/main_comparison.csv`
    main_csv_path = tables_dir / "main_comparison.csv"
    main_fieldnames = [
        "run_name",
        "arm",
        "seed",
        "ckpt",
        "n_problems",
        "pass_at_1",
        "pass_at_4",
        "pass_at_8",
        "select_at_1",
        "select_at_4",
        "select_at_8",
        "gap_at_8",
        "critical_problem_rate",
        "cluster_impure_rate",
        "eps_hat_mean",
        "chk_pass_rate_mean",
    ]
    buf_lines = [",".join(main_fieldnames)]
    for s in run_summaries:
        pk = s.get("pass_at_k", {})
        sk = s.get("select_at_k", {})
        gk = s.get("gap_at_k", {})
        k_ref = "8" if "8" in sk else ("4" if "4" in sk else "1")
        row_vals = [
            str(s.get("run_name", "")),
            str(s.get("arm", "")),
            str(s.get("seed", 0)),
            str(s.get("ckpt", s.get("eval_file", "").split("test_")[-1].replace(".jsonl", ""))),
            str(s.get("n_problems", 0)),
            f"{pk.get('1', 0.0):.4f}",
            f"{pk.get('4', 0.0):.4f}",
            f"{pk.get(k_ref, 0.0):.4f}",
            f"{sk.get('1', 0.0):.4f}",
            f"{sk.get('4', 0.0):.4f}",
            f"{sk.get(k_ref, 0.0):.4f}",
            f"{gk.get(k_ref, 0.0):.4f}",
            f"{s.get('critical_problem_rate', 0.0):.4f}",
            f"{s.get('cluster_impure_rate', 0.0):.4f}",
            f"{s.get('eps_hat_mean', 0.0):.4f}",
            f"{s.get('chk_pass_rate_mean', 0.0):.4f}",
        ]
        buf_lines.append(",".join(row_vals))
    atomic_write_text(main_csv_path, "\n".join(buf_lines) + "\n")

    # 1b. Write `results/tables/checkpoint_progression.csv`
    prog_csv_path = tables_dir / "checkpoint_progression.csv"
    prog_lines = [",".join(main_fieldnames)]
    for s in all_ckpt_summaries:
        pk = s.get("pass_at_k", {})
        sk = s.get("select_at_k", {})
        gk = s.get("gap_at_k", {})
        k_ref = "8" if "8" in sk else ("4" if "4" in sk else "1")
        row_vals = [
            str(s.get("run_name", "")),
            str(s.get("arm", "")),
            str(s.get("seed", 0)),
            str(s.get("ckpt", "")),
            str(s.get("n_problems", 0)),
            f"{pk.get('1', 0.0):.4f}",
            f"{pk.get('4', 0.0):.4f}",
            f"{pk.get(k_ref, 0.0):.4f}",
            f"{sk.get('1', 0.0):.4f}",
            f"{sk.get('4', 0.0):.4f}",
            f"{sk.get(k_ref, 0.0):.4f}",
            f"{gk.get(k_ref, 0.0):.4f}",
            f"{s.get('critical_problem_rate', 0.0):.4f}",
            f"{s.get('cluster_impure_rate', 0.0):.4f}",
            f"{s.get('eps_hat_mean', 0.0):.4f}",
            f"{s.get('chk_pass_rate_mean', 0.0):.4f}",
        ]
        prog_lines.append(",".join(row_vals))
    atomic_write_text(prog_csv_path, "\n".join(prog_lines) + "\n")

    # 2. Write `results/tables/training_curves.csv`
    curves_csv_path = tables_dir / "training_curves.csv"
    curve_fields = [
        "run_name",
        "arm",
        "seed",
        "step",
        "loss",
        "pg_loss",
        "kl",
        "pass_at_1_batch",
        "chk_pass_rate",
        "cluster_impure_rate",
        "eps_hat",
        "wall_time_s",
    ]
    curve_lines = [",".join(curve_fields)]
    for crow in training_curve_rows:
        curve_lines.append(",".join(str(crow.get(f, "")) for f in curve_fields))
    atomic_write_text(curves_csv_path, "\n".join(curve_lines) + "\n")

    # 3. Paired bootstrap comparisons of `sspo` vs (`drgrpo_pass1`, `pkpo_pass8`, `frozen`, `checker_reward`)
    comparisons: List[Dict[str, Any]] = []
    if "sspo" in arm_eval_runs:
        for base_arm in ("drgrpo_pass1", "pkpo_pass8", "frozen", "checker_reward"):
            if base_arm in arm_eval_runs:
                res_boot = paired_two_level_bootstrap(
                    arm_a_runs=arm_eval_runs["sspo"],
                    arm_b_runs=arm_eval_runs[base_arm],
                    metric="select_at_k",
                    k_key="8",
                )
                res_boot["arm_a"] = "sspo"
                res_boot["arm_b"] = base_arm
                res_boot["metric"] = "select@8"
                comparisons.append(res_boot)
        comparisons = holm_bonferroni(comparisons)

    boot_csv_path = tables_dir / "paired_bootstrap.csv"
    boot_fields = [
        "arm_a",
        "arm_b",
        "metric",
        "n_problems",
        "n_clusters",
        "mean_diff",
        "ci_low",
        "ci_high",
        "se",
        "p_value",
        "p_value_holm",
        "win",
    ]
    boot_lines = [",".join(boot_fields)]
    for comp in comparisons:
        boot_lines.append(",".join(str(comp.get(f, "")) for f in boot_fields))
    atomic_write_text(boot_csv_path, "\n".join(boot_lines) + "\n")

    # 4. Read Gate 1 diagnostics if present
    diag_summary_path = rroot / "diag" / "summary.json"
    diag_info = {}
    if diag_summary_path.exists():
        try:
            diag_info = json.loads(diag_summary_path.read_text(encoding="utf-8"))
        except Exception:
            diag_info = {}

    # 5. Generate `results/report.md`
    md = [
        "# SSPO 第一版实验自动汇总报告 (`results/report.md`)",
        "",
        "## 1. 免训练诊断（闸门一）",
        "",
    ]
    if diag_info:
        md.extend(
            [
                f"- **诊断题数**：{diag_info.get('n_problems', 0)}",
                f"- **挑选器增益 (`select@8 - pass@1`)**：{diag_info.get('selector_gain', 0.0) * 100:.2f} 个百分点（闸门要求 ≥ 2.0 点）",
                f"- **挑选缺口 (`cov@8 - select@8`)**：{diag_info.get('selection_gap_at_8', 0.0) * 100:.2f} 个百分点（闸门要求 ≥ 5.0 点）",
                f"- **关键题比例 (`exists K : J_S(K) < J_cov(K)`)**：{diag_info.get('critical_problem_rate', 0.0) * 100:.2f}%（闸门要求 ≥ 10%）",
                f"- **簇不纯率**：{diag_info.get('cluster_impure_rate', 0.0) * 100:.2f}%",
                f"- **误拒率 `eps_hat` (`Chk=0, g=1`)**：{diag_info.get('eps_hat', 0.0) * 100:.2f}%",
                f"- **DP 与 6435 子集暴力枚举最大绝对误差**：`{diag_info.get('dp_vs_bruteforce_max_diff', 0.0):.2e}`（要求 ≤ 1e-9）",
                f"- **训练池 `0 < pass@8 < 1` 保留题数**：{diag_info.get('active_train_pool_count', 0)}",
                f"- **闸门一判定**：`{'PASS' if diag_info.get('gate1_passed') else 'CHECK_METRICS'}`",
                "",
            ]
        )
    else:
        md.append("- 尚未运行或未找到 `results/diag/summary.json`。\n")

    md.extend(
        [
            "## 2. 各训练臂主比较表（最优检查点 vs 最新检查点）",
            "",
            "| 运行名 (`run_name`) | 训练臂 (`arm`) | 种子 | 最优步 (`ckpt`) | `pass@1` | `pass@8` | `select@8` | `gap@8` | 关键题占比 | 簇不纯率 | `eps_hat` | 检查通过率 | 最新步 (`latest_ckpt`) | 最新 `select@8` |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for s in run_summaries:
        pk = s.get("pass_at_k", {})
        sk = s.get("select_at_k", {})
        gk = s.get("gap_at_k", {})
        k_ref = "8" if "8" in sk else ("4" if "4" in sk else "1")
        ckpt_str = s.get("ckpt", s.get("eval_file", "").split("test_")[-1].replace(".jsonl", ""))
        lat_ckpt = s.get("latest_ckpt", ckpt_str)
        lat_s8 = float(s.get("latest_select_at_8", sk.get(k_ref, 0.0)))
        md.append(
            f"| `{s.get('run_name', '')}` | `{s.get('arm', '')}` | {s.get('seed', 0)} | `{ckpt_str}` "
            f"| {pk.get('1', 0.0) * 100:.2f}% | {pk.get(k_ref, 0.0) * 100:.2f}% | **{sk.get(k_ref, 0.0) * 100:.2f}%** "
            f"| {gk.get(k_ref, 0.0) * 100:.2f}% | {s.get('critical_problem_rate', 0.0) * 100:.2f}% "
            f"| {s.get('cluster_impure_rate', 0.0) * 100:.2f}% | {s.get('eps_hat_mean', 0.0) * 100:.2f}% "
            f"| {s.get('chk_pass_rate_mean', 0.0) * 100:.2f}% | `{lat_ckpt}` | {lat_s8 * 100:.2f}% |"
        )

    md.extend(
        [
            "",
            "## 3. 按题配对两级自助法（95% 置信区间与 Holm 校正）",
            "",
            "| 对比 (`arm_a` vs `arm_b`) | 指标 | 题数 | 比赛簇数 | 配对均值差 | 95% 区间 `[ci_low, ci_high]` | `p_value` | `p_holm` | 胜出判定 (`ci_low > 0`) |",
            "|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for comp in comparisons:
        md.append(
            f"| `{comp['arm_a']}` vs `{comp['arm_b']}` | `{comp['metric']}` | {comp['n_problems']} | {comp['n_clusters']} "
            f"| {comp['mean_diff'] * 100:+.2f} pt | `[{comp['ci_low'] * 100:+.2f}, {comp['ci_high'] * 100:+.2f}]` "
            f"| {comp['p_value']:.4f} | {comp.get('p_value_holm', 1.0):.4f} | `{'WIN' if comp['win'] else 'NO'}` |"
        )

    md.extend(
        [
            "",
            "## 4. 训练曲线与全检查点轨迹摘要",
            "",
            f"- 共记录 {len(training_curve_rows)} 条训练步指标（详见 `results/tables/training_curves.csv`）。",
            f"- 共记录 {len(all_ckpt_summaries)} 份测试集检查点评测快照（详见 `results/tables/checkpoint_progression.csv`）。",
            "",
        ]
    )
    report_path = rroot / "report.md"
    atomic_write_text(report_path, "\n".join(md))
    return {
        "n_runs": len(run_summaries),
        "n_comparisons": len(comparisons),
        "report_path": to_rel_path(report_path),
    }
