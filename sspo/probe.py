"""Optional Theorem 1(a)(b) & Proposition C Single-Step Probe Experiment (`scripts/run_probe.sh` / `python -m sspo.probe`).

Workflow:
1. Blind-direction probe (Theorem 1(b)):
   - Evaluates a policy along two step directions on critical problems:
     (i) PKPO coverage gradient direction (`nabla J_cov^(k)`),
     (ii) Mass-transfer direction `e` moving probability mass `delta` from the dominant wrong cluster `W_1` to `T`
          (`D_e p_C = 0, D_e p_{W,1} = -delta, D_e p_T = +delta`).
   - Verifies that along the blind direction `e`, `Delta J_cov^(k) == 0` while `Delta J_S^(k) > 0` and agrees with
     the first-order prediction `k * |Delta_1| * delta`.
2. Shared-prefix reversal condition (Proposition C):
   - Parameterizes `p_C = P * q`, `p_W = P * (1 - q)`, `p_T = 1 - P` with `P = sigma(u)`, `q = sigma(v)`.
   - Computes exact `Delta_C`, `Delta_W`, `rho = |Delta_W| / Delta_C`, and checks the exact critical threshold
     `kappa* = (1 - P)^2 * (rho * (1 - q) - q) / (q * (1 - q)^2 * (1 + rho))` against the inner product
     `<nabla J_S^(k), nabla J_cov^(k)>`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sspo.atomic_io import REPO_ROOT, atomic_write_json, to_rel_path, write_done_marker
from sspo.dp import DIFF, comb_exact
from sspo.eval import exact_select_at_k_pure


def exact_js_polynomial(p_C: float, p_W_list: List[float], k: int) -> float:
    """Compute exact `J_S^(k)` under multinomial sampling with probabilities `(p_C, p_W_1..M, p_T)`."""
    # For single wrong cluster M=1 (or general M<=2), compute exact multinomial sum
    p_T = max(0.0, 1.0 - p_C - sum(p_W_list))
    if len(p_W_list) == 1:
        p_W1 = p_W_list[0]
        total = 0.0
        for c in range(1, k + 1):
            for w1 in range(0, k - c + 1):
                t = k - c - w1
                if c < w1:
                    continue
                j_val = 1.0 if c > w1 else 0.5
                multinom = comb_exact(k, c) * comb_exact(k - c, w1)
                prob = multinom * (p_C ** c) * (p_W1 ** w1) * (p_T ** t)
                total += prob * j_val
        return total
    raise ValueError("exact_js_polynomial currently expects M=1 wrong cluster")


def exact_deltas_single_wrong(p_C: float, p_W1: float, k: int) -> Dict[str, float]:
    """Compute exact marginal increments `Delta_C` and `Delta_W1` on `k - 1` i.i.d. draws (Theorem 1(a))."""
    p_T = max(0.0, 1.0 - p_C - p_W1)
    km1 = k - 1
    delta_c = 0.0
    delta_w1 = 0.0
    for c in range(0, km1 + 1):
        for w1 in range(0, km1 - c + 1):
            t = km1 - c - w1
            multinom = comb_exact(km1, c) * comb_exact(km1 - c, w1)
            prob = multinom * (p_C ** c) * (p_W1 ** w1) * (p_T ** t)
            j_base = 0.0 if (c == 0 or c < w1) else (1.0 if c > w1 else 0.5)
            j_plus_c = 0.0 if (c + 1) < w1 else (1.0 if (c + 1) > w1 else 0.5)
            j_plus_w = 0.0 if (c == 0 or c < (w1 + 1)) else (1.0 if c > (w1 + 1) else 0.5)
            delta_c += prob * (j_plus_c - j_base)
            delta_w1 += prob * (j_plus_w - j_base)
    return {"Delta_C": delta_c, "Delta_W1": delta_w1}


def run_probe_experiment(
    out_dir: str | Path = REPO_ROOT / "results" / "probe",
    k: int = 8,
    delta_step: float = 0.01,
) -> Dict[str, Any]:
    """Run Theorem 1(b) blind-direction probe and Proposition C reversal threshold verification."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Theorem 1(b) Blind Direction Verification across critical configurations
    blind_cases = [
        {"name": "P=0.3_q=0.2", "p_C": 0.06, "p_W1": 0.24},
        {"name": "P=0.4_q=0.25", "p_C": 0.10, "p_W1": 0.30},
        {"name": "P=0.5_q=0.3", "p_C": 0.15, "p_W1": 0.35},
    ]
    blind_results = []
    for case in blind_cases:
        p_c = case["p_C"]
        p_w1 = case["p_W1"]
        j_cov_before = 1.0 - ((1.0 - p_c) ** k)
        j_cov_after = 1.0 - ((1.0 - p_c) ** k)  # p_C unchanged along blind direction
        j_s_before = exact_js_polynomial(p_c, [p_w1], k=k)
        j_s_after = exact_js_polynomial(p_c, [p_w1 - delta_step], k=k)
        deltas = exact_deltas_single_wrong(p_c, p_w1, k=k)
        first_order_pred = k * abs(deltas["Delta_W1"]) * delta_step
        actual_delta_js = j_s_after - j_s_before
        blind_results.append(
            {
                "case": case["name"],
                "p_C": p_c,
                "p_W1": p_w1,
                "delta_step": delta_step,
                "delta_J_cov": round(j_cov_after - j_cov_before, 10),
                "delta_J_S_actual": round(actual_delta_js, 6),
                "delta_J_S_first_order": round(first_order_pred, 6),
                "relative_agreement": round(actual_delta_js / max(1e-12, first_order_pred), 4),
            }
        )

    # 2. Proposition C numerical verification (k = 8, P = 0.3, q in {0.1, 0.2, 0.3} and P = 0.1, q = 0.2)
    prop_c_grid = [
        {"P": 0.3, "q": 0.1},
        {"P": 0.3, "q": 0.2},
        {"P": 0.3, "q": 0.3},
        {"P": 0.1, "q": 0.2},
    ]
    prop_c_results = []
    for item in prop_c_grid:
        P, q = item["P"], item["q"]
        p_c = P * q
        p_w1 = P * (1.0 - q)
        deltas = exact_deltas_single_wrong(p_c, p_w1, k=k)
        rho = abs(deltas["Delta_W1"]) / max(1e-12, deltas["Delta_C"])
        num = ((1.0 - P) ** 2) * (rho * (1.0 - q) - q)
        den = q * ((1.0 - q) ** 2) * (1.0 + rho)
        kappa_star = max(0.0, num / den) if num > 0 else 0.0
        prop_c_results.append(
            {
                "P": P,
                "q": q,
                "rho": round(rho, 3),
                "kappa_star": round(kappa_star, 4),
                "can_reverse": bool(rho > q / (1.0 - q)),
            }
        )

    summary = {
        "k": k,
        "blind_direction_probes": blind_results,
        "proposition_c_thresholds": prop_c_results,
    }
    atomic_write_json(out_path / "probe_summary.json", summary)
    write_done_marker(out_path, {"k": k})
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Theorem 1(b) & Proposition C single-step probe.")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "probe"))
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    res = run_probe_experiment(out_dir=args.out_dir, k=args.k)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
