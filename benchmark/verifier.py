"""Three-Layer Anti-Hacking & Pre-Flight Verification Engine (`benchmark/verifier.py`).

Enforces strict benchmark integrity before and after agent RL training:
1. **Layer 1 — Cryptographic File Lock (`verify_repo_integrity`)**:
   Verifies SHA-256 checksums of all locked framework, config, and dataset files
   against `benchmark/manifest_locks.json`. Only `benchmark/agent_algorithm.py`
   may be edited by the agent.
2. **Layer 2 — Static AST & Import Sandbox (`verify_ast_sandbox`)**:
   Parses `benchmark/agent_algorithm.py` via Python's `ast` module and blocks
   filesystem, network, process, and introspection calls (`os`, `sys`, `open`,
   `eval`, `exec`, `subprocess`, `__globals__`, etc.).
3. **Layer 3 — Mathematical & Functional Contract (`verify_algorithm_contract`)**:
   Tests `compute_group_advantages` and `compute_policy_loss` on synthetic rollout
   groups for shape/finiteness, permutation equivariance, differentiability,
   gradient flow, and per-problem CPU latency (< 50 ms).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sspo.atomic_io import REPO_ROOT, atomic_write_json

LOCKED_GLOBS = [
    "sspo/*.py",
    "configs/benchmark_locked.yaml",
    "configs/pilot.yaml",
    "configs/smoke.yaml",
    "data/train.jsonl",
    "data/dev.jsonl",
    "data/test.jsonl",
    "data/test_hidden.enc",
    "data/sample_problems.jsonl",
    "benchmark/hidden_test_vault.py",
    "benchmark/reference_algorithms/*.py",
]

ALLOWED_IMPORT_PREFIXES = (
    "__future__",
    "math",
    "itertools",
    "collections",
    "functools",
    "typing",
    "dataclasses",
    "random",
    "numpy",
    "torch",
    "sspo.dp",
    "sspo.selector",
    "benchmark.agent_algorithm",
)

FORBIDDEN_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "exit",
    "quit",
}

FORBIDDEN_ATTRIBUTES = {
    "__subclasses__",
    "__globals__",
    "__code__",
    "__closure__",
    "__builtins__",
    "__dict__",
    "__bases__",
    "__mro__",
    "load_state_dict",
    "save_pretrained",
    "from_pretrained",
    "system",
    "popen",
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest_locks(
    repo_root: Path = REPO_ROOT,
    out_file: Path = REPO_ROOT / "benchmark" / "manifest_locks.json",
) -> Dict[str, str]:
    """Compute and write SHA-256 hashes for all locked benchmark files."""
    locks: Dict[str, str] = {}
    for pattern in LOCKED_GLOBS:
        for p in sorted(repo_root.glob(pattern)):
            if p.is_file() and not p.name.startswith("."):
                rel = p.relative_to(repo_root).as_posix()
                locks[rel] = compute_file_sha256(p)
    atomic_write_json(out_file, {"version": "1.0.0", "locked_files": locks})
    return locks


def verify_repo_integrity(
    repo_root: Path = REPO_ROOT,
    manifest_file: Path = REPO_ROOT / "benchmark" / "manifest_locks.json",
) -> Tuple[bool, List[str]]:
    """Verify that all locked files match their recorded SHA-256 hashes."""
    errors: List[str] = []
    if not manifest_file.exists():
        return False, [f"Missing lock manifest: {manifest_file}"]

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    locked_files: Dict[str, str] = data.get("locked_files", {})
    for rel_path, expected_hash in sorted(locked_files.items()):
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            errors.append(f"TAMPER_DETECTED: Locked file missing: {rel_path}")
            continue
        actual_hash = compute_file_sha256(abs_path)
        if actual_hash != expected_hash:
            errors.append(
                f"TAMPER_DETECTED: Locked file modified: {rel_path} "
                f"(expected {expected_hash[:12]}..., got {actual_hash[:12]}...)"
            )
    return len(errors) == 0, errors


def verify_ast_sandbox(algo_file: Path = REPO_ROOT / "benchmark" / "agent_algorithm.py") -> Tuple[bool, List[str]]:
    """Statically inspect `agent_algorithm.py` AST for forbidden imports, builtins, or attributes."""
    errors: List[str] = []
    if not algo_file.exists():
        return False, [f"Missing algorithm file: {algo_file}"]

    try:
        source = algo_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(algo_file))
    except SyntaxError as e:
        return False, [f"SyntaxError in {algo_file.name}: {e}"]

    top_level_funcs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for req_fn in ("compute_group_advantages", "compute_policy_loss"):
        if req_fn not in top_level_funcs:
            errors.append(f"Missing required top-level function `{req_fn}` in {algo_file.name}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if not any(mod == p or mod.startswith(p + ".") for p in ALLOWED_IMPORT_PREFIXES):
                    errors.append(f"Forbidden import `{mod}` at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not any(mod == p or mod.startswith(p + ".") for p in ALLOWED_IMPORT_PREFIXES):
                errors.append(f"Forbidden from-import `{mod}` at line {node.lineno}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALL_NAMES:
                errors.append(f"Forbidden identifier `{node.id}` at line {node.lineno}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                errors.append(f"Forbidden attribute access `.{node.attr}` at line {node.lineno}")

    return len(errors) == 0, errors


def _load_module_from_path(algo_file: Path):
    spec = importlib.util.spec_from_file_location("_agent_algo_ under_test", str(algo_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {algo_file}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_algorithm_contract(
    algo_file: Path = REPO_ROOT / "benchmark" / "agent_algorithm.py",
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Run functional, mathematical, permutation-equivariance, and gradient checks on `algo_file`."""
    errors: List[str] = []
    stats: Dict[str, Any] = {}

    try:
        mod = _load_module_from_path(algo_file)
    except Exception as e:
        return False, [f"Failed to import {algo_file.name}: {e}"], stats

    from benchmark.agent_algorithm import GroupContext, PolicyLossBatch

    compute_group_advantages = getattr(mod, "compute_group_advantages", None)
    compute_policy_loss = getattr(mod, "compute_policy_loss", None)
    if compute_group_advantages is None or compute_policy_loss is None:
        return False, ["Missing compute_group_advantages or compute_policy_loss"], stats

    # Synthetic test pools
    test_pools = [
        (
            "mixed_pool",
            GroupContext(
                pid="prob_mixed",
                step=5,
                total_steps=30,
                n=16,
                k=8,
                g_list=(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                chk_list=(1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                sig_list=(
                    "sig_ok",
                    "sig_ok",
                    "sig_fail",
                    "sig_w1",
                    "sig_w1",
                    "sig_w1",
                    "sig_w1",
                    "sig_w2",
                    "sig_w2",
                    "TLE",
                    "RE",
                    "sig_t1",
                    "sig_t2",
                    "sig_t3",
                    "sig_t4",
                    "sig_t5",
                ),
                response_lengths=tuple(100 + 10 * i for i in range(16)),
                sequence_logprobs=tuple(-15.0 - 0.5 * i for i in range(16)),
                ref_sequence_logprobs=tuple(-15.2 - 0.5 * i for i in range(16)),
                turn_count=1,
                history_state={"sigma_plus": []},
            ),
        ),
        (
            "all_zero_pool",
            GroupContext(
                pid="prob_zero",
                step=1,
                total_steps=30,
                n=16,
                k=8,
                g_list=tuple([0] * 16),
                chk_list=tuple([0] * 16),
                sig_list=tuple(["TLE"] * 16),
                response_lengths=tuple([200] * 16),
                sequence_logprobs=tuple([-20.0] * 16),
                ref_sequence_logprobs=tuple([-20.0] * 16),
            ),
        ),
        (
            "all_correct_pool",
            GroupContext(
                pid="prob_all_c",
                step=10,
                total_steps=30,
                n=16,
                k=8,
                g_list=tuple([1] * 16),
                chk_list=tuple([1] * 16),
                sig_list=tuple(["sig_ok"] * 16),
                response_lengths=tuple([120] * 16),
                sequence_logprobs=tuple([-12.0] * 16),
                ref_sequence_logprobs=tuple([-12.5] * 16),
            ),
        ),
        (
            "collusion_trap_pool",
            GroupContext(
                pid="prob_trap",
                step=15,
                total_steps=30,
                n=16,
                k=8,
                g_list=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                chk_list=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                sig_list=(
                    "sig_c",
                    "sig_w_dom",
                    "sig_w_dom",
                    "sig_w_dom",
                    "sig_w_dom",
                    "sig_w_dom",
                    "sig_w_dom",
                    "TLE",
                    "TLE",
                    "RE",
                    "RE",
                    "s1",
                    "s2",
                    "s3",
                    "s4",
                    "s5",
                ),
                response_lengths=tuple(150 + 5 * i for i in range(16)),
                sequence_logprobs=tuple(-18.0 - 0.2 * i for i in range(16)),
                ref_sequence_logprobs=tuple(-18.1 - 0.2 * i for i in range(16)),
            ),
        ),
        (
            "impure_cluster_pool",
            GroupContext(
                pid="prob_impure",
                step=20,
                total_steps=30,
                n=16,
                k=8,
                g_list=(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                chk_list=(1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                sig_list=(
                    "sig_shared",
                    "sig_shared",
                    "sig_shared",
                    "sig_w",
                    "sig_w",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                    "TLE",
                ),
                response_lengths=tuple([180] * 16),
                sequence_logprobs=tuple([-14.0] * 16),
                ref_sequence_logprobs=tuple([-14.0] * 16),
            ),
        ),
    ]

    for pool_name, ctx in test_pools:
        try:
            out = compute_group_advantages(ctx)
        except Exception as e:
            errors.append(f"`compute_group_advantages` raised exception on `{pool_name}`: {e}")
            continue

        adv = np.asarray(out.advantages, dtype=np.float64)
        if adv.shape != (ctx.n,):
            errors.append(f"`{pool_name}`: expected advantages shape ({ctx.n},), got {adv.shape}")
            continue
        if not np.all(np.isfinite(adv)):
            errors.append(f"`{pool_name}`: advantages contain NaN or Inf")
        if np.max(np.abs(adv)) > 50.0 + 1e-6:
            errors.append(f"`{pool_name}`: max |advantage| = {np.max(np.abs(adv)):.3f} exceeds bound 50.0")
        if not (0.0 <= float(out.loss_scale) <= 2.0):
            errors.append(f"`{pool_name}`: loss_scale = {out.loss_scale} outside [0.0, 2.0]")

        # Permutation Equivariance Check on mixed_pool and collusion_trap_pool
        if pool_name in ("mixed_pool", "collusion_trap_pool"):
            rng = np.random.default_rng(42)
            for p_trial in range(3):
                perm = rng.permutation(ctx.n)
                ctx_perm = GroupContext(
                    pid=ctx.pid,
                    step=ctx.step,
                    total_steps=ctx.total_steps,
                    n=ctx.n,
                    k=ctx.k,
                    g_list=tuple(ctx.g_list[i] for i in perm),
                    chk_list=tuple(ctx.chk_list[i] for i in perm),
                    sig_list=tuple(ctx.sig_list[i] for i in perm),
                    response_lengths=tuple(ctx.response_lengths[i] for i in perm),
                    sequence_logprobs=tuple(ctx.sequence_logprobs[i] for i in perm),
                    ref_sequence_logprobs=tuple(ctx.ref_sequence_logprobs[i] for i in perm),
                    turn_count=ctx.turn_count,
                    history_state=dict(ctx.history_state),
                )
                out_perm = compute_group_advantages(ctx_perm)
                adv_perm = np.asarray(out_perm.advantages, dtype=np.float64)
                if not np.allclose(adv_perm, adv[perm], atol=1e-5, rtol=1e-4):
                    errors.append(
                        f"`{pool_name}` (trial {p_trial}): failed permutation equivariance check "
                        f"(max diff = {np.max(np.abs(adv_perm - adv[perm])):.2e})"
                    )
                    break

    # Latency benchmark on mixed_pool
    if not errors:
        t0 = time.perf_counter()
        n_iters = 20
        for _ in range(n_iters):
            compute_group_advantages(test_pools[0][1])
        avg_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
        stats["avg_group_advantage_ms"] = round(avg_ms, 3)
        if avg_ms > 50.0:
            errors.append(f"`compute_group_advantages` latency {avg_ms:.2f} ms exceeds 50.0 ms limit")

    # Differentiability & gradient flow check on compute_policy_loss
    try:
        torch.manual_seed(7)
        B, T = 8, 32
        policy_lp = (-1.5 + 0.2 * torch.randn(B, T, dtype=torch.float32)).requires_grad_(True)
        old_lp = policy_lp.detach() + 0.05 * torch.randn(B, T, dtype=torch.float32)
        ref_lp = policy_lp.detach() - 0.05 * torch.randn(B, T, dtype=torch.float32)
        mask = torch.ones(B, T, dtype=torch.float32)
        mask[:, :6] = 0.0
        adv_t = torch.tensor([0.8, 0.5, -0.4, -0.3, 0.2, -0.2, -0.3, -0.3], dtype=torch.float32)
        scales_t = torch.full((B,), 0.5, dtype=torch.float32)

        batch = PolicyLossBatch(
            policy_logprobs=policy_lp,
            old_logprobs=old_lp,
            ref_logprobs=ref_lp,
            action_mask=mask,
            advantages=adv_t,
            loss_scales=scales_t,
            num_problems_in_batch=2,
            l_max=64,
            clip_eps=0.2,
            kl_coef=0.001,
            step=5,
            total_steps=30,
        )
        res = compute_policy_loss(batch)
        loss_tensor = res[0] if isinstance(res, tuple) else res
        if not isinstance(loss_tensor, torch.Tensor) or loss_tensor.ndim != 0:
            errors.append("`compute_policy_loss` must return a scalar torch.Tensor")
        elif not loss_tensor.requires_grad:
            errors.append("`compute_policy_loss` returned tensor with requires_grad=False")
        elif not torch.isfinite(loss_tensor).item():
            errors.append("`compute_policy_loss` returned non-finite loss")
        else:
            loss_tensor.backward()
            if policy_lp.grad is None:
                errors.append("`compute_policy_loss` produced None gradient on policy_logprobs")
            elif not torch.all(torch.isfinite(policy_lp.grad)).item():
                errors.append("`compute_policy_loss` produced NaN/Inf gradients on policy_logprobs")
            elif float(policy_lp.grad.abs().sum().item()) < 1e-12:
                errors.append("`compute_policy_loss` produced identically zero gradient on active tokens")
            else:
                stats["policy_loss_val"] = round(float(loss_tensor.detach().item()), 6)
                stats["grad_l1_norm"] = round(float(policy_lp.grad.abs().sum().item()), 6)
    except Exception as e:
        errors.append(f"`compute_policy_loss` failed differentiability check: {e}")

    return len(errors) == 0, errors, stats


def run_full_verification(
    algo_file: Path = REPO_ROOT / "benchmark" / "agent_algorithm.py",
    check_locks: bool = True,
) -> Dict[str, Any]:
    """Run all 3 verification layers and return a structured verification report."""
    lock_ok, lock_errors = (True, [])
    if check_locks:
        lock_ok, lock_errors = verify_repo_integrity()

    ast_ok, ast_errors = verify_ast_sandbox(algo_file)
    contract_ok, contract_errors, contract_stats = (False, [], {})
    if ast_ok:
        contract_ok, contract_errors, contract_stats = verify_algorithm_contract(algo_file)

    all_passed = bool(lock_ok and ast_ok and contract_ok)
    return {
        "passed": all_passed,
        "algorithm_file": str(algo_file.relative_to(REPO_ROOT)) if algo_file.is_relative_to(REPO_ROOT) else str(algo_file),
        "algorithm_sha256": compute_file_sha256(algo_file) if algo_file.exists() else None,
        "layer1_repo_integrity": {"passed": lock_ok, "errors": lock_errors},
        "layer2_ast_sandbox": {"passed": ast_ok, "errors": ast_errors},
        "layer3_math_contract": {"passed": contract_ok, "errors": contract_errors, "stats": contract_stats},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify agent RL algorithm and repository locks.")
    parser.add_argument(
        "--algo-file",
        type=str,
        default=str(REPO_ROOT / "benchmark" / "agent_algorithm.py"),
        help="Path to the agent algorithm file to verify",
    )
    parser.add_argument("--init-locks", action="store_true", help="Regenerate benchmark/manifest_locks.json")
    parser.add_argument("--skip-locks", action="store_true", help="Skip Layer 1 repo lock verification")
    args = parser.parse_args()

    if args.init_locks:
        locks = generate_manifest_locks()
        print(json.dumps({"status": "generated_locks", "n_locked_files": len(locks)}, indent=2))
        return

    report = run_full_verification(Path(args.algo_file), check_locks=not args.skip_locks)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
