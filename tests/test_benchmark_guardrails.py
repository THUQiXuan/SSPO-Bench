"""Unit tests for SSPO-Bench anti-hacking guardrails, verifier, and encrypted hidden test vault."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.hidden_test_vault import inject_hidden_tests_in_memory
from benchmark.verifier import (
    run_full_verification,
    verify_algorithm_contract,
    verify_ast_sandbox,
    verify_repo_integrity,
)
from sspo.atomic_io import REPO_ROOT, read_jsonl


def test_default_and_reference_algorithms_pass_verification():
    """Verify default agent_algorithm.py and all 4 reference baseline implementations pass all checks."""
    default_rep = run_full_verification(REPO_ROOT / "benchmark" / "agent_algorithm.py", check_locks=True)
    assert default_rep["passed"], f"Default agent_algorithm failed: {default_rep}"

    ref_dir = REPO_ROOT / "benchmark" / "reference_algorithms"
    for name in ("drgrpo_pass1.py", "pkpo_pass8.py", "checker_reward.py", "tang_maj_loo.py", "sspo.py"):
        rep = run_full_verification(ref_dir / name, check_locks=False)
        assert rep["passed"], f"Reference algorithm {name} failed verification: {rep}"


def test_ast_sandbox_blocks_forbidden_imports_and_builtins(tmp_path: Path):
    """Verify Layer 2 AST sandbox rejects os/sys/open/eval/__globals__."""
    bad_import_code = (
        "import os\n"
        "def compute_group_advantages(ctx): pass\n"
        "def compute_policy_loss(batch): pass\n"
    )
    f1 = tmp_path / "bad_import.py"
    f1.write_text(bad_import_code, encoding="utf-8")
    ok1, errs1 = verify_ast_sandbox(f1)
    assert not ok1 and any("Forbidden import `os`" in e for e in errs1)

    bad_builtin_code = (
        "import numpy as np\n"
        "def compute_group_advantages(ctx):\n"
        "    data = open('data/test.jsonl').read()\n"
        "    return None\n"
        "def compute_policy_loss(batch): pass\n"
    )
    f2 = tmp_path / "bad_builtin.py"
    f2.write_text(bad_builtin_code, encoding="utf-8")
    ok2, errs2 = verify_ast_sandbox(f2)
    assert not ok2 and any("Forbidden identifier `open`" in e for e in errs2)

    bad_dunder_code = (
        "import numpy as np\n"
        "def compute_group_advantages(ctx):\n"
        "    x = compute_group_advantages.__globals__\n"
        "    return None\n"
        "def compute_policy_loss(batch): pass\n"
    )
    f3 = tmp_path / "bad_dunder.py"
    f3.write_text(bad_dunder_code, encoding="utf-8")
    ok3, errs3 = verify_ast_sandbox(f3)
    assert not ok3 and any("Forbidden attribute access `.__globals__`" in e for e in errs3)


def test_contract_blocks_non_equivariant_algorithm(tmp_path: Path):
    """Verify Layer 3 rejects an algorithm whose advantages depend on candidate index rather than candidate content."""
    non_equiv_code = """
import numpy as np
import torch
from benchmark.agent_algorithm import GroupAdvantageOutput

def compute_group_advantages(ctx):
    # Index-dependent hack: always reward index 0 regardless of permutation
    adv = np.zeros(ctx.n, dtype=np.float64)
    if ctx.n > 0:
        adv[0] = 1.0
    return GroupAdvantageOutput(advantages=adv, loss_scale=0.5, drop_problem=False)

def compute_policy_loss(batch):
    return (batch.policy_logprobs * batch.action_mask).sum() * 0.0
"""
    f = tmp_path / "non_equiv.py"
    f.write_text(non_equiv_code, encoding="utf-8")
    ok, errs, _ = verify_algorithm_contract(f)
    assert not ok
    assert any("permutation equivariance" in e for e in errs)


def test_hidden_test_vault_strips_on_disk_and_decrypts_in_memory():
    """Verify data/test.jsonl has no plaintext hidden tests on disk and test_hidden.enc restores them in memory."""
    test_rows = read_jsonl(REPO_ROOT / "data" / "test.jsonl")
    assert len(test_rows) == 100
    for r in test_rows:
        assert r.get("hidden") == [], f"Plaintext hidden tests leaked on disk for {r['pid']}"
        assert r.get("reference_solution") is None, f"Reference solution leaked on disk for {r['pid']}"

    enriched = inject_hidden_tests_in_memory(test_rows)
    assert len(enriched) == 100
    for r in enriched:
        assert len(r.get("hidden") or []) >= 1, f"Vault failed to restore hidden tests for {r['pid']}"
