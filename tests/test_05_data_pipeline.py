"""Test 05: Data Pipeline (Filtering, Representative Program Extraction, Hidden-Test Labeling, Schema).

Section 5.5 requirement:
Run filtering, representative program extraction, and hidden-test labeling on 20 problems (from the
built-in sample dataset), and assert the output schema matches `data/schema.json` and `README.md`.
"""
from __future__ import annotations

import json
from pathlib import Path

from sspo.atomic_io import REPO_ROOT, read_jsonl
from sspo.checker import extract_representative_program
from sspo.data_pipeline import prepare_datasets, run_data_pipeline_on_problems


def test_data_pipeline_20_problems_and_schema(tmp_path: Path):
    raw_problems = read_jsonl(REPO_ROOT / "data" / "sample_problems.jsonl")
    assert len(raw_problems) == 25

    res = run_data_pipeline_on_problems(raw_problems)
    stats = res["filter_stats"]
    kept = res["kept_problems"]

    # Exactly 20 unique-answer problems kept and 5 invalid/multi-answer problems removed
    assert stats["total_raw"] == 25
    assert stats["kept"] == 20
    assert stats["removed_total"] == 5
    assert abs(stats["removed_fraction"] - 0.20) < 1e-6
    assert len(kept) == 20

    # Verify schema of every kept problem
    required_keys = {
        "pid",
        "source",
        "split",
        "contest_id",
        "difficulty",
        "timeout",
        "question",
        "public",
        "hidden",
    }
    for prob in kept:
        assert required_keys.issubset(set(prob.keys()))
        assert len(prob["public"]) >= 1
        assert len(prob["hidden"]) >= 5

        pid = prob["pid"]
        rep_list = res["extracted_programs"][pid]
        assert len(rep_list) >= 1
        rep = rep_list[0]
        # Trajectory had [wrong_sol, reference_sol]; representative program must pick the passing reference_sol!
        assert rep["public_all"] is True
        label_key = f"{pid}:{rep['key']}"
        assert res["labels"][label_key]["all"] is True
        assert res["labels"][label_key]["frac"] == 1.0

    # Also test representative program extraction when NO turn passes public examples (falls back to last turn)
    prob0 = kept[0]
    fallback_turns = [
        {"text": "```python\nprint('turn1_wrong')\n```"},
        {"text": "```python\nprint('turn2_wrong')\n```"},
    ]
    rep_fb = extract_representative_program(fallback_turns, prob=prob0)
    assert rep_fb["turn_index"] == 1
    assert "turn2_wrong" in rep_fb["code"]

    # Also test prepare_datasets writing train.jsonl, dev.jsonl, test.jsonl, filter_stats.json
    prep_stats = prepare_datasets(tmp_path, source="sample")
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "dev.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "filter_stats.json").exists()
    assert prep_stats["kept"] == 20
