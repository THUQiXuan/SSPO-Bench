"""Test 04: Checker Pipeline (`ec_gen` -> `ec_build` -> `ec_run` -> `Chk(y)` & `sig(y)`).

Section 5.4 requirement:
Use 3 hand-written small problems (with public examples and hand-written generators/brute-forces/validators
replacing model output, mocking vLLM calls) to run `build -> run -> Chk/sig`, and assert the 3 conditions
of `Chk` as well as `sig` normalization behavior (whitespace, 6-significant-digit floats, input order).
"""
from __future__ import annotations

from pathlib import Path

from sspo.checker import evaluate_chk_and_sig
from sspo.ec_build import build_all
from sspo.ec_common import out_key
from sspo.ec_gen import generate_helpers
from sspo.ec_run import run_candidate_on_cases


def _make_three_handwritten_problems() -> dict:
    gen_code = (
        "import sys, random\n"
        "mode, seed = sys.stdin.read().split()\n"
        "seed = int(seed)\n"
        "random.seed(seed)\n"
        "if mode == 'max':\n"
        "    print(f'{100 + seed} {200 + seed}')\n"
        "elif mode == 'edge':\n"
        "    print(f'{seed} {seed}')\n"
        "else:\n"
        "    print(f'{seed + 2} {seed + 5}')\n"
    )
    good_val = (
        "import sys\n"
        "toks = sys.stdin.read().split()\n"
        "print('VALID' if len(toks) == 2 and all(t.lstrip('-').isdigit() for t in toks) else 'INVALID')\n"
    )
    bad_val_rejects_examples = "import sys\nprint('INVALID')\n"

    probs = {}
    specs = [
        ("hw1_add", "a + b", "a, b = map(int, input().split())\nprint(a + b)\n", [["2 3\n", "5\n"], ["10 20\n", "30\n"]]),
        ("hw2_mul", "a * b", "a, b = map(int, input().split())\nprint(a * b)\n", [["2 3\n", "6\n"], ["4 5\n", "20\n"]]),
        (
            "hw3_float",
            "(a + b) / 3.0",
            "a, b = map(int, input().split())\nprint(f'{(a + b) / 3.0:.8f}')\n",
            [["1 2\n", "1.000000\n"], ["2 2\n", "1.333333\n"]],
        ),
    ]
    for pid, desc, bf_good, pub in specs:
        bf_bad_example = "print(-999999)\n"
        probs[pid] = {
            "pid": pid,
            "question": f"Compute {desc}.",
            "public": pub,
            "timeout": 2.0,
            "mock_helpers": {
                "gen": [gen_code] * 4,
                # 3 good brute-forces + 1 bad brute-force (filtered out by public examples -> a(x) = 3 >= 2)
                "bf": [bf_good, bf_good, bf_good, bf_bad_example],
                # 1 good validator + 1 bad validator (bad validator rejected on examples)
                "val": [good_val, bad_val_rejects_examples],
            },
        }
    return probs


def test_checker_pipeline_three_problems_and_chk_sig_semantics(tmp_path: Path):
    probs = _make_three_handwritten_problems()
    art_dir = tmp_path / "artifacts"
    cases_dir = tmp_path / "cases"

    # 1. Generate helpers (mocking vLLM) and build cases
    generate_helpers(probs, str(art_dir), mock_fn=lambda p, kind, n: [f"```python\n{c}\n```" for c in p["mock_helpers"][kind][:n]])
    built = build_all(probs, str(art_dir), str(cases_dir))
    assert set(built.keys()) == {"hw1_add", "hw2_mul", "hw3_float"}

    for pid, c_data in built.items():
        # Bad validator & bad brute-force were filtered out on public examples
        assert c_data["n_val_kept"] == 1
        assert c_data["n_bf_kept"] == 3
        confirmed = [x for x in c_data["inputs"] if x.get("ref") is not None and x.get("ref_agree", 0) >= 2]
        assert len(confirmed) >= 2

    # 2. Test the 3 conditions of Chk(y) on hw1_add
    p1 = probs["hw1_add"]
    c1 = built["hw1_add"]

    # (a) Correct program: passes public examples, finishes max inputs, matches all confirmed references -> Chk = 1
    code_correct = "a, b = map(int, input().split())\nprint(a + b)\n"
    rec_correct = run_candidate_on_cases(code_correct, p1, c1)
    chk_c, sig_c = evaluate_chk_and_sig(rec_correct, c1)
    assert chk_c == 1

    # (b) Fails public example condition -> Chk = 0
    code_fail_pub = "a, b = map(int, input().split())\nprint(0 if (a == 2 and b == 3) else a + b)\n"
    rec_fail_pub = run_candidate_on_cases(code_fail_pub, p1, c1)
    chk_fp, _ = evaluate_chk_and_sig(rec_fail_pub, c1)
    assert rec_fail_pub["public_all"] is False
    assert chk_fp == 0

    # (c) Passes public examples & small references, but crashes/times out on max inputs (a >= 100) -> Chk = 0
    code_fail_max = (
        "import sys\n"
        "a, b = map(int, input().split())\n"
        "if a >= 100:\n"
        "    sys.exit(1)\n"
        "print(a + b)\n"
    )
    rec_fail_max = run_candidate_on_cases(code_fail_max, p1, c1)
    chk_fm, _ = evaluate_chk_and_sig(rec_fail_max, c1)
    assert rec_fail_max["public_all"] is True
    assert chk_fm == 0

    # (d) Passes public examples & finishes max inputs, but disagrees with confirmed reference on small inputs -> Chk = 0
    code_fail_ref = (
        "a, b = map(int, input().split())\n"
        "if (a, b) in ((2, 3), (10, 20)) or a >= 100:\n"
        "    print(a + b)\n"
        "else:\n"
        "    print(a + b + 1)\n"
    )
    rec_fail_ref = run_candidate_on_cases(code_fail_ref, p1, c1)
    chk_fr, _ = evaluate_chk_and_sig(rec_fail_ref, c1)
    assert rec_fail_ref["public_all"] is True
    assert chk_fr == 0

    # 3. Test sig(y) normalization: whitespace, 6-significant-digit floats, and input order sensitivity
    p3 = probs["hw3_float"]
    c3 = built["hw3_float"]
    code_f1 = "a, b = map(int, input().split())\nprint(f'  {(a + b) / 3.0:.9f}   \\n')\n"
    code_f2 = "a, b = map(int, input().split())\nprint(f'{(a + b) / 3.0 + 1e-9:.7f}')\n"
    code_f3_diff = "a, b = map(int, input().split())\nprint(f'{(a + b) / 3.0 + 1e-2:.6f}')\n"

    rec_f1 = run_candidate_on_cases(code_f1, p3, c3)
    rec_f2 = run_candidate_on_cases(code_f2, p3, c3)
    rec_f3 = run_candidate_on_cases(code_f3_diff, p3, c3)

    chk_f1, sig_f1 = evaluate_chk_and_sig(rec_f1, c3)
    chk_f2, sig_f2 = evaluate_chk_and_sig(rec_f2, c3)
    _, sig_f3 = evaluate_chk_and_sig(rec_f3, c3)

    assert chk_f1 == 1 and chk_f2 == 1
    assert sig_f1 == sig_f2, "Whitespace and <1e-7 float differences must yield identical sig(y)"
    assert sig_f1 != sig_f3, "Outputs differing at 2nd decimal place must yield distinct sig(y)"

    # Verify order sensitivity of sig(y): reversing the order of small outputs changes sig(y)
    rec_swapped = dict(rec_f1)
    rec_swapped["small"] = [
        [orig[0], rev[1], rev[2]]
        for orig, rev in zip(rec_f1["small"], reversed(rec_f1["small"]))
    ]
    _, sig_swapped = evaluate_chk_and_sig(rec_swapped, c3)
    assert sig_swapped != sig_f1, "Swapping outputs across check inputs must change sig(y)"

    # Direct unit checks on out_key
    assert out_key("  1.234561   \n\t 42 ") == out_key("1.234562 42")
    assert out_key("1.23456 42") != out_key("42 1.23456")
