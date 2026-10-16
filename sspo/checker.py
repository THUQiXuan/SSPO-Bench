"""Execution checker (`Chk(y)` and `sig(y)`), representative program extraction, and persistent hash cache.

Definitions:
- Representative program `y(tau)`: the last program in trajectory `tau` that passes all public examples;
  if none passes all public examples, the last program in `tau`.
- `Chk(y) in {0, 1}`: 1 iff (1) passes all public examples, AND (2) finishes with status `ok` within the time limit
  on every valid/unverified stress (`max`) input, AND (3) finishes with status `ok` and matches `out_key(ref(u))`
  on every confirmed reference input (`a(u) >= 2`).
- `sig(y)`: SHA-1[:16] hash of the space-joined canonical output keys (`out_key`) across all check inputs in fixed order.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sspo.atomic_io import atomic_write_json
from sspo.ec_common import code_hash, last_code_block, out_key, run_prog, sha
from sspo.ec_run import run_candidate_on_cases
from sspo.es_label import label_program_on_hidden


def evaluate_chk_and_sig(exec_rec: Optional[Dict[str, Any]], cases: Optional[Dict[str, Any]]) -> Tuple[int, str]:
    """Compute `(Chk(y), sig(y))` from a candidate execution record and problem `cases` dict."""
    if not exec_rec or not cases:
        return 0, sha("NOCODE")

    # Condition 1: Passes all public examples
    pub_all = bool(exec_rec.get("public_all", True))

    inputs = cases.get("inputs", [])
    valid_max_ids = [
        x["id"]
        for x in inputs
        if x.get("mode") == "max" and x.get("valid") is not False and x.get("inp")
    ]
    valid_small_ids = [
        x["id"]
        for x in inputs
        if x.get("mode") != "max" and x.get("valid") is not False and x.get("inp")
    ]
    confirmed_refs: Dict[str, str] = {}
    for x in inputs:
        if (
            x.get("mode") != "max"
            and x.get("valid") is not False
            and x.get("ref") is not None
            and int(x.get("ref_agree", 0)) >= 2
        ):
            confirmed_refs[x["id"]] = x.get("ref_key") or out_key(x["ref"])

    max_map: Dict[str, Tuple[str, Optional[str]]] = {}
    for item in exec_rec.get("max", []):
        inp_id = item[0]
        status = item[1]
        okey = item[4] if len(item) >= 5 else None
        max_map[inp_id] = (status, okey)

    small_map: Dict[str, Tuple[str, Optional[str]]] = {}
    for item in exec_rec.get("small", []):
        inp_id = item[0]
        status = item[1]
        okey = item[2] if len(item) >= 3 else None
        small_map[inp_id] = (status, okey)

    # Condition 2: Every valid/unverified stress (max) input finishes with status == 'ok'
    max_ok = all(
        inp_id in max_map and max_map[inp_id][0] == "ok"
        for inp_id in valid_max_ids
    )

    # Condition 3: Every confirmed reference input (a(x) >= 2) finishes 'ok' and matches ref out_key
    ref_ok = all(
        inp_id in small_map
        and small_map[inp_id][0] == "ok"
        and small_map[inp_id][1] == ref_key
        for inp_id, ref_key in confirmed_refs.items()
    )

    chk = 1 if (pub_all and max_ok and ref_ok) else 0

    # Signature sig(y): canonical output keys on all check inputs in fixed order (inputs list order)
    sig_parts: List[str] = []
    for x in inputs:
        if x.get("valid") is False or not x.get("inp"):
            continue
        iid = x["id"]
        if x.get("mode") != "max":
            st, ok_hash = small_map.get(iid, ("missing", None))
            sig_parts.append(ok_hash if (st == "ok" and ok_hash) else f"ERR:{st}")
        else:
            st, ok_hash = max_map.get(iid, ("missing", None))
            if ok_hash is not None:
                sig_parts.append(ok_hash if st == "ok" else f"ERR:{st}")
            else:
                sig_parts.append(f"ST:{st}")

    if not sig_parts:
        # Fallback to public example output keys if problem had 0 valid check inputs
        pub_keys = exec_rec.get("public_keys", [])
        sig_parts = [str(k) for k in pub_keys] if pub_keys else ["EMPTY"]

    sig = sha(" ".join(sig_parts))
    return chk, sig


def check_public_examples(code: Optional[str], prob: Dict[str, Any]) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """Run `code` on public examples of `prob`.
    Returns `(passed_all, pass_fraction, details_per_example)`.
    """
    public = prob.get("public", [])
    if not public:
        return bool(code and code.strip()), (1.0 if code and code.strip() else 0.0), []
    if not code or not code.strip():
        return False, 0.0, [{"index": idx, "status": "nocode", "output": ""} for idx in range(len(public))]

    tl = float(prob.get("timeout", 6.0))
    ex_in = [t[0] for t in public]
    ex_out = [t[1] for t in public]
    res = run_prog(code, ex_in, tl, expected=ex_out, keep_out=True)
    details = []
    n_pass = 0
    for idx, r in enumerate(res):
        st, out, wall_s, cpu_s = r[0], r[1], r[2], r[3]
        if st == "pass":
            n_pass += 1
        details.append(
            {
                "index": idx,
                "status": st,
                "output": out[:300] if out else "",
                "expected": ex_out[idx][:300] if ex_out[idx] else "",
                "wall_s": wall_s,
                "cpu_s": cpu_s,
            }
        )
    return n_pass == len(public), n_pass / len(public), details


def format_public_feedback(details: List[Dict[str, Any]]) -> str:
    """Format public example execution results for Turn 2 revision feedback."""
    if not details:
        return "No public examples were available."
    lines = ["Execution results on public examples:"]
    for d in details:
        idx = d["index"] + 1
        st = d["status"]
        if st == "pass":
            lines.append(f"- Example {idx}: PASSED ({d.get('cpu_s', 0.0):.3f}s)")
        elif st == "wrong":
            lines.append(
                f"- Example {idx}: WRONG ANSWER\n"
                f"  Expected: {repr(d.get('expected', ''))}\n"
                f"  Got:      {repr(d.get('output', ''))}"
            )
        else:
            lines.append(f"- Example {idx}: {st.upper()}\n  Detail: {repr(d.get('output', ''))}")
    lines.append(
        "\nPlease fix the bug and provide the corrected complete Python 3 program in a single ```python code block."
    )
    return "\n".join(lines)


def extract_representative_program(
    turns: List[Dict[str, Any]],
    prob: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select the representative program `y(tau)` from a single-turn or multi-turn trajectory.

    Rule:
    `y(tau)` is the LAST program in the trajectory that passes all public examples;
    if no program passes all public examples, `y(tau)` is the LAST program in the trajectory.
    """
    evaluated_turns: List[Dict[str, Any]] = []
    for idx, t in enumerate(turns):
        code = t.get("code")
        if code is None and "text" in t:
            code = last_code_block(t["text"])
        if "public_all" in t and "public_frac" in t:
            pub_all = bool(t["public_all"])
            pub_frac = float(t["public_frac"])
            details = t.get("public_details", [])
        elif prob is not None and code is not None:
            pub_all, pub_frac, details = check_public_examples(code, prob)
        else:
            pub_all = False
            pub_frac = 0.0
            details = []
        evaluated_turns.append(
            {
                "turn_index": idx,
                "code": code,
                "key": code_hash(code),
                "public_all": pub_all,
                "public_frac": pub_frac,
                "public_details": details,
            }
        )

    if not evaluated_turns:
        return {
            "turn_index": -1,
            "code": None,
            "key": "nocode",
            "public_all": False,
            "public_frac": 0.0,
            "public_details": [],
        }

    # Find the last turn with a valid code block that passed all public examples
    passing = [t for t in evaluated_turns if t["code"] is not None and t["public_all"]]
    if passing:
        return passing[-1]

    # Otherwise take the last turn with a code block, or the very last turn
    with_code = [t for t in evaluated_turns if t["code"] is not None]
    if with_code:
        return with_code[-1]
    return evaluated_turns[-1]


class ExecutionCache:
    """Persistent disk cache keyed by `(pid, code_hash, checker_version)` for candidate execution & hidden labels."""

    def __init__(self, cache_root: str | Path, checker_version: str = "v1"):
        self.cache_root = Path(cache_root)
        self.checker_version = checker_version
        self.cases_dir = self.cache_root / "checkers" / "cases"
        self.artifacts_dir = self.cache_root / "checkers" / "artifacts"
        self.exec_dir = self.cache_root / "exec" / checker_version
        self.label_dir = self.cache_root / "labels"
        self.sigma_plus_file = self.cache_root / "sigma_plus.json"
        self.exec_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

        self._cases_mem: Dict[str, dict] = {}
        self._exec_mem: Dict[str, Dict[str, dict]] = {}
        self._label_mem: Dict[str, Dict[str, dict]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def get_cases(self, pid: str, prob: Optional[dict] = None) -> dict:
        if pid in self._cases_mem:
            return self._cases_mem[pid]
        cf = self.cases_dir / f"{pid}.json"
        if cf.exists():
            with open(cf, "r", encoding="utf-8") as f:
                self._cases_mem[pid] = json.load(f)
            return self._cases_mem[pid]
        # If cases file not built yet and problem has inline cases or we build empty fallback
        if prob and "cases" in prob:
            self._cases_mem[pid] = prob["cases"]
            return self._cases_mem[pid]
        return {"pid": pid, "inputs": []}

    def _load_pid_exec(self, pid: str) -> Dict[str, dict]:
        if pid not in self._exec_mem:
            ef = self.exec_dir / f"{pid}.json"
            if ef.exists():
                try:
                    with open(ef, "r", encoding="utf-8") as f:
                        self._exec_mem[pid] = json.load(f)
                except Exception:
                    self._exec_mem[pid] = {}
            else:
                self._exec_mem[pid] = {}
        return self._exec_mem[pid]

    def _load_pid_labels(self, pid: str) -> Dict[str, dict]:
        if pid not in self._label_mem:
            lf = self.label_dir / f"{pid}.json"
            if lf.exists():
                try:
                    with open(lf, "r", encoding="utf-8") as f:
                        self._label_mem[pid] = json.load(f)
                except Exception:
                    self._label_mem[pid] = {}
            else:
                self._label_mem[pid] = {}
        return self._label_mem[pid]

    def evaluate_program(
        self,
        pid: str,
        code: Optional[str],
        prob: dict,
        cases: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Evaluate or retrieve cached `(g(y), Chk(y), sig(y))` for `code` on problem `pid`."""
        key = code_hash(code)
        if key == "nocode" or not code or not code.strip():
            return {
                "key": "nocode",
                "g": 0,
                "chk": 0,
                "sig": sha("NOCODE"),
                "public_all": False,
                "public_frac": 0.0,
                "hidden_frac": 0.0,
                "cached": True,
            }

        c_cases = cases if cases is not None else self.get_cases(pid, prob)
        exec_map = self._load_pid_exec(pid)
        label_map = self._load_pid_labels(pid)

        cached_exec = key in exec_map
        cached_label = key in label_map
        if cached_exec and cached_label:
            self.hits += 1
            exec_rec = exec_map[key]
            label_rec = label_map[key]
        else:
            self.misses += 1
            if not cached_exec:
                exec_rec = run_candidate_on_cases(code, prob, c_cases)
                exec_map[key] = exec_rec
                atomic_write_json(self.exec_dir / f"{pid}.json", exec_map)
            else:
                exec_rec = exec_map[key]

            if not cached_label:
                label_rec = label_program_on_hidden(code, prob)
                label_map[key] = label_rec
                atomic_write_json(self.label_dir / f"{pid}.json", label_map)
            else:
                label_rec = label_map[key]

        chk, sig = evaluate_chk_and_sig(exec_rec, c_cases)
        g = 1 if bool(label_rec.get("all", False)) else 0
        return {
            "key": key,
            "g": g,
            "chk": chk,
            "sig": sig,
            "public_all": bool(exec_rec.get("public_all", False)),
            "public_frac": float(exec_rec.get("public_frac", 0.0)),
            "hidden_frac": float(label_rec.get("frac", 0.0)),
            "cached": cached_exec and cached_label,
        }

    def evaluate_batch_programs(
        self,
        items: List[Tuple[str, Optional[str], dict]],
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of `(pid, code, prob)` items, running uncached distinct `(pid, key)` pairs in parallel."""
        from sspo.ec_common import pmap

        todo_map: Dict[Tuple[str, str], Tuple[str, str, dict, dict, bool, bool]] = {}
        dirty_exec_pids = set()
        dirty_label_pids = set()

        for pid, code, prob in items:
            key = code_hash(code)
            if key == "nocode" or not code or not code.strip():
                continue
            c_cases = self.get_cases(pid, prob)
            exec_map = self._load_pid_exec(pid)
            label_map = self._load_pid_labels(pid)
            need_exec = key not in exec_map
            need_label = key not in label_map
            if (need_exec or need_label) and (pid, key) not in todo_map:
                todo_map[(pid, key)] = (pid, code, prob, c_cases, need_exec, need_label)

        if todo_map:
            todo_list = list(todo_map.items())

            def _run_one(entry):
                (pid, key), (_, code, prob, c_cases, need_exec, need_label) = entry
                er = run_candidate_on_cases(code, prob, c_cases) if need_exec else None
                lr = label_program_on_hidden(code, prob) if need_label else None
                return pid, key, er, lr

            results = pmap(_run_one, todo_list)
            for pid, key, er, lr in results:
                if er is not None:
                    self._load_pid_exec(pid)[key] = er
                    dirty_exec_pids.add(pid)
                if lr is not None:
                    self._load_pid_labels(pid)[key] = lr
                    dirty_label_pids.add(pid)

            for pid in dirty_exec_pids:
                atomic_write_json(self.exec_dir / f"{pid}.json", self._exec_mem[pid])
            for pid in dirty_label_pids:
                atomic_write_json(self.label_dir / f"{pid}.json", self._label_mem[pid])

        out: List[Dict[str, Any]] = []
        for pid, code, prob in items:
            key = code_hash(code)
            was_uncached = (pid, key) in todo_map
            ev = self.evaluate_program(pid, code, prob)
            if was_uncached:
                # Adjust counter since evaluate_program saw it in memory cache after parallel prefill
                self.hits = max(0, self.hits - 1)
                self.misses += 1
                ev["cached"] = False
            out.append(ev)
        return out

