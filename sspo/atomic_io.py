"""Atomic file I/O, resume-safe JSONL writing, done markers, and manifest utilities.

Guarantees:
1. Every file write goes to `<target>.tmp.<pid>.<tid>` first and uses `os.replace` for atomic rename.
2. `append_metric_step_atomic` deduplicates steps >= `step` before writing, preventing duplicate
   or corrupted rows when a killed run is resumed from an earlier checkpoint.
3. All paths stored inside manifests/results are converted to repository-relative paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def to_rel_path(path: str | Path, root: Optional[Path] = None) -> str:
    """Return a repository-relative POSIX path string, stripping any absolute prefix."""
    base = (root or REPO_ROOT).resolve()
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(base).as_posix()
    except ValueError:
        # If outside repo (e.g., temporary test dir), return relative to its parent or basename
        return p.name


def sanitize_paths_in_obj(obj: Any, root: Optional[Path] = None) -> Any:
    """Recursively convert any absolute paths under `root` to relative paths in dicts/lists."""
    base = str((root or REPO_ROOT).resolve())
    if isinstance(obj, Path):
        return to_rel_path(obj, root)
    if isinstance(obj, str):
        if obj.startswith(base):
            rel = obj[len(base) :].lstrip("/")
            return rel or "."
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_paths_in_obj(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_paths_in_obj(v, root) for v in obj]
    return obj


def _tmp_path(target: Path) -> Path:
    pid = os.getpid()
    tid = threading.get_ident()
    return target.parent / f".{target.name}.tmp.{pid}.{tid}"


def atomic_write_bytes(target: str | Path, data: bytes) -> None:
    """Write bytes to a temporary file in the same directory and atomically rename."""
    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_text(target: str | Path, text: str) -> None:
    """Atomically write UTF-8 text to `target`."""
    atomic_write_bytes(target, text.encode("utf-8"))


def atomic_write_json(target: str | Path, obj: Any, indent: Optional[int] = 2) -> None:
    """Atomically write JSON to `target`."""
    payload = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False) + "\n"
    atomic_write_text(target, payload)


def atomic_write_jsonl(target: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Atomically write an iterable of JSON objects as JSONL to `target`."""
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    text = ("\n".join(lines) + "\n") if lines else ""
    atomic_write_text(target, text)


def read_jsonl(target: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, ignoring blank or truncated trailing lines safely."""
    p = Path(target)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_metric_step_atomic(target: str | Path, step: int, row: Dict[str, Any]) -> None:
    """Append or overwrite metric entry for `step` atomically, truncating any stale steps >= `step`."""
    existing = [r for r in read_jsonl(target) if int(r.get("step", -1)) < step]
    row_with_step = {"step": step, **row}
    existing.append(row_with_step)
    existing.sort(key=lambda r: int(r.get("step", 0)))
    atomic_write_jsonl(target, existing)


def done_marker_path(stage_dir_or_file: str | Path) -> Path:
    p = Path(stage_dir_or_file)
    if p.name == "done" or p.suffix == ".done":
        return p
    return p / "done"


def has_done_marker(stage_dir_or_file: str | Path) -> bool:
    return done_marker_path(stage_dir_or_file).exists()


def write_done_marker(stage_dir_or_file: str | Path, info: Optional[Dict[str, Any]] = None) -> Path:
    marker = done_marker_path(stage_dir_or_file)
    payload = {
        "status": "done",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(info or {}),
    }
    atomic_write_json(marker, payload)
    return marker


def get_git_commit(root: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root or REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        ).strip()
        return out or "uncommitted"
    except Exception:
        return "uncommitted"


def compute_model_hash(model_name_or_path: str) -> str:
    """Compute a deterministic hash identifier for the model configuration or weight directory."""
    p = Path(model_name_or_path)
    h = hashlib.sha256()
    h.update(str(model_name_or_path).encode("utf-8"))
    if p.exists() and p.is_dir():
        cfg = p / "config.json"
        if cfg.exists():
            h.update(cfg.read_bytes())
    return h.hexdigest()[:16]


def write_manifest(
    run_dir: str | Path,
    config: Dict[str, Any],
    arm: str,
    seed: int,
    num_gpus: int,
    gpu_mem_gb: int,
    preset: Dict[str, Any],
    model_name: str = "Qwen/Qwen3-8B",
    root: Optional[Path] = None,
) -> Path:
    """Create or update `manifest.json` in `run_dir` using repository-relative paths."""
    rdir = Path(run_dir)
    manifest_file = rdir / "manifest.json"
    existing: Dict[str, Any] = {}
    if manifest_file.exists():
        try:
            existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    start_time = existing.get(
        "start_time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    resumed_count = int(existing.get("resumed_count", -1)) + 1

    manifest = {
        "run_dir": to_rel_path(rdir, root),
        "arm": arm,
        "seed": seed,
        "start_time": start_time,
        "last_resumed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resumed_count": resumed_count,
        "git_commit": get_git_commit(root),
        "model_name": model_name,
        "model_hash": compute_model_hash(model_name),
        "num_gpus": int(num_gpus),
        "gpu_mem_gb": int(gpu_mem_gb),
        "preset": sanitize_paths_in_obj(preset, root),
        "config": sanitize_paths_in_obj(config, root),
    }
    atomic_write_json(manifest_file, manifest)
    return manifest_file
