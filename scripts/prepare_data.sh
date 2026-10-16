#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/prepare_data.sh [--source code_contests|sample] [--include-taco] [--dry-run] [--help]
#
# Description:
#   Downloads and filters `deepmind/code_contests` (and optional `BAAI/TACO`) or uses
#   the built-in offline sample (`--source sample`), filtering to unique-output problems
#   and writing `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`,
#   `data/filter_stats.json`, and `data/done`.
# ==============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE="${SSPO_DATA_SOURCE:-code_contests}"
INCLUDE_TACO=0
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --include-taco)
      INCLUDE_TACO=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,11p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] prepare_data.sh: source=${SOURCE} include_taco=${INCLUDE_TACO}"
  exit 0
fi

mkdir -p data logs
if [[ "${FORCE}" -eq 0 && -f "data/done" && -f "data/train.jsonl" && -f "data/test.jsonl" ]]; then
  echo "[prepare_data] data/done exists. Skipping data preparation."
  exit 0
fi

EXTRA_ARGS=()
if [[ "${INCLUDE_TACO}" -eq 1 ]]; then
  EXTRA_ARGS+=("--include-taco")
fi

"${PYTHON_BIN}" -m sspo.data_pipeline --out-dir data --source "${SOURCE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
"${PYTHON_BIN}" -c "from sspo.atomic_io import write_done_marker; write_done_marker('data/done', {'source': '${SOURCE}'})"
echo "[prepare_data] Completed."
