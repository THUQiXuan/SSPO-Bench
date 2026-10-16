#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/run_diag.sh [--config configs/pilot.yaml] [--mock] [--dry-run] [--help]
#
# Description:
#   Runs Gate 1 Training-Free Diagnostics:
#   - Evaluates selector gain, selection gap@8, critical problem rate, cluster impurity,
#     and `eps_hat` on the training pool.
#   - Verifies exact agreement between Dynamic Programming and 6435-subset brute-force.
#   - Filters active training pool (`0 < pass@8 < 1`) into `data/train_active.jsonl`.
#   - Writes `results/diag/summary.json` and `results/diag/done`.
# ==============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/pilot.yaml"
MOCK_FLAG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --mock)
      MOCK_FLAG="--mock"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,13p' "${BASH_SOURCE[0]}"
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
  echo "[DRY-RUN] run_diag.sh: config=${CONFIG} mock=${MOCK_FLAG}"
  exit 0
fi

mkdir -p results/diag logs
if [[ -f "results/diag/done" ]]; then
  echo "[run_diag] results/diag/done exists. Skipping diagnostics."
  exit 0
fi

CMD=("${PYTHON_BIN}" -m sspo.diag --config "${CONFIG}" --out-dir results/diag)
if [[ -n "${MOCK_FLAG}" ]]; then
  CMD+=("--mock")
fi
"${CMD[@]}" | tee logs/diag.log
