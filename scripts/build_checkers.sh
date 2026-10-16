#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/build_checkers.sh [--config configs/pilot.yaml] [--mock] [--dry-run] [--help]
#
# Description:
#   Generates frozen execution checker helper programs (`ec_gen`: 4 generators,
#   4 brute-force solutions, 2 validators per problem, thinking off) and builds
#   verified test cases & confirmed reference outputs (`ec_build`, `a(x) >= 2`).
#   Caches everything under `cache/checkers/` and writes `cache/checkers/done`.
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
  echo "[DRY-RUN] build_checkers.sh: config=${CONFIG} mock=${MOCK_FLAG}"
  exit 0
fi

mkdir -p cache/checkers/artifacts cache/checkers/cases logs
if [[ -f "cache/checkers/done" ]]; then
  echo "[build_checkers] cache/checkers/done exists. Skipping checker construction."
  exit 0
fi

MODE="$("${PYTHON_BIN}" -c "from sspo.config import load_config; print(load_config('${CONFIG}').get('mode', 'pilot'))")"
if [[ "${MODE}" == "smoke" || "${SSPO_MOCK_LLM:-0}" == "1" ]]; then
  MOCK_FLAG="--mock"
fi

for SPLIT_FILE in data/train.jsonl data/dev.jsonl data/test.jsonl; do
  if [[ -f "${SPLIT_FILE}" ]]; then
    echo "[build_checkers] Generating helpers and building cases for ${SPLIT_FILE}..."
    if [[ -n "${MOCK_FLAG}" ]]; then
      "${PYTHON_BIN}" -m sspo.ec_gen 0 1 --problems "${SPLIT_FILE}" --out-dir cache/checkers/artifacts --mock
    else
      "${PYTHON_BIN}" -m sspo.ec_gen 0 1 --problems "${SPLIT_FILE}" --out-dir cache/checkers/artifacts
    fi
    "${PYTHON_BIN}" -m sspo.ec_build 0 1 --problems "${SPLIT_FILE}" --artifacts-dir cache/checkers/artifacts --cases-dir cache/checkers/cases
  fi
done

"${PYTHON_BIN}" -c "from sspo.atomic_io import write_done_marker; write_done_marker('cache/checkers/done', {'config': '${CONFIG}'})"
echo "[build_checkers] Completed."
