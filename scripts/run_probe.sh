#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/run_probe.sh [--k 8] [--out-dir results/probe] [--dry-run] [--help]
#
# Description:
#   Optional single-step probe script verifying Theorem 1(b) (blind-direction mass
#   transfer W_1 -> T) and Proposition C (shared-prefix reversal thresholds).
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

K_VAL=8
OUT_DIR="results/probe"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --k)
      K_VAL="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,9p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] run_probe.sh: k=${K_VAL} out_dir=${OUT_DIR}"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

"${PYTHON_BIN}" -m sspo.probe --k "${K_VAL}" --out-dir "${OUT_DIR}"
