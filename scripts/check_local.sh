#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/check_local.sh [--dry-run] [--help]
#
# Description:
#   Runs all 10 local GPU-free verification suites (Section 5 of specification):
#     1. bash -n syntax check and shellcheck (if installed) on all scripts/*.sh
#     2. pytest on tests/test_01_dp_correctness.py .. tests/test_10_eval_estimators.py
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,10p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] check_local.sh"
  exit 0
fi

echo "=== [1/3] Checking bash syntax (bash -n) on scripts/*.sh ==="
for s in scripts/*.sh; do
  bash -n "${s}"
  echo "  [OK] bash -n ${s}"
done

if command -v shellcheck >/dev/null 2>&1; then
  echo "=== [2/3] Running shellcheck on scripts/*.sh ==="
  shellcheck -S error scripts/*.sh
  echo "  [OK] shellcheck passed with 0 errors"
else
  echo "=== [2/3] shellcheck not installed; skipping ==="
fi

echo "=== [3/3] Running pytest suite (Tests 01 - 10) ==="
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x "../.venv_sspo_test/bin/python" ]]; then
  PYTHON_BIN="../.venv_sspo_test/bin/python"
fi

"${PYTHON_BIN}" -m pytest tests/ -v
echo "=== All local GPU-free checks PASSED ==="
