#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/collect_results.sh [--output sspo_results_bundle.zip] [--dry-run] [--help]
#
# Description:
#   Packages all experiment metrics (`results/*/metrics.jsonl`, `results/*/manifest.json`,
#   `results/*/summary.json`, `results/*/eval/*.jsonl`), diagnostic results (`results/diag/`),
#   summary CSV tables (`results/tables/*.csv`), `results/report.md`, and `logs/` into a
#   lightweight zip archive (excluding heavy model checkpoint weights) ready to send back.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_ZIP="sspo_results_bundle.zip"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output|-o)
      OUT_ZIP="$2"
      shift 2
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

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] collect_results.sh: output=${OUT_ZIP}"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

# Refresh tables and report.md before archiving
"${PYTHON_BIN}" -c "from sspo.eval import generate_tables_and_report; generate_tables_and_report('results')" || true

rm -f "${OUT_ZIP}"
zip -r "${OUT_ZIP}" \
  results/ \
  logs/ \
  data/filter_stats.json \
  -x "results/*/checkpoints/*.pt" \
  -x "results/*/rollout_cache/*" \
  -x "*.tmp.*" >/dev/null || true

echo "[collect_results] Created results archive: ${OUT_ZIP}"
ls -lh "${OUT_ZIP}"
