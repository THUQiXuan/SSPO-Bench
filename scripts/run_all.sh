#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/run_all.sh [--config configs/pilot.yaml] [--smoke-only] [--mock] [--dry-run] [--help]
#
# Description:
#   End-to-end orchestrator that runs:
#     1. 10-minute smoke verification (`configs/smoke.yaml`)
#     2. Data preparation & frozen checker construction (`prepare_data.sh`, `build_checkers.sh`)
#     3. Gate 1 training-free diagnostics (`run_diag.sh`)
#     4. All 5 experimental arms (`sspo` x2 seeds, `drgrpo_pass1` x2 seeds, `pkpo_pass8`, `checker_reward`, `frozen`)
#     5. Full evaluation (`run_eval.sh`) & automatic table/report generation (`results/report.md`)
#   Every completed stage writes a `done` marker and is automatically skipped on re-run.
# ==============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/pilot.yaml"
SMOKE_ONLY=0
MOCK_FLAG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --smoke-only)
      SMOKE_ONLY=1
      shift
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
      sed -n '2,14p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] run_all.sh: config=${CONFIG} smoke_only=${SMOKE_ONLY} mock=${MOCK_FLAG}"
  exit 0
fi

mkdir -p results logs

# Stage 0: Smoke verification (8 problems, n=8, 2 steps)
if [[ ! -f "results/smoke_sspo/done" ]]; then
  echo "=== [Stage 0] Running 10-minute smoke verification (configs/smoke.yaml) ==="
  bash scripts/prepare_data.sh --source sample
  bash scripts/build_checkers.sh --config configs/smoke.yaml --mock
  bash scripts/run_train.sh sspo 0 --config configs/smoke.yaml --run-dir results/smoke_sspo --mock
  bash scripts/run_eval.sh results/smoke_sspo latest --config configs/smoke.yaml --mock
else
  echo "=== [Stage 0] Smoke verification already done (results/smoke_sspo/done). Skipping. ==="
fi

if [[ "${SMOKE_ONLY}" -eq 1 ]]; then
  echo "[run_all] --smoke-only specified. Smoke verification succeeded!"
  exit 0
fi

# Stage 1: Models, Data, Checkers
echo "=== [Stage 1] Preparing models, dataset, and frozen execution checkers ==="
bash scripts/download_models.sh --config "${CONFIG}"
if [[ -n "${MOCK_FLAG}" ]]; then
  bash scripts/prepare_data.sh --source sample
  bash scripts/build_checkers.sh --config "${CONFIG}" --mock
else
  bash scripts/prepare_data.sh
  bash scripts/build_checkers.sh --config "${CONFIG}"
fi

# Stage 2: Gate 1 Training-Free Diagnostics
echo "=== [Stage 2] Running Gate 1 Training-Free Diagnostics ==="
if [[ -n "${MOCK_FLAG}" ]]; then
  bash scripts/run_diag.sh --config "${CONFIG}" --mock
else
  bash scripts/run_diag.sh --config "${CONFIG}"
fi

# Stage 3: Five Training/Baseline Arms (sspo & drgrpo_pass1 with 2 seeds; pkpo_pass8, checker_reward, frozen with 1 seed)
ARMS_AND_SEEDS=(
  "frozen:0"
  "sspo:0"
  "sspo:1"
  "drgrpo_pass1:0"
  "drgrpo_pass1:1"
  "pkpo_pass8:0"
  "checker_reward:0"
)

echo "=== [Stage 3 & 4] Running Training Arms and Evaluation ==="
for ITEM in "${ARMS_AND_SEEDS[@]}"; do
  ARM="${ITEM%%:*}"
  SEED="${ITEM##*:}"
  RUN_DIR="results/${ARM}_seed${SEED}"
  if [[ -n "${MOCK_FLAG}" ]]; then
    bash scripts/run_train.sh "${ARM}" "${SEED}" --config "${CONFIG}" --run-dir "${RUN_DIR}" --mock
    bash scripts/run_eval.sh "${RUN_DIR}" latest --config "${CONFIG}" --mock
  else
    bash scripts/run_train.sh "${ARM}" "${SEED}" --config "${CONFIG}" --run-dir "${RUN_DIR}"
    bash scripts/run_eval.sh "${RUN_DIR}" latest --config "${CONFIG}"
  fi
done

echo "=== [Complete] All stages finished. Summary report at: results/report.md ==="
