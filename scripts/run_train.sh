#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/run_train.sh <arm> [seed] [--config configs/pilot.yaml] [--turns 1|2] [--mock] [--dry-run] [--help]
#
# Arguments:
#   <arm>   One of: sspo | drgrpo_pass1 | pkpo_pass8 | checker_reward | frozen
#   [seed]  Random seed integer (default: 0)
#
# Description:
#   Runs or resumes (`--resume` default) training for `<arm>` with seed `[seed]`.
#   Saves checkpoints every 10 steps in `results/<arm>_seed<seed>/checkpoints/`,
#   maintains `latest` pointer, logs per-step metrics to `results/<arm>_seed<seed>/metrics.jsonl`,
#   and writes `results/<arm>_seed<seed>/done` upon completion.
# ==============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  sed -n '2,15p' "${BASH_SOURCE[0]}"
  exit 1
fi

ARM=""
SEED="0"
CONFIG="configs/pilot.yaml"
TURNS=""
RUN_DIR=""
MOCK_FLAG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '2,15p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --turns)
      TURNS="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
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
    *)
      if [[ -z "${ARM}" ]]; then
        ARM="$1"
      elif [[ "${SEED}" == "0" && "$1" =~ ^[0-9]+$ ]]; then
        SEED="$1"
      else
        echo "ERROR: Unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${ARM}" ]]; then
  echo "ERROR: Missing required <arm> argument (sspo | drgrpo_pass1 | pkpo_pass8 | checker_reward | frozen)" >&2
  exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="results/${ARM}_seed${SEED}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] run_train.sh: arm=${ARM} seed=${SEED} config=${CONFIG} run_dir=${RUN_DIR} turns=${TURNS:-1}"
  exit 0
fi

mkdir -p "${RUN_DIR}" logs
if [[ -f "${RUN_DIR}/done" ]]; then
  echo "[run_train] ${RUN_DIR}/done already exists. Skipping completed run."
  exit 0
fi

CMD=("${PYTHON_BIN}" -m sspo.trainer train --config "${CONFIG}" --arm "${ARM}" --seed "${SEED}" --run-dir "${RUN_DIR}" --resume)
if [[ -n "${TURNS}" ]]; then
  CMD+=("--turns" "${TURNS}")
fi
if [[ -n "${MOCK_FLAG}" ]]; then
  CMD+=("--mock")
fi

"${CMD[@]}" | tee -a "logs/train_${ARM}_seed${SEED}.log"
