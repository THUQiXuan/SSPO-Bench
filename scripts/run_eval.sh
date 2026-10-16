#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/run_eval.sh <run_dir> [ckpt] [--config configs/pilot.yaml] [--split test] [--mock] [--dry-run] [--help]
#
# Arguments:
#   <run_dir>  Run directory under results/, e.g. results/sspo_seed0
#   [ckpt]     Checkpoint identifier (default: latest)
#
# Description:
#   Evaluates `<run_dir>` at checkpoint `[ckpt]` on `n_eval = 64` rollouts per problem,
#   computes unbiased `pass@k` and random-subset / exact `select@k`, writes
#   `<run_dir>/eval/<split>_<ckpt>.jsonl`, `<run_dir>/summary.json`, and updates
#   `results/tables/*.csv` and `results/report.md`.
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

RUN_DIR=""
CKPT="latest"
CONFIG="configs/pilot.yaml"
SPLIT="test"
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
    --split)
      SPLIT="$2"
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
      if [[ -z "${RUN_DIR}" ]]; then
        RUN_DIR="$1"
      elif [[ "${CKPT}" == "latest" ]]; then
        CKPT="$1"
      else
        echo "ERROR: Unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: Missing required <run_dir> argument" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] run_eval.sh: run_dir=${RUN_DIR} ckpt=${CKPT} config=${CONFIG} split=${SPLIT}"
  exit 0
fi

mkdir -p "${RUN_DIR}/eval" logs
CMD=("${PYTHON_BIN}" -m sspo.trainer eval --run-dir "${RUN_DIR}" --ckpt "${CKPT}" --config "${CONFIG}" --split "${SPLIT}" --resume)
if [[ -n "${MOCK_FLAG}" ]]; then
  CMD+=("--mock")
fi
RUN_TAG="$(basename "${RUN_DIR}")"
"${CMD[@]}" | tee -a "logs/eval_${RUN_TAG}_${CKPT}.log"
