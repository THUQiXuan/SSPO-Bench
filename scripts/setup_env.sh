#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/setup_env.sh [--env-dir .venv] [--dev-only] [--dry-run] [--help]
#
# Description:
#   Creates an isolated Python environment (using `uv` or `conda`/`venv`), verifies
#   NVIDIA GPU driver & CUDA compatibility (on Linux GPU machines), and installs
#   locked dependencies from `requirements-gpu.txt` (or `requirements-dev.txt` when
#   `--dev-only` is passed). Writes `cache/env_setup.done` upon completion.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_DIR=".venv"
DEV_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --dev-only)
      DEV_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] setup_env.sh: env_dir=${ENV_DIR} dev_only=${DEV_ONLY}"
  exit 0
fi

mkdir -p logs cache
if [[ -f "cache/env_setup.done" ]]; then
  echo "[setup_env] Environment setup already completed (cache/env_setup.done exists). Skipping."
  exit 0
fi

if [[ "${DEV_ONLY}" -eq 0 ]]; then
  echo "[setup_env] Checking GPU hardware and NVIDIA driver..."
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found! Target machine must have NVIDIA drivers (>=535 for CUDA 12.x) and H100/A100 GPUs." >&2
    echo "       For local CPU-only testing on macOS/CPU, run: bash scripts/setup_env.sh --dev-only" >&2
    exit 1
  fi
  DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  echo "[setup_env] Detected ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM_MIB} MiB), Driver major=${DRIVER_VER}"
  if [[ "${DRIVER_VER}" -lt 525 ]]; then
    echo "ERROR: NVIDIA driver version (${DRIVER_VER}) is too old for CUDA 12.x / vLLM. Please use driver >= 535." >&2
    exit 1
  fi
fi

if command -v uv >/dev/null 2>&1; then
  echo "[setup_env] Creating virtual environment at ${ENV_DIR} via uv..."
  uv venv "${ENV_DIR}" --python 3.11 || uv venv "${ENV_DIR}"
  if [[ "${DEV_ONLY}" -eq 1 ]]; then
    uv pip install --python "${ENV_DIR}/bin/python" -r requirements-dev.txt
  else
    uv pip install --python "${ENV_DIR}/bin/python" -r requirements-gpu.txt
  fi
  uv pip install --python "${ENV_DIR}/bin/python" --no-deps -e .
else
  echo "[setup_env] Creating virtual environment at ${ENV_DIR} via python3 -m venv..."
  python3 -m venv "${ENV_DIR}"
  "${ENV_DIR}/bin/pip" install --upgrade pip
  if [[ "${DEV_ONLY}" -eq 1 ]]; then
    "${ENV_DIR}/bin/pip" install -r requirements-dev.txt
  else
    "${ENV_DIR}/bin/pip" install -r requirements-gpu.txt
  fi
  "${ENV_DIR}/bin/pip" install --no-deps -e .
fi

"${ENV_DIR}/bin/python" -m sspo.atomic_io_cli mark-done cache/env_setup.done 2>/dev/null || \
  "${ENV_DIR}/bin/python" -c "from sspo.atomic_io import write_done_marker; write_done_marker('cache/env_setup.done', {'dev_only': bool(${DEV_ONLY})})"
echo "[setup_env] Done."
