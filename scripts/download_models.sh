#!/usr/bin/env bash
# ==============================================================================
# Usage:
#   bash scripts/download_models.sh [--config configs/pilot.yaml] [--dry-run] [--help]
#
# Description:
#   Downloads the policy/helper model (`Qwen/Qwen3-8B` or `Qwen/Qwen3-4B` selected by
#   hardware preset) from Hugging Face Hub using the built-in `HF_TOKEN`.
#   Writes `models/done` upon completion.
# ==============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/pilot.yaml"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
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

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] download_models.sh: config=${CONFIG} python=${PYTHON_BIN}"
  exit 0
fi

mkdir -p models logs
if [[ -f "models/done" ]]; then
  echo "[download_models] models/done exists. Skipping download."
  exit 0
fi

MODEL_NAME="$("${PYTHON_BIN}" -c "from sspo.config import load_config; print(load_config('${CONFIG}')['hardware_preset']['model_name'])")"
echo "[download_models] Resolved model for hardware preset: ${MODEL_NAME}"

if [[ "${SSPO_MOCK_LLM:-0}" == "1" ]]; then
  echo "[download_models] SSPO_MOCK_LLM=1 set; skipping remote Hugging Face weight download."
  "${PYTHON_BIN}" -c "from sspo.atomic_io import write_done_marker; write_done_marker('models/done', {'model': '${MODEL_NAME}', 'mock': True})"
  exit 0
fi

"${PYTHON_BIN}" -c "
import os
from huggingface_hub import snapshot_download
from sspo.atomic_io import write_done_marker

model_id = '${MODEL_NAME}'
local_dir = os.path.join('models', model_id.replace('/', '--'))
snapshot_download(repo_id=model_id, local_dir=local_dir, token=os.environ.get('HF_TOKEN', ''))
write_done_marker('models/done', {'model_name': model_id, 'local_dir': local_dir})
print('[download_models] Successfully downloaded', model_id, 'to', local_dir)
"
