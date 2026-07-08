#!/usr/bin/env bash
# Full (non-smoke) training run on a single GPU.
# Extra args are forwarded to src.train_lora (e.g. --lora-r 32 --eval-after).
#
# Env knobs:
#   FULL_GPU        GPU index (default 0)
#   FULL_RUN_NAME   run label (default "full")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_CACHE="${ROOT_DIR}/data/hf_cache/datasets"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

GPU="${FULL_GPU:-0}"
RUN_NAME="${FULL_RUN_NAME:-full}"

echo "==> Full training: run_name=${RUN_NAME} gpu=${GPU} args=$*"
python -m src.train_lora --run-name "${RUN_NAME}" --gpu "${GPU}" "$@"
