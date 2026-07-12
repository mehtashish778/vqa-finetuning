#!/usr/bin/env bash
# Download the open datasets and build the unified train/val/test JSONL.
# Extra args are forwarded to src.build_dataset, e.g.:
#   bash scripts/download_data.sh --limit 300
#   bash scripts/download_data.sh --sources vqa_rad iu_xray vqa_med
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Pin HF cache inside the repo for portability.
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_CACHE="${ROOT_DIR}/data/hf_cache/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

# Activate venv if present (harmless if already active).
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> Building unified dataset (args: $*)"
python -m src.build_dataset "$@"
