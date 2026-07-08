#!/usr/bin/env bash
# Create a portable Python venv for the VLM finetuning pipeline.
# Target: WSL Ubuntu / Linux with NVIDIA GPUs.
#
# Usage:
#   bash scripts/setup_env.sh
#   source .venv/bin/activate
set -euo pipefail

# Resolve repo root (this script lives in <root>/scripts).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "==> Repo root: ${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 1. Preflight: GPUs must be visible (this pipeline needs CUDA).
# ---------------------------------------------------------------------------
echo "==> Preflight: checking for NVIDIA GPUs (nvidia-smi)..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. In WSL, install the NVIDIA CUDA driver on"
  echo "       Windows and use a CUDA-enabled WSL2 distro. Aborting."
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
  echo "ERROR: nvidia-smi failed to query GPUs. Aborting."
  exit 1
}
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
echo "==> Detected ${GPU_COUNT} GPU(s)."

# ---------------------------------------------------------------------------
# 2. Pin the HF cache inside the repo so a copied repo reuses downloads.
# ---------------------------------------------------------------------------
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_CACHE="${ROOT_DIR}/data/hf_cache/datasets"
mkdir -p "${HF_DATASETS_CACHE}"
echo "==> HF_HOME=${HF_HOME}"

# ---------------------------------------------------------------------------
# 3. Create + activate the venv.
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "==> Creating venv (.venv) with ${PYTHON_BIN}..."
  "${PYTHON_BIN}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 4. Install torch (CUDA wheel) first, then the rest.
# ---------------------------------------------------------------------------
echo "==> Installing torch from ${TORCH_INDEX_URL} ..."
pip install torch --index-url "${TORCH_INDEX_URL}"

echo "==> Installing pipeline requirements..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 5. Sanity check: torch sees CUDA, and Unsloth imports.
# ---------------------------------------------------------------------------
echo "==> Verifying torch CUDA visibility..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available to torch inside this venv."
PY

cat <<'NOTE'

==> Environment ready.
    Activate it with:  source .venv/bin/activate

    NOTE: If Qwen3-VL fails to load during training, it is almost always a
    `transformers` version mismatch with Unsloth. Pin `transformers` to the
    version referenced by the current Unsloth Qwen3-VL notebook and re-run a
    small (10-sample) check before a full run.
NOTE
