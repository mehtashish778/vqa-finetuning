#!/usr/bin/env bash
# Cross-site (external) evaluation of a trained run on the SLAKE chest X-ray set.
# Builds the cross-site set if missing, then scores the run's adapter + frozen
# base. Results go to the run dir (crosssite_<name>.json) and a `crosssite`
# field on the registry row; the in-domain metrics.json is NOT touched.
#
# Usage:
#   bash scripts/eval_crosssite.sh <run_id>
#   CS_GPU=1 CS_NUM_SAMPLES=200 bash scripts/eval_crosssite.sh <run_id>
#
# Env knobs:
#   CS_GPU           GPU index (default 0)
#   CS_NUM_SAMPLES   subset size (default: full set)
#   CS_NAME          cross-site set name (default slake_xray_en)
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/eval_crosssite.sh <run_id>" >&2
  exit 1
fi
RUN_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_CACHE="${ROOT_DIR}/data/hf_cache/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

GPU="${CS_GPU:-0}"
NAME="${CS_NAME:-slake_xray_en}"
EVAL_PATH="data/crosssite/${NAME}.jsonl"

if [ ! -f "${EVAL_PATH}" ]; then
  echo "==> Cross-site set missing; building ${EVAL_PATH}"
  python -m src.build_crosssite --name "${NAME}"
fi

EXTRA=()
if [ -n "${CS_NUM_SAMPLES:-}" ]; then
  EXTRA+=(--num-samples "${CS_NUM_SAMPLES}")
fi

echo "==> Cross-site eval: run=${RUN_ID} set=${NAME} gpu=${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m src.eval \
  --run-id "${RUN_ID}" \
  --crosssite "${EVAL_PATH}" \
  --crosssite-name "${NAME}" \
  "${EXTRA[@]}"
