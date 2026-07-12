#!/usr/bin/env bash
# End-to-end 200-sample LoRA smoke test on GPU 0.
#   1. build a small dataset subset
#   2. train the LoRA adapter (--smoke) and evaluate on a held-out subset
#   3. run inference on 2 sample images
#   4. print the leaderboard
#
# Env knobs:
#   SMOKE_LIMIT    rows imported per source per split (default 400)
#   SMOKE_SOURCES  space-separated sources (default: vqa_rad iu_xray vqa_med)
#   SMOKE_GPU      GPU index (default 0)
set -euo pipefail

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

SMOKE_LIMIT="${SMOKE_LIMIT:-400}"
SMOKE_GPU="${SMOKE_GPU:-0}"
SMOKE_SOURCES="${SMOKE_SOURCES:-vqa_rad iu_xray vqa_med}"

echo "==> [1/4] Building dataset subset (limit ${SMOKE_LIMIT} per source; sources: ${SMOKE_SOURCES})"
# shellcheck disable=SC2086
python -m src.build_dataset --sources ${SMOKE_SOURCES} --limit "${SMOKE_LIMIT}"

echo "==> [2/4] Smoke training + eval on GPU ${SMOKE_GPU}"
python -m src.train_lora --smoke --gpu "${SMOKE_GPU}" --eval-after

RUN_ID="$(python -c "from src.registry import load_registry; r=load_registry(); print(r[-1]['run_id'] if r else '')")"
echo "==> Latest run_id: ${RUN_ID}"

echo "==> [3/4] Inference on 2 sample test images"
CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" RUN_ID="${RUN_ID}" python - <<'PY'
import os
from src.dataset import load_rows, open_image
from src.infer import load_for_inference, generate_answer

run_id = os.environ["RUN_ID"]
rows = load_rows("data/processed", "test")
vqa = next((r for r in rows if r["task"] == "vqa"), None)
report = next((r for r in rows if r["task"] == "report"), None)
samples = [r for r in (vqa, report) if r]

model, tok = load_for_inference(f"outputs/runs/{run_id}/lora")
for r in samples:
    img = open_image(r)
    mx = 64 if r["task"] == "vqa" else 256
    out = generate_answer(model, tok, img, r["prompt"], mx)
    print(f"\n[{r['task']}] {r['prompt']}")
    print(f"  gold: {r['target'][:160]}")
    print(f"  pred: {out[:160]}")
PY

echo "==> [4/4] Leaderboard"
python scripts/leaderboard.py

echo "==> Smoke test complete."
