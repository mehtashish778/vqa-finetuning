# VLM Chest X-Ray Finetuning Pipeline (VQA + Report Generation)

Automated, portable pipeline to finetune a Vision-Language Model (default
`unsloth/Qwen3-VL-4B-Instruct`) for two chest X-ray tasks:

1. **VQA (primary)** — image + question -> short answer (`VQA-RAD`)
2. **Report generation (secondary)** — image -> findings paragraph (`IU X-Ray`)

One base model, one LoRA adapter, mixed multi-task SFT with **Unsloth** +
TRL `SFTTrainer`. Every training run is recorded in a **model registry** so
you can train many variants and compare their benchmarks.

---

## Quickstart (WSL / Linux)

```bash
# 1. Create the venv and install deps (asserts GPUs are visible first)
bash scripts/setup_env.sh
source .venv/bin/activate

# 2. Download + build the unified dataset (open datasets only)
bash scripts/download_data.sh

# 3. Run the 200-sample LoRA smoke test on GPU 0 (end to end)
bash scripts/run_smoke.sh
```

`scripts/setup_env.sh` + `scripts/download_data.sh` reproduce the environment
and data on any GPU server, so the pipeline is portable.

---

## Pipeline stages

| Stage | Command | Output |
|-------|---------|--------|
| Env | `scripts/setup_env.sh` | `.venv/` |
| Import + build | `python -m src.build_dataset` | `data/processed/{train,val,test}.jsonl`, `stats.json` |
| Train | `python -m src.train_lora --config configs/train.yaml` | `outputs/runs/<run_id>/lora/` |
| Evaluate | `python -m src.eval --run-id <run_id>` | `metrics.json`, updated `registry.jsonl` |
| Infer | `python -m src.infer --run-id <run_id> --image path --question "..."` | stdout |
| Leaderboard | `python scripts/leaderboard.py` | `outputs/leaderboard.md` |

---

## Data

Open datasets only (closed data like MIMIC can be added later via a new
adapter in `src/datasets_registry.py` with no other changes):

- **VQA-RAD** — `flaviagiammarino/vqa-rad` (radiology Q&A, official train/test).
- **IU X-Ray** — `dz-osamu/IU-Xray` (chest X-ray + findings, official
  train/val/test). Falls back to `Shrey-1329/cxiu_hf_dataset` if the primary
  repo has no image files.

Unified storage schema (one JSON object per line):

```json
{"task":"vqa","source":"vqa_rad","access":"open","image":"data/raw/vqa_rad/images/xxx.png","study_id":"...","prompt":"[TASK: VQA] Question: ... Answer briefly.","target":"yes","answer_type":"closed","split":"train"}
```

The test split is **frozen** and a `data_version` hash is recorded so every
model is benchmarked on the exact same examples.

---

## Model registry

- Each run -> `outputs/runs/<run_id>/` with `config.snapshot.yaml`,
  `run.json`, `metrics.json`, `train.log`, and the `lora/` adapter (the
  finetuned model artifact — adapter only by default).
- `outputs/registry.jsonl` — append-only index (one row per run: config +
  benchmarks + `adapter_path`).
- `python scripts/leaderboard.py` regenerates `outputs/leaderboard.md`
  sorted by a chosen metric.

`registry.jsonl` and `leaderboard.md` are committed to git so benchmark
history travels across servers even though the raw adapters are gitignored.

### Running many experiments (2 GPUs)

Each run pins one GPU via `--gpu`, so you can run two experiments at once:

```bash
python -m src.train_lora --run-name expA --gpu 0 --lora-r 16 &
python -m src.train_lora --run-name expB --gpu 1 --lora-r 32 &
wait
python scripts/leaderboard.py
```

---

## Troubleshooting

- **Qwen3-VL fails to load** — almost always a `transformers` version
  mismatch with Unsloth. Pin to the version referenced by the current Unsloth
  Qwen3-VL notebook, then re-run a 10-sample check.
- **CUDA OOM on 12GB** — lower `image_max_side`, `max_seq_length`, or
  `gradient_accumulation_steps` in `configs/train.yaml`.
