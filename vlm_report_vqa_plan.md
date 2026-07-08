# VLM Finetuning Plan — Chest X-Ray VQA (primary) + Report Generation

A simple, from-scratch plan to finetune one Vision-Language Model (VLM) for **two tasks**:

1. **VQA (PRIMARY)** — image + question → short answer
2. **Report generation (SECONDARY)** — image → radiology report (FINDINGS + IMPRESSION)

One base model, one LoRA adapter, mixed multi-task SFT.

**Data policy:** use **open datasets only** for now. Keep the pipeline generic so **closed / credentialed data (e.g. MIMIC-CXR) can be plugged in later** without code changes — just add rows to the unified JSONL.

**Stack:** Qwen 3.5 VL series, finetuned with **Unsloth** (`FastVisionModel`) + TRL `SFTTrainer`.

---

## 0. TL;DR

- **Priority:** VQA first, report generation second.
- **Base model:** Qwen 3.5 VL series (e.g. `unsloth/Qwen3.5-VL-8B-Instruct` or the 2B/4B variant you pick)
- **Framework:** **Unsloth** — `FastVisionModel` + `UnslothVisionDataCollator` + TRL `SFTTrainer`
- **Method:** LoRA SFT (4-bit / QLoRA), chat format `image + instruction → text`
- **Data (open only for now):** VQA — VQA-RAD, SLAKE; reports — IU X-Ray (Open-i). Closed data (MIMIC-CXR, etc.) = future, optional.
- **Mix:** start **VQA-only**, then add reports → settle around **~60% VQA / ~40% report**, task tag in every prompt.
- **Eval:** VQA → accuracy / F1 (primary metric); reports → BLEU/ROUGE + CheXbert/RadGraph F1 (evaluated separately)

> **Version note:** Unsloth + Qwen 3.5 VL is sensitive to the `transformers` version. Pin a compatible pair (e.g. `transformers==5.3.0` worked for Qwen3.5 vision per Unsloth issue #4202) and a recent `unsloth`. Verify on a 10-sample smoke run before full training.

---

## 1. Objective

| Priority | Task | Input | Output | Success = |
|----------|------|-------|--------|-----------|
| **1 (primary)** | VQA | Image + question | Short answer (yes/no / label / phrase) | High closed-question accuracy, sensible open answers |
| 2 (secondary) | Report generation | Chest X-ray image | FINDINGS + IMPRESSION text | Correct findings, no hallucinations, readable |

**Goal:** a single deployable model that primarily *answers questions* about an X-ray, and can also *describe* it in a report.

---

## 2. Data required

> **Policy:** only the **OPEN** datasets below are used now. The **CLOSED / FUTURE** ones are listed so the pipeline is designed to accept them later — no schema change needed.

### 2.1 VQA (PRIMARY) — open, use now
| Dataset | Content | Access |
|---------|---------|--------|
| VQA-RAD | ~3.5k radiology Q&A | **Open** |
| SLAKE | Med VQA incl. CXR (bilingual) | **Open** |
| ROCO / PMC-VQA | Large open medical image–text / Q&A | **Open** |

Row schema:
```json
{ "path": "images/study_001.png", "patient_id": "p123", "question": "Is there pleural effusion?", "answer": "Yes", "answer_type": "yes_no" }
```

### 2.2 Report generation (SECONDARY) — open, use now
| Dataset | Content | Access |
|---------|---------|--------|
| IU X-Ray (Open-i) | ~7k images + reports | **Open**, main report source for now |
| PadChest | Reports + labels (Spanish) | Registration (semi-open) |

Row schema:
```json
{ "path": "images/study_001/frontal.png", "patient_id": "p123", "report": "FINDINGS: ... IMPRESSION: ...", "view": "PA" }
```

### 2.3 Closed / credentialed — FUTURE (optional, not used now)
| Dataset | Content | Access | Adds |
|---------|---------|--------|------|
| MIMIC-CXR | ~227k reports + images | PhysioNet credentialed | Scale for reports |
| MIMIC-CXR-VQA | Report-derived Q&A | PhysioNet credentialed | Scale for VQA |

> When access is granted later: convert to the same unified JSONL (Section 3), tag rows with `"source": "mimic"`, add to `train.jsonl`, re-run training. No pipeline changes.

### 2.4 Optional augmentation (open)
- Findings labels (CheXpert / NIH) — auxiliary / weak supervision
- Bounding boxes — for "where is X?" VQA
- LLM-generated Q&A from open reports (IU X-Ray) — cheap way to scale VQA

---

## 3. Unified data format (for training)

Two layers:

**(a) Storage JSONL** — simple, one line per example (easy to build/split/inspect):

```json
{"task": "vqa",    "source": "vqa_rad", "access": "open", "image": "images/b.png", "patient_id": "p2", "prompt": "[TASK: VQA] Question: Is there cardiomegaly? Answer briefly.", "target": "Yes"}
{"task": "report", "source": "iu_xray", "access": "open", "image": "images/a.png", "patient_id": "p1", "prompt": "[TASK: REPORT] Write a chest X-ray report with FINDINGS and IMPRESSION.", "target": "FINDINGS: ... IMPRESSION: ..."}
```

The `source` + `access` fields let you filter (`access == "open"`) now and simply flip in closed data later.

**(b) Unsloth messages format** — what `FastVisionModel` / `UnslothVisionDataCollator` consume. Convert each row on load into a `messages` conversation with an image content block:

```python
def to_messages(row):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": row["image"]},   # PIL image or path
                {"type": "text",  "text": row["prompt"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": row["target"]},
            ]},
        ]
    }
```

Rules:
- **Task tag** (`[TASK: REPORT]` / `[TASK: VQA]`) in every prompt.
- Split by **patient/study**, never by image (avoid leakage).
- Splits: 80% train / 10% val / 10% test.
- Train on responses only (mask the user/vision prefix) — configure via `UnslothVisionDataCollator(..., train_on_responses_only=True, instruction_part=..., response_part=...)` using Qwen 3.5's chat markers.

---

## 4. Repo structure (fresh repo)

```
vlm-cxr-finetune/
├── README.md
├── requirements.txt
├── configs/
│   └── train.yaml            # model, lora, lr, epochs, data mix
├── data/
│   ├── raw/                  # downloaded images + reports/QA
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── src/
│   ├── build_dataset.py      # raw -> unified JSONL, patient-level splits
│   ├── dataset.py            # loads JSONL -> Unsloth messages format
│   ├── train_lora.py         # Unsloth FastVisionModel + SFTTrainer
│   ├── infer.py              # generate report / answer for an image
│   └── eval.py               # per-task metrics
└── scripts/
    ├── download_data.sh
    └── run_train.sh
```

---

## 5. Execution steps

### Step 1 — Environment (Unsloth)
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install unsloth
# Pin transformers to a Qwen3.5-VL-compatible version (see version note in TL;DR)
pip install "transformers==5.3.0" trl datasets pillow tqdm evaluate
```
> Unsloth pulls in `peft`/`accelerate`/`bitsandbytes`. If Qwen 3.5 VL fails to load, the fix is almost always the `transformers` version — try the version referenced in the Unsloth Qwen3.5 notebook/issue tracker.

### Step 2 — Get data (open only)
- Download **VQA-RAD** (primary task) first.
- Optionally add **SLAKE** for more VQA coverage.
- Download **IU X-Ray** (Open-i) for the report task.
- Place under `data/raw/`. (Leave a `data/raw/closed/` folder empty as a placeholder for future MIMIC data.)

### Step 3 — Build unified dataset
`src/build_dataset.py`:
- Parse Q&A and reports into the unified schema (Section 3), tagging `task`, `source`, `access="open"`.
- Add task tags, do patient-level split, write `train/val/test.jsonl`.
- Provide a `--access open` filter (default) so closed rows are ignored until you opt in.

### Step 4 — Train (Unsloth LoRA SFT)
`src/train_lora.py` — core skeleton:

```python
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 1. Load Qwen 3.5 VL in 4-bit
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-VL-8B-Instruct",   # pick your variant
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# 2. Attach LoRA to vision + language layers
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=32, target_modules="all-linear",
)

# 3. dataset = mixed report+VQA rows -> to_messages() (Section 3b)
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        bf16=is_bf16_supported(), fp16=not is_bf16_supported(),
        optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", warmup_steps=10,
        logging_steps=20, save_steps=200, save_total_limit=2,
        seed=3407, output_dir="outputs",
        # REQUIRED for vision SFT:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    ),
)
trainer.train()
model.save_pretrained("outputs/lora")   # LoRA adapter
tokenizer.save_pretrained("outputs/lora")
```

Suggested hyperparameters:
| Param | Value |
|-------|-------|
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 2e-4 (Unsloth LoRA default) |
| Batch size | 1–2 |
| Grad accumulation | 8–16 |
| Epochs | 1–3 (early stop on val) |
| Quantization | 4-bit (QLoRA) |
| Precision | bf16 (fp16 fallback) |
| Grad checkpointing | `"unsloth"` |
| `max_seq_length` | 2048 (raise if reports are long) |

> Note: Unsloth LoRA uses a higher LR (~2e-4) than full finetune. Shuffle mixed report+VQA rows; use `train_on_responses_only` in the collator so loss is on the answer/report, not the prompt.

### Step 5 — Evaluate (per task, separately)
`src/eval.py`:
| Task | Metrics |
|------|---------|
| **VQA (primary)** | Accuracy (closed), exact-match / token-F1 (open), per answer-type breakdown — **main headline number** |
| Report | BLEU, ROUGE-L, METEOR + CheXbert label F1 / RadGraph F1 + manual read of ~50 |

Always run the **frozen base model** with the same prompts first = your baseline to beat.

### Step 6 — Iterate
- **VQA is primary** → if VQA underperforms, keep VQA-only training longer or oversample VQA before adding reports.
- If reports hallucinate → more report data, lower lr, fewer epochs.
- If adding reports hurts VQA → lower report share (e.g. 60/40 → 70/30 VQA) or use a staged VQA→report schedule.

---

## 6. Multi-task training options

| Option | How | When |
|--------|-----|------|
| **VQA-first staged (recommended)** | Train VQA-only → then mixed ~60% VQA / 40% report | VQA is the priority |
| Mixed from start | Shuffle VQA+report in one JSONL (~60/40) | If VQA data is large enough |
| Two adapters | Separate LoRA per task, route by prompt | If tasks conflict badly |

Start with **VQA-only** to lock in the primary metric, then add reports and re-check VQA didn't regress.

---

## 7. Milestones

| Week | Deliverable |
|------|-------------|
| 1 | Repo + env + **VQA-only** finetune (VQA-RAD/SLAKE), accuracy on test — primary milestone |
| 2 | Add IU X-Ray report data, mixed run (~60/40), confirm VQA didn't regress |
| 3 | Report metrics (BLEU/CheXbert) + tune mix; final open-data model |
| 4 | Error analysis; wire the closed-data hook (MIMIC) for future without running it |

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Data leakage across splits | Split by patient/study |
| Report task hurts primary VQA | VQA-first staged training; monitor VQA every stage |
| Report hallucination | Clinical F1 metric, manual review, lower lr |
| Small open VQA set | Add SLAKE; LLM-generated Q&A from open IU X-Ray reports |
| Closed data unavailable | Whole plan runs on open data; MIMIC is an optional future add-on |

---

## 9. Future: integrating closed data

Designed-in, no rework needed:
1. Get credentialed access (e.g. PhysioNet for MIMIC-CXR / MIMIC-CXR-VQA).
2. Convert to the unified JSONL (Section 3) with `"access": "closed"`, `"source": "mimic"`.
3. Drop into `data/raw/closed/`, rebuild with `--access all`.
4. Re-run the same Unsloth training; optionally raise report/VQA scale.

Keep open-data checkpoints as the reproducible public baseline.

---

## 10. Definition of done

- One LoRA checkpoint (open data) that beats the frozen base on **VQA (primary)** and on report metrics.
- Reproducible: `scripts/run_train.sh` regenerates the model from `data/processed/*.jsonl`.
- Eval report with per-task numbers vs baseline, VQA highlighted as the headline result.
- Closed-data hook (`--access all`) present and documented, but not required to reproduce.
