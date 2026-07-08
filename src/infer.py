"""Inference helpers + CLI for the finetuned VLM.

Also used by ``src.eval``. Loading a saved adapter directory restores the base
model + LoRA automatically (Unsloth reads the adapter config).

Examples
--------
    python -m src.infer --run-id <id> --image data/raw/vqa_rad/images/x.png \
        --question "Is there cardiomegaly?"
    python -m src.infer --run-id <id> --image path.png --report
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

from .datasets_registry import REPORT_PROMPT, VQA_PROMPT
from .registry import RUNS_DIR
from .utils import ROOT, abs_from_root, load_yaml


def resolve_model_ref(
    run_id: Optional[str], adapter: Optional[str], base_model: Optional[str]
) -> str:
    """Return the model path/name to load (adapter dir if given, else base)."""
    if adapter:
        return adapter
    if run_id:
        lora_dir = RUNS_DIR / run_id / "lora"
        if not lora_dir.exists():
            raise FileNotFoundError(f"No adapter at {lora_dir}")
        return str(lora_dir)
    if base_model:
        return base_model
    raise ValueError("Provide one of --run-id, --adapter, or --base-model.")


def load_for_inference(
    model_ref: str, load_in_4bit: bool = True, max_seq_length: int = 2048
) -> Tuple[object, object]:
    # Avoid torch.compile/inductor (missing nvcc under WSL).
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_ref,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def generate_answer(
    model, tokenizer, image, prompt: str, max_new_tokens: int = 128
) -> str:
    """Generate a text response for a single (image, prompt) pair."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
    )
    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with the finetuned VLM.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--image", required=True)
    p.add_argument("--question", default=None, help="VQA question")
    p.add_argument("--report", action="store_true", help="generate a report instead")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    from PIL import Image

    args = parse_args()
    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    gpu = args.gpu if args.gpu is not None else cfg.get("gpu", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    model_ref = resolve_model_ref(args.run_id, args.adapter, args.base_model)
    model, tokenizer = load_for_inference(
        model_ref,
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
    )

    image = Image.open(abs_from_root(args.image)).convert("RGB")
    if args.report or not args.question:
        prompt = REPORT_PROMPT
        max_new = args.max_new_tokens or 256
    else:
        prompt = VQA_PROMPT.format(q=args.question)
        max_new = args.max_new_tokens or 64

    print("PROMPT:", prompt)
    print("OUTPUT:", generate_answer(model, tokenizer, image, prompt, max_new))


if __name__ == "__main__":
    main()
