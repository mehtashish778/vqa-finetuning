"""Finetune a Qwen3-VL model with Unsloth QLoRA + TRL SFTTrainer.

Config-driven (configs/train.yaml) with CLI overrides. Every run is recorded in
the model registry and written to its own ``outputs/runs/<run_id>/`` directory.

Examples
--------
    python -m src.train_lora --smoke                 # 200-sample smoke test
    python -m src.train_lora --run-name expA --gpu 0 --lora-r 32
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from .utils import ROOT, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth QLoRA finetuning.")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny 200-sample run")
    p.add_argument("--eval-after", action="store_true",
                   help="run in-domain src.eval automatically after training")
    p.add_argument("--crosssite-after", action="store_true",
                   help="run cross-site eval(s) after training "
                        "(builds each set if missing)")
    p.add_argument("--crosssite-name", default="slake_xray_en",
                   help="comma-separated cross-site set name(s), or 'all' for "
                        "every registered set")
    return p.parse_args()


def resolve_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    c = dict(cfg)
    if args.run_name is not None:
        c["run_name"] = args.run_name
    if args.gpu is not None:
        c["gpu"] = args.gpu
    if args.base_model is not None:
        c["base_model"] = args.base_model
    if args.lora_r is not None:
        c["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        c["lora_alpha"] = args.lora_alpha
    if args.lr is not None:
        c["learning_rate"] = args.lr
    if args.epochs is not None:
        c["num_train_epochs"] = args.epochs
    if args.max_steps is not None:
        c["max_steps"] = args.max_steps

    c["max_samples"] = args.max_samples

    if args.smoke:
        smoke = cfg.get("smoke", {}) or {}
        c["max_samples"] = args.max_samples or smoke.get("max_samples", 200)
        c["max_steps"] = smoke.get("max_steps", 30)
        c["logging_steps"] = smoke.get("logging_steps", 5)
        c["save_steps"] = smoke.get("save_steps", 30)
        c["_num_eval_samples"] = smoke.get("num_eval_samples", 40)
        if args.run_name is None:
            c["run_name"] = "smoke"
    return c


def _read_stats(processed_dir: Path) -> Dict[str, Any]:
    stats_path = processed_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_sft_config(SFTConfig, c: Dict[str, Any], run_dir: Path, bf16: bool):
    desired = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": c.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": c.get("gradient_accumulation_steps", 8),
        "warmup_steps": c.get("warmup_steps", 10),
        "num_train_epochs": c.get("num_train_epochs", 2),
        "max_steps": c.get("max_steps", -1),
        "learning_rate": c.get("learning_rate", 2e-4),
        "logging_steps": c.get("logging_steps", 20),
        "save_steps": c.get("save_steps", 200),
        "save_total_limit": c.get("save_total_limit", 2),
        "optim": c.get("optim", "adamw_8bit"),
        "weight_decay": c.get("weight_decay", 0.01),
        "lr_scheduler_type": c.get("lr_scheduler_type", "linear"),
        "seed": c.get("seed", 3407),
        "report_to": "none",
        "bf16": bf16,
        "fp16": not bf16,
        # Required for Unsloth vision SFT:
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        # seq length key name differs across TRL versions; include both.
        "max_seq_length": c.get("max_seq_length", 2048),
        "max_length": c.get("max_seq_length", 2048),
    }
    allowed = {f.name for f in dataclasses.fields(SFTConfig)}
    kwargs = {k: v for k, v in desired.items() if k in allowed}
    return SFTConfig(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    c = resolve_config(cfg, args)

    # Pin the GPU BEFORE importing torch/unsloth.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(c.get("gpu", 0))

    # Disable torch.compile/inductor (needs a working nvcc toolchain, which is
    # often missing under WSL). Set UNSLOTH_COMPILE_DISABLE=0 to re-enable.
    if c.get("disable_torch_compile", True):
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

    import torch  # noqa: E402
    from unsloth import FastVisionModel  # noqa: E402
    from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402

    from . import registry  # noqa: E402
    from .dataset import make_sft_dataset  # noqa: E402

    processed_dir = ROOT / str(c.get("processed_dir", "data/processed"))
    stats = _read_stats(processed_dir)
    data_version = stats.get("data_version", "unknown")
    sources = sorted(stats.get("counts_by_source", {}).keys()) or ["vqa_rad", "iu_xray"]

    print("==> Building training dataset...")
    dataset = make_sft_dataset(
        processed_dir=processed_dir,
        split="train",
        data_mix=c.get("data_mix"),
        max_samples=c.get("max_samples"),
        seed=int(c.get("seed", 3407)),
        balance_tasks=bool(c.get("balance_tasks", True)),
    )
    n_train = len(dataset)
    print(f"    {n_train} training examples")

    handle = registry.start_run(
        {
            "run_name": c.get("run_name", "run"),
            "base_model": c["base_model"],
            "lora_r": c.get("lora_r", 16),
            "lora_alpha": c.get("lora_alpha", 32),
            "learning_rate": c.get("learning_rate", 2e-4),
            "num_train_epochs": c.get("num_train_epochs", 2),
            "max_steps": c.get("max_steps", -1),
            "quant": "4bit" if c.get("load_in_4bit", True) else "16bit",
            "data_mix": c.get("data_mix"),
            "sources": sources,
            "data_version": data_version,
            "n_train": n_train,
            "seed": c.get("seed", 3407),
            "gpu": c.get("gpu", 0),
            "config": c,
        }
    )
    run_id = handle["run_id"]
    run_dir = handle["run_dir"]

    print(f"==> Loading base model: {c['base_model']}")
    model, tokenizer = FastVisionModel.from_pretrained(
        c["base_model"],
        load_in_4bit=bool(c.get("load_in_4bit", True)),
        use_gradient_checkpointing="unsloth",
        max_seq_length=int(c.get("max_seq_length", 2048)),
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(c.get("finetune_vision_layers", True)),
        finetune_language_layers=bool(c.get("finetune_language_layers", True)),
        finetune_attention_modules=bool(c.get("finetune_attention_modules", True)),
        finetune_mlp_modules=bool(c.get("finetune_mlp_modules", True)),
        r=int(c.get("lora_r", 16)),
        lora_alpha=int(c.get("lora_alpha", 32)),
        lora_dropout=float(c.get("lora_dropout", 0.0)),
        bias="none",
        random_state=int(c.get("seed", 3407)),
    )
    FastVisionModel.for_training(model)

    try:
        from unsloth import is_bf16_supported

        bf16 = bool(is_bf16_supported())
    except Exception:  # noqa: BLE001
        bf16 = bool(torch.cuda.is_bf16_supported())

    sft_config = _build_sft_config(SFTConfig, c, run_dir, bf16)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=sft_config,
    )

    print("==> Training...")
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60.0

    lora_dir = run_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    registry.update_run(run_id, status="trained", train_minutes=round(train_minutes, 2))
    print(f"==> Saved adapter to {lora_dir} ({train_minutes:.1f} min)")

    num_eval = c.get("_num_eval_samples") if args.smoke else None

    if args.eval_after:
        from .eval import evaluate_run

        evaluate_run(run_id=run_id, num_samples=num_eval)

    if args.crosssite_after:
        from .build_crosssite import CROSSSITE_DIR, build
        from .datasets_registry import CROSSSITE_ADAPTERS
        from .eval import evaluate_crosssite

        if args.crosssite_name.strip().lower() == "all":
            cs_names = sorted(CROSSSITE_ADAPTERS)
        else:
            cs_names = [s.strip() for s in args.crosssite_name.split(",") if s.strip()]

        for cs_name in cs_names:
            eval_path = CROSSSITE_DIR / f"{cs_name}.jsonl"
            if not eval_path.exists():
                build(
                    name=cs_name,
                    limit=None,
                    image_max_side=int(c.get("image_max_side", 896)),
                    out_dir=CROSSSITE_DIR,
                )
            evaluate_crosssite(
                run_id=run_id,
                eval_path=str(eval_path),
                name=cs_name,
                num_samples=num_eval,
            )

    print(f"==> Done. run_id = {run_id}")


if __name__ == "__main__":
    main()
