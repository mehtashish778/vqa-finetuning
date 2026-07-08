#!/usr/bin/env python3
"""Aggregate all runs in outputs/registry.jsonl into a comparison table.

Prints the leaderboard and regenerates outputs/leaderboard.md.

Usage:
    python scripts/leaderboard.py                 # sort by vqa_acc (default)
    python scripts/leaderboard.py --sort report_bleu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import REGISTRY_PATH, load_registry  # noqa: E402

COLUMNS = [
    ("run_id", "run_id"),
    ("base_model", "base_model"),
    ("lora_r", "r"),
    ("lora_alpha", "alpha"),
    ("lr", "lr"),
    ("epochs", "epochs"),
    ("n_train", "n_train"),
    ("n_eval", "n_eval"),
    ("train_minutes", "min"),
    ("vqa_acc", "vqa_acc"),
    ("vqa_f1", "vqa_f1"),
    ("report_bleu", "bleu"),
    ("report_rougeL", "rougeL"),
    ("d_vqa_acc", "d_vqa_acc"),
    ("data_version", "data_ver"),
    ("status", "status"),
]


def _short_model(name: str) -> str:
    return str(name).rstrip("/").split("/")[-1] if name else ""


def _flatten(row: dict) -> dict:
    metrics = row.get("metrics") or {}
    baseline = row.get("baseline") or {}
    flat = dict(row)
    for k, v in metrics.items():
        flat[k] = v
    flat["base_model"] = _short_model(row.get("base_model", ""))
    if "vqa_acc" in metrics and "vqa_acc" in baseline:
        flat["d_vqa_acc"] = round(metrics["vqa_acc"] - baseline["vqa_acc"], 4)
    return flat


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ",".join(f"{k}:{val}" for k, val in v.items())
    return str(v)


def build_table(sort_key: str):
    rows = [_flatten(r) for r in load_registry()]
    rows.sort(key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))
    header = [label for _, label in COLUMNS]
    body = [[_fmt(r.get(field)) for field, _ in COLUMNS] for r in rows]
    return header, body


def to_markdown(header, body) -> str:
    lines = ["# Model Leaderboard", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def to_text(header, body) -> str:
    widths = [len(h) for h in header]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    out = [fmt.format(*header)]
    out.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in body:
        out.append(fmt.format(*row))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Render the run leaderboard.")
    p.add_argument("--sort", default="vqa_acc", help="metric/field to sort by (desc)")
    args = p.parse_args()

    rows = load_registry()
    if not rows:
        print(f"No runs found in {REGISTRY_PATH}. Train a model first.")
        return

    header, body = build_table(args.sort)
    print(to_text(header, body))

    md_path = ROOT / "outputs" / "leaderboard.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(to_markdown(header, body), encoding="utf-8")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
