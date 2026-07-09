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

# In-domain columns. Cross-site columns are inserted dynamically (one per
# registered/observed cross-site set) between these and the tail columns.
HEAD_COLUMNS = [
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
]
TAIL_COLUMNS = [
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

    # Cross-site (external) generalization: one pair of columns per set.
    for name, entry in (row.get("crosssite") or {}).items():
        cs_metrics = (entry or {}).get("metrics") or {}
        cs_baseline = (entry or {}).get("baseline") or {}
        if "vqa_acc" in cs_metrics:
            flat[f"cs_{name}_acc"] = cs_metrics["vqa_acc"]
            if "vqa_acc" in cs_baseline:
                flat[f"cs_{name}_d"] = round(
                    cs_metrics["vqa_acc"] - cs_baseline["vqa_acc"], 4
                )
    return flat


def _crosssite_columns(rows: list) -> list:
    """Discover cross-site set names across all runs and emit acc/delta columns."""
    names: list = []
    for r in rows:
        for name in (r.get("crosssite") or {}):
            if name not in names:
                names.append(name)
    cols = []
    for name in sorted(names):
        cols.append((f"cs_{name}_acc", f"cs:{name}"))
        cols.append((f"cs_{name}_d", f"csd:{name}"))
    return cols


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ",".join(f"{k}:{val}" for k, val in v.items())
    return str(v)


def build_table(sort_key: str):
    raw = load_registry()
    columns = HEAD_COLUMNS + _crosssite_columns(raw) + TAIL_COLUMNS
    rows = [_flatten(r) for r in raw]
    rows.sort(key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))
    header = [label for _, label in columns]
    body = [[_fmt(r.get(field)) for field, _ in columns] for r in rows]
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
