#!/usr/bin/env python3
"""Probe VQA-Med sources: HF 2019 schema, XR filter counts, zip layouts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vqa_med_io import (  # noqa: E402
    HF_DATASET_2019,
    MANIFEST_PATH,
    URLS,
    collect_vqa_med_rows,
    infer_question_category,
    is_xr_modality_answer,
    probe_summary,
)


def main() -> None:
    print("=== VQA-Med probe ===\n")
    print(f"HF 2019 dataset: {HF_DATASET_2019}")
    print("Zip URLs:")
    for k, v in URLS.items():
        print(f"  {k}: {v}")

    print("\n=== Quick probe (2019 sample + 2020 val sample) ===")
    summary = probe_summary()
    print(json.dumps(summary, indent=2))

    print("\n=== Full collect (limit 30 per split/year, XR filter) ===")
    rows, stats = collect_vqa_med_rows(limit_per_split=30)
    print(f"merged rows: {len(rows)}")
    print(json.dumps(stats, indent=2))

    if rows:
        print("\n=== Sample rows ===")
        for r in rows[:5]:
            print(
                f"  [{r['year']}/{r['split']}] {r['question_category']}: "
                f"{r['question'][:60]} -> {r['answer'][:40]}"
            )

    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print("\nHelpers:")
    print(f"  is_xr_modality_answer('xr - plain film') = {is_xr_modality_answer('xr - plain film')}")
    print(f"  infer_question_category('what kind of image is this?') = "
          f"{infer_question_category('what kind of image is this?')}")


if __name__ == "__main__":
    main()
