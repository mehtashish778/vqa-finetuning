"""Build a held-out CROSS-SITE (external) evaluation set from SLAKE.

This is deliberately separate from ``src.build_dataset``: the cross-site set is
used ONLY to measure generalization of already-trained runs. It is never mixed
into training, never carved into train/val/test, and never contributes to the
training ``data_version``. Instead it writes a single flat JSONL plus a
``crosssite_version`` hash used to key the (namespaced) baseline cache.

Usage
-----
    python -m src.build_crosssite                 # default set (slake_xray_en)
    python -m src.build_crosssite --name slake_xray_en
    python -m src.build_crosssite --all           # every registered cross-site set
    python -m src.build_crosssite --limit 40      # quick subset (smoke)

Each cross-site set is defined by one entry in
``src.datasets_registry.CROSSSITE_ADAPTERS``; add a loader there to make a new
set buildable here without touching this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .datasets_registry import CROSSSITE_ADAPTERS
from .utils import ROOT, load_yaml, write_jsonl

CROSSSITE_DIR = ROOT / "data" / "crosssite"


def _crosssite_version(rows: List[Dict]) -> str:
    """Stable hash over the row set (order-independent, per-QA)."""
    keys = sorted(
        f"{r['study_id']}|{r['prompt']}|{r['target']}" for r in rows
    )
    payload = "|".join(keys) + f"|n={len(keys)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]  # noqa: S324


def _stats(rows: List[Dict], crosssite_version: str, image_max_side: int) -> Dict:
    by_answer_type = Counter(r.get("answer_type") for r in rows)
    by_split_src = Counter(r.get("split_src") for r in rows)
    n_images = len({r["study_id"] for r in rows})
    return {
        "crosssite_version": crosssite_version,
        "source": sorted({str(r.get("source")) for r in rows}),
        "image_max_side": image_max_side,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": len(rows),
        "n_images": n_images,
        "by_answer_type": dict(by_answer_type),
        "by_split_src": dict(by_split_src),
    }


def build(name: str, limit: int | None, image_max_side: int, out_dir: Path) -> Dict:
    if name not in CROSSSITE_ADAPTERS:
        raise KeyError(
            f"Unknown cross-site set '{name}'. Registered: "
            f"{sorted(CROSSSITE_ADAPTERS)}"
        )
    loader = CROSSSITE_ADAPTERS[name]
    print(f"==> Importing cross-site set '{name}' via {loader.__name__}")
    rows = loader(image_max_side=image_max_side, limit=limit)
    if not rows:
        raise RuntimeError(f"Cross-site build '{name}' produced 0 rows; check filters.")
    print(f"    {len(rows)} rows over {len({r['study_id'] for r in rows})} images")

    crosssite_version = _crosssite_version(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{name}.jsonl"
    n = write_jsonl(jsonl_path, rows)
    print(f"==> Wrote {jsonl_path} ({n} rows)")

    stats = _stats(rows, crosssite_version, image_max_side)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"==> crosssite_version = {crosssite_version}")
    print(f"==> Wrote {out_dir / 'stats.json'}")
    return stats


def parse_args() -> argparse.Namespace:
    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    p = argparse.ArgumentParser(description="Build a cross-site eval set.")
    p.add_argument(
        "--name",
        default="slake_xray_en",
        choices=sorted(CROSSSITE_ADAPTERS),
        help="which registered cross-site set to build",
    )
    p.add_argument(
        "--all", action="store_true", help="build every registered cross-site set"
    )
    p.add_argument("--limit", type=int, default=None, help="max rows per split")
    p.add_argument(
        "--image-max-side", type=int, default=int(cfg.get("image_max_side", 896))
    )
    p.add_argument("--out-dir", type=Path, default=CROSSSITE_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = sorted(CROSSSITE_ADAPTERS) if args.all else [args.name]
    for name in names:
        build(
            name=name,
            limit=args.limit,
            image_max_side=args.image_max_side,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
