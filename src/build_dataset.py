"""Build the unified train/val/test JSONL from open dataset sources.

Split policy
------------
- Honor each source's OFFICIAL split (train/val/test) when present.
- For a source that has no validation split, carve ``--val-frac`` of its TRAIN
  studies into ``val`` with a seeded, study-level split (no image leakage).
- The TEST split is treated as frozen; a ``data_version`` hash of the test set
  is written to ``stats.json`` so every model is benchmarked identically.

Usage
-----
    python -m src.build_dataset --sources vqa_rad iu_xray
    python -m src.build_dataset --limit 300           # quick subset per source
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .datasets_registry import ADAPTERS
from .utils import ROOT, load_yaml, write_jsonl

SPLITS = ("train", "val", "test")


def _ensure_splits(
    rows: List[Dict], val_frac: float, test_frac: float, seed: int
) -> None:
    """Ensure every source has train/val/test; carve missing splits from train.

    Mutates ``rows`` in place. Carving is by ``study_id`` (no image leakage),
    seeded per source, and val/test carves are disjoint.
    """
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    for source, srows in by_source.items():
        present = {r["split"] for r in srows}
        train_studies = sorted({r["study_id"] for r in srows if r["split"] == "train"})
        if not train_studies:
            continue
        rng = random.Random(f"{seed}:{source}")
        rng.shuffle(train_studies)

        moves: Dict[str, str] = {}
        cursor = 0
        if "test" not in present and test_frac > 0:
            n_test = max(1, int(len(train_studies) * test_frac))
            for s in train_studies[cursor : cursor + n_test]:
                moves[s] = "test"
            cursor += n_test
        if "val" not in present and val_frac > 0:
            n_val = max(1, int(len(train_studies) * val_frac))
            for s in train_studies[cursor : cursor + n_val]:
                moves[s] = "val"
            cursor += n_val

        for r in srows:
            if r["split"] == "train" and r["study_id"] in moves:
                r["split"] = moves[r["study_id"]]


def _compute_data_version(rows: List[Dict]) -> str:
    keys = sorted(
        f"{r['source']}:{r['study_id']}" for r in rows if r["split"] == "test"
    )
    payload = "|".join(keys) + f"|n={len(keys)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]  # noqa: S324


def _stats(rows: List[Dict], data_version: str, image_max_side: int) -> Dict:
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for r in rows:
        counts[r["source"]][r["task"]][r["split"]] += 1
    # Convert nested defaultdicts to plain dicts for JSON.
    counts_plain = {
        src: {task: dict(splits) for task, splits in tasks.items()}
        for src, tasks in counts.items()
    }
    totals = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    return {
        "data_version": data_version,
        "image_max_side": image_max_side,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals_by_split": totals,
        "counts_by_source": counts_plain,
        "n_rows": len(rows),
    }


def build(
    sources: List[str],
    access: str,
    limit: int | None,
    val_frac: float,
    test_frac: float,
    seed: int,
    image_max_side: int,
    out_dir: Path,
) -> Dict:
    all_rows: List[Dict] = []
    for source in sources:
        if source not in ADAPTERS:
            raise ValueError(f"Unknown source '{source}'. Known: {list(ADAPTERS)}")
        print(f"==> Importing source: {source}")
        rows = ADAPTERS[source](image_max_side=image_max_side, limit=limit)
        print(f"    {len(rows)} rows")
        all_rows.extend(rows)

    if access != "all":
        all_rows = [r for r in all_rows if r.get("access") == access]

    _ensure_splits(all_rows, val_frac=val_frac, test_frac=test_frac, seed=seed)

    data_version = _compute_data_version(all_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_rows = [r for r in all_rows if r["split"] == split]
        n = write_jsonl(out_dir / f"{split}.jsonl", split_rows)
        print(f"==> Wrote {out_dir / f'{split}.jsonl'} ({n} rows)")

    stats = _stats(all_rows, data_version, image_max_side)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"==> data_version = {data_version}")
    print(f"==> Wrote {out_dir / 'stats.json'}")
    return stats


def parse_args() -> argparse.Namespace:
    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    p = argparse.ArgumentParser(description="Build unified VQA + report dataset.")
    p.add_argument("--sources", nargs="+", default=list(ADAPTERS.keys()))
    p.add_argument("--access", default="open", choices=["open", "closed", "all"])
    p.add_argument("--limit", type=int, default=None, help="max rows per source")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 3407)))
    p.add_argument(
        "--image-max-side", type=int, default=int(cfg.get("image_max_side", 896))
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / str(cfg.get("processed_dir", "data/processed")),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(
        sources=args.sources,
        access=args.access,
        limit=args.limit,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        image_max_side=args.image_max_side,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
