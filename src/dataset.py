"""Load unified JSONL rows into the Unsloth ``messages`` format.

Images are loaded lazily (only when an example is accessed) via a HuggingFace
``Dataset`` transform, so we never hold the whole image set in memory.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .utils import abs_from_root, load_jsonl


def open_image(row: Dict) -> Image.Image:
    """Load the RGB PIL image referenced by a row."""
    return Image.open(abs_from_root(row["image"])).convert("RGB")


def to_messages(row: Dict, image: Optional[Image.Image] = None) -> Dict:
    """Convert one unified row into an Unsloth chat conversation.

    If ``image`` is None it is loaded lazily from ``row['image']``.
    """
    if image is None:
        image = open_image(row)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["target"]}],
            },
        ]
    }


def _balance_by_mix(
    rows: List[Dict],
    data_mix: Optional[Dict[str, float]],
    seed: int,
) -> List[Dict]:
    """Downsample to honor task mix weights on the full split (no max_samples cap)."""
    rng = random.Random(seed)
    if not data_mix or not rows:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        return shuffled

    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for task_rows in by_task.values():
        rng.shuffle(task_rows)

    weights = {t: data_mix.get(t, 0.0) for t in by_task}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        return shuffled

    norm = {t: w / total_weight for t, w in weights.items() if w > 0}
    max_total = min(
        len(by_task[t]) / norm[t] for t in norm if norm[t] > 0
    )
    selected: List[Dict] = []
    for task, task_rows in by_task.items():
        share = norm.get(task, 0.0)
        if share <= 0:
            continue
        take = min(len(task_rows), max(1, int(round(max_total * share))))
        selected.extend(task_rows[:take])
    rng.shuffle(selected)
    return selected


def _select_by_mix(
    rows: List[Dict],
    data_mix: Optional[Dict[str, float]],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict]:
    """Optionally subsample rows to ``max_samples`` following a task mix."""
    rng = random.Random(seed)
    if max_samples is None or max_samples >= len(rows):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        return shuffled

    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for task_rows in by_task.values():
        rng.shuffle(task_rows)

    mix = data_mix or {}
    total_weight = sum(mix.get(t, 0.0) for t in by_task) or 0.0
    selected: List[Dict] = []
    if total_weight > 0:
        for task, task_rows in by_task.items():
            share = mix.get(task, 0.0) / total_weight
            take = min(len(task_rows), int(round(max_samples * share)))
            selected.extend(task_rows[:take])
    # Top up (or fall back) with any remaining rows to reach max_samples.
    if len(selected) < max_samples:
        chosen = {id(r) for r in selected}
        leftovers = [r for r in rows if id(r) not in chosen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])
    rng.shuffle(selected)
    return selected[:max_samples]


def load_rows(processed_dir: str | Path, split: str) -> List[Dict]:
    """Read raw unified rows for a split (used by eval/infer)."""
    return load_jsonl(Path(processed_dir) / f"{split}.jsonl")


def make_sft_dataset(
    processed_dir: str | Path,
    split: str = "train",
    data_mix: Optional[Dict[str, float]] = None,
    max_samples: Optional[int] = None,
    seed: int = 3407,
    balance_tasks: bool = False,
):
    """Return a HuggingFace ``Dataset`` yielding {'messages': ...} lazily."""
    from datasets import Dataset

    rows = load_rows(processed_dir, split)
    if not rows:
        raise ValueError(f"No rows found for split '{split}' in {processed_dir}.")
    if max_samples is not None:
        rows = _select_by_mix(rows, data_mix, max_samples, seed)
    elif balance_tasks:
        rows = _balance_by_mix(rows, data_mix, seed)

    ds = Dataset.from_list(rows)

    def _transform(batch: Dict[str, list]) -> Dict[str, list]:
        messages = []
        for prompt, target, image_path in zip(
            batch["prompt"], batch["target"], batch["image"]
        ):
            image = Image.open(abs_from_root(image_path)).convert("RGB")
            messages.append(
                to_messages(
                    {"prompt": prompt, "target": target}, image=image
                )["messages"]
            )
        return {"messages": messages}

    ds.set_transform(_transform)
    return ds
