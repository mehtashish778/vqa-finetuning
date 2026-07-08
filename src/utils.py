"""Shared helpers: repo paths, config loading, JSONL and image utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


def rel_to_root(path: Path | str) -> str:
    """Return a POSIX, repo-relative path string (portable across machines)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def abs_from_root(path: str | Path) -> Path:
    """Resolve a repo-relative path back to an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def resize_max_side(img, max_side: int):
    """Downscale a PIL image so its longest side is <= max_side (keeps aspect)."""
    w, h = img.size
    longest = max(w, h)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img
