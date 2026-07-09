"""Pluggable dataset source adapters.

Each adapter downloads an open dataset from the HuggingFace Hub, persists its
images to ``data/raw/<source>/images/`` (downscaled to cap Qwen3-VL visual
tokens), and yields rows in the unified storage schema:

    {
      "task": "vqa" | "report",
      "source": "vqa_rad" | "iu_xray" | ...,
      "access": "open",
      "image": "data/raw/.../xxx.png",   # repo-relative POSIX path
      "study_id": "...",                  # split unit (no leakage)
      "prompt": "[TASK: ...] ...",
      "target": "...",
      "split": "train" | "val" | "test",  # official split when available
      ... task-specific extras ...
    }

Adding a new source (e.g. MIMIC) = add one function here + register it in
``ADAPTERS``; nothing else in the pipeline changes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from .utils import ROOT, rel_to_root, resize_max_side

RAW_DIR = ROOT / "data" / "raw"

VQA_PROMPT = "[TASK: VQA] Question: {q} Answer briefly."
REPORT_PROMPT = "[TASK: REPORT] Describe the findings in this chest X-ray."

_YES_NO = {"yes", "no"}


def _classify_answer_type(answer: str) -> str:
    return "closed" if answer.strip().lower() in _YES_NO else "open"


def _to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def _signature(img: Image.Image) -> str:
    """Content hash of an RGB image (used to dedupe repeated images)."""
    return hashlib.md5(img.tobytes()).hexdigest()  # noqa: S324 (non-crypto use)


def _save_image(img: Image.Image, out_path: Path, max_side: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = resize_max_side(_to_rgb(img), max_side)
    img.save(out_path, format="PNG")


# ---------------------------------------------------------------------------
# VQA-RAD  (flaviagiammarino/vqa-rad)
# ---------------------------------------------------------------------------
def load_vqa_rad(image_max_side: int = 896, limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("flaviagiammarino/vqa-rad")
    img_dir = RAW_DIR / "vqa_rad" / "images"
    sig_to_study: Dict[str, str] = {}
    rows: List[Dict] = []

    # VQA-RAD ships train/test only (no val); build_dataset carves val later.
    for hf_split, split in (("train", "train"), ("test", "test")):
        if hf_split not in ds:
            continue
        count = 0
        for ex in ds[hf_split]:
            if limit is not None and count >= limit:
                break
            image = _to_rgb(ex["image"])
            sig = _signature(image)
            if sig not in sig_to_study:
                study_id = f"vqa_rad_{sig[:12]}"
                sig_to_study[sig] = study_id
                _save_image(image, img_dir / f"{study_id}.png", image_max_side)
            study_id = sig_to_study[sig]
            answer = str(ex["answer"]).strip()
            rows.append(
                {
                    "task": "vqa",
                    "source": "vqa_rad",
                    "access": "open",
                    "image": rel_to_root(img_dir / f"{study_id}.png"),
                    "study_id": study_id,
                    "prompt": VQA_PROMPT.format(q=str(ex["question"]).strip()),
                    "target": answer,
                    "answer_type": _classify_answer_type(answer),
                    "split": split,
                }
            )
            count += 1
    return rows


# ---------------------------------------------------------------------------
# IU X-Ray  (dz-osamu/IU-Xray, fallback Shrey-1329/cxiu_hf_dataset)
# ---------------------------------------------------------------------------
def _coerce_image(elem, snapshot_dir: Optional[Path]) -> Optional[Image.Image]:
    """Turn an IU-Xray 'images' element into a PIL image, however it's stored."""
    if isinstance(elem, Image.Image):
        return elem
    if isinstance(elem, dict):
        if elem.get("bytes"):
            import io

            return Image.open(io.BytesIO(elem["bytes"]))
        if elem.get("path"):
            elem = elem["path"]
        else:
            return None
    if isinstance(elem, str):
        candidates = [Path(elem)]
        if snapshot_dir is not None:
            candidates.append(snapshot_dir / elem.lstrip("/"))
        for c in candidates:
            if c.exists():
                return Image.open(c)
    return None


def _iu_zip_index(snapshot_dir: Path):
    """Index the images inside dz-osamu/IU-Xray's image.zip by '<study>/<file>'.

    The zip stores images under a long, machine-specific prefix while the
    dataset rows reference '/iu_xray/image/<study>/<n>.png'; the last two path
    components are the stable join key.
    """
    import zipfile

    zip_path = snapshot_dir / "image.zip"
    if not zip_path.exists():
        return None, {}
    zf = zipfile.ZipFile(zip_path)
    index: Dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        parts = name.rstrip("/").split("/")
        if len(parts) >= 2:
            index.setdefault(f"{parts[-2]}/{parts[-1]}", name)
    return zf, index


def _load_iu_primary(image_max_side: int, limit: Optional[int]) -> List[Dict]:
    import io

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    ds = load_dataset("dz-osamu/IU-Xray")
    snapshot_dir = Path(
        snapshot_download(repo_id="dz-osamu/IU-Xray", repo_type="dataset")
    )
    zf, index = _iu_zip_index(snapshot_dir)
    if zf is None:
        raise FileNotFoundError("dz-osamu/IU-Xray image.zip not found in snapshot.")

    img_dir = RAW_DIR / "iu_xray" / "images"
    rows: List[Dict] = []
    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}

    for hf_split in ds.keys():
        split = split_map.get(hf_split, hf_split)
        count = 0
        for idx, ex in enumerate(ds[hf_split]):
            if limit is not None and count >= limit:
                break
            images = ex.get("images") or []
            if not images:
                continue
            parts = [p for p in str(images[0]).split("/") if p]
            if len(parts) < 2:
                continue
            study, key = parts[-2], f"{parts[-2]}/{parts[-1]}"
            zip_name = index.get(key)
            if zip_name is None:
                continue  # missing image; empty result -> caller falls back
            with zf.open(zip_name) as fh:
                img = Image.open(io.BytesIO(fh.read()))
                img.load()
            study_id = f"iu_{study}"
            out_name = f"{study_id}_{hf_split}_{idx}.png"
            _save_image(img, img_dir / out_name, image_max_side)
            rows.append(
                {
                    "task": "report",
                    "source": "iu_xray",
                    "access": "open",
                    "image": rel_to_root(img_dir / out_name),
                    "study_id": study_id,
                    "prompt": REPORT_PROMPT,
                    "target": str(ex.get("response", "")).strip(),
                    "view": "frontal",
                    "split": split,
                }
            )
            count += 1
    return rows


def _load_iu_fallback(image_max_side: int, limit: Optional[int]) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("Shrey-1329/cxiu_hf_dataset")
    img_dir = RAW_DIR / "iu_xray" / "images"
    rows: List[Dict] = []
    for hf_split in ds.keys():
        count = 0
        for idx, ex in enumerate(ds[hf_split]):
            if limit is not None and count >= limit:
                break
            img = _coerce_image(ex["image"], None)
            if img is None:
                continue
            study_id = f"iu_{hf_split}_{idx}"
            out_name = f"{study_id}.png"
            _save_image(img, img_dir / out_name, image_max_side)
            rows.append(
                {
                    "task": "report",
                    "source": "iu_xray",
                    "access": "open",
                    "image": rel_to_root(img_dir / out_name),
                    "study_id": study_id,
                    "prompt": REPORT_PROMPT,
                    "target": str(ex.get("text", "")).strip(),
                    "view": "frontal",
                    # Single split upstream; build_dataset will split by study_id.
                    "split": "train",
                }
            )
            count += 1
    return rows


def load_iu_xray(image_max_side: int = 896, limit: Optional[int] = None) -> List[Dict]:
    """Load IU X-Ray, falling back to a mirror with embedded images if needed."""
    try:
        rows = _load_iu_primary(image_max_side, limit)
        if rows:
            return rows
        raise FileNotFoundError("dz-osamu/IU-Xray returned no rows.")
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        print(f"[iu_xray] primary source failed ({exc}); using fallback mirror.")
        return _load_iu_fallback(image_max_side, limit)


# ---------------------------------------------------------------------------
# SLAKE  (Keetawan/SLAKE) -- EVAL-ONLY cross-site set, NOT for training.
# ---------------------------------------------------------------------------
def load_slake(image_max_side: int = 896, limit: Optional[int] = None) -> List[Dict]:
    """Load SLAKE English chest X-ray VQA for cross-site (external) evaluation.

    Kept OUT of ``ADAPTERS`` on purpose: this is a held-out generalization set
    and must never enter the training build. Consumed only by
    ``src.build_crosssite``. Filters to English (``q_lang == 'en'``) chest
    X-ray (``modality == 'X-Ray'``); since SLAKE's only X-rays are the 179
    chest films from ChestX-ray8, this excludes all CT/MRI and non-chest.
    """
    from datasets import load_dataset

    ds = load_dataset("Keetawan/SLAKE")  # train/validation/test, both languages
    img_dir = RAW_DIR / "slake" / "images"
    saved: set = set()
    rows: List[Dict] = []
    for split in ds.keys():
        count = 0
        for ex in ds[split]:
            if limit is not None and count >= limit:
                break
            if str(ex.get("q_lang")) != "en":
                continue
            if str(ex.get("modality", "")).lower().replace("-", "") != "xray":
                continue
            study = str(ex["img_name"]).split("/")[0]  # e.g. 'xmlab120'
            study_id = f"slake_{study}"
            out_name = f"{study_id}.png"
            if study_id not in saved:
                _save_image(_to_rgb(ex["image"]), img_dir / out_name, image_max_side)
                saved.add(study_id)
            answer = str(ex["answer"]).strip()
            atype = "closed" if str(ex.get("answer_type", "")).upper() == "CLOSED" else "open"
            rows.append(
                {
                    "task": "vqa",
                    "source": "slake",
                    "access": "open",
                    "image": rel_to_root(img_dir / out_name),
                    "study_id": study_id,
                    "prompt": VQA_PROMPT.format(q=str(ex["question"]).strip()),
                    "target": answer,
                    "answer_type": atype,
                    "content_type": ex.get("content_type"),
                    "split_src": split,
                }
            )
            count += 1
    return rows


ADAPTERS: Dict[str, Callable[..., List[Dict]]] = {
    "vqa_rad": load_vqa_rad,
    "iu_xray": load_iu_xray,
}

# Cross-site (external) EVAL-ONLY datasets. Kept separate from ADAPTERS so they
# can never leak into the training build. Add a new held-out set by writing a
# loader with signature ``(image_max_side, limit) -> List[Dict]`` and adding one
# entry here; ``src.build_crosssite`` and ``--crosssite-after`` pick it up
# automatically.
CROSSSITE_ADAPTERS: Dict[str, Callable[..., List[Dict]]] = {
    "slake_xray_en": load_slake,
}
