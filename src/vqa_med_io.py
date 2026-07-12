"""Download, parse, and filter VQA-Med (ImageCLEF 2019–2021) for training.

Chest X-ray / plain-film filter only. Used by ``load_vqa_med`` in
``datasets_registry``; probe script imports helpers from here.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

from PIL import Image

from .utils import ROOT

RAW_VQA_MED = ROOT / "data" / "raw" / "vqa_med"
DOWNLOADS_DIR = RAW_VQA_MED / "downloads"
MANIFEST_PATH = RAW_VQA_MED / "manifest.json"

HF_DATASET_2019 = "claudioreeves/imageclef-vqa-med-2019"

URLS = {
    "2020_val": "https://github.com/abachaa/VQA-Med-2020/raw/main/VQA-ValidationSet-VQAMed2020-Task1.zip",
    "2020_test": "https://github.com/abachaa/VQA-Med-2020/raw/main/VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1.zip",
    "2021_val": "https://github.com/abachaa/VQA-Med-2021/raw/main/VQA-Med-2021-Tasks-1-2-NewValidationSets.zip",
    "2021_test": "https://github.com/abachaa/VQA-Med-2021/raw/main/Task1-VQA-2021-TestSet-w-GroundTruth.zip",
}

SPLIT_RANK = {"train": 0, "val": 1, "test": 2}

XR_ANSWER_HINTS = (
    "xr - plain film",
    "plain film",
    "radiograph",
    "x-ray",
    "xray",
)
XR_TEXT_HINTS = (
    "x-ray",
    "xray",
    "cxr",
    "chest",
    "radiograph",
    "plain film",
)
NON_XR_HINTS = (
    " ct",
    "ct ",
    "ct scan",
    "cta ",
    " mri",
    "mri ",
    "ultrasound",
    "mammograph",
    "mammo",
    "nuclear medicine",
    "fluoroscopy",
    "angiogram",
    "pet ",
    " tomography",
)

LAST_FILTER_STATS: Dict = {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _signature(img: Image.Image) -> str:
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    return hashlib.md5(rgb.tobytes()).hexdigest()  # noqa: S324


def infer_question_category(question: str) -> str:
    q = _norm(question)
    if re.search(r"what (type|kind) of image|imaging modality|modality", q):
        return "modality"
    if "plane" in q:
        return "plane"
    if re.search(r"which organ|organ system|organ principally", q):
        return "organ"
    if re.search(r"abnormal|alarming|what is wrong|primary abnormality", q):
        return "abnormality"
    return "unknown"


def is_xr_modality_answer(answer: str) -> bool:
    a = _norm(answer)
    if any(h in a for h in NON_XR_HINTS):
        return False
    return any(h in a for h in XR_ANSWER_HINTS) or a == "xr" or a.startswith("xr ")


def classify_zip_xr(
    question: str,
    answer: str,
    image_id: str,
    xr_image_ids: Optional[set] = None,
) -> str:
    """Return keep | drop_non_xr | drop_unknown."""
    if xr_image_ids and image_id in xr_image_ids:
        return "keep"
    text = _norm(f"{question} {answer}")
    if any(h in text for h in NON_XR_HINTS):
        return "drop_non_xr"
    if any(h in text for h in XR_TEXT_HINTS):
        return "keep"
    if answer and is_xr_modality_answer(answer):
        return "keep"
    return "drop_unknown"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"    downloading {dest.name} ...")
        urlretrieve(url, dest)  # noqa: S310
    return dest


def _read_zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _parse_pipe_qa(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _find_member(zf: zipfile.ZipFile, *needles: str) -> Optional[str]:
    for name in zf.namelist():
        if "__MACOSX" in name or name.endswith("/"):
            continue
        low = name.lower()
        if all(n.lower() in low for n in needles):
            return name
    return None


def _image_from_zip(zf: zipfile.ZipFile, image_id: str) -> Optional[Image.Image]:
    candidates = [
        n
        for n in zf.namelist()
        if image_id in n and not n.endswith("/") and "__MACOSX" not in n
    ]
    for name in candidates:
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            with zf.open(name) as fh:
                img = Image.open(io.BytesIO(fh.read()))
                img.load()
                return img
    return None


def _load_2019_hf(
    limit_per_split: Optional[int],
    filter_stats: Dict,
) -> Tuple[List[Dict], set]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_2019)
    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}

    # Group by image content hash.
    groups: Dict[str, Dict] = {}
    for hf_split in ds.keys():
        split = split_map.get(hf_split, hf_split)
        count = 0
        for ex in ds[hf_split]:
            if limit_per_split is not None and count >= limit_per_split:
                break
            img = ex["image"]
            if not isinstance(img, Image.Image):
                continue
            sig = _signature(img)
            grp = groups.setdefault(
                sig,
                {"image": img, "qas": [], "split": split, "year": "2019"},
            )
            # Prefer highest split rank if same image appears in multiple splits.
            if SPLIT_RANK.get(split, 0) > SPLIT_RANK.get(grp["split"], 0):
                grp["split"] = split
            q = str(ex["question"]).strip()
            a = str(ex["answer"]).strip()
            cat = infer_question_category(q)
            grp["qas"].append({"question": q, "answer": a, "category": cat})
            count += 1

    xr_ids: set = set()
    rows: List[Dict] = []
    kept_images = 0
    dropped_images = 0
    for sig, grp in groups.items():
        modality_answers = [
            qa["answer"]
            for qa in grp["qas"]
            if qa["category"] == "modality"
        ]
        if not modality_answers:
            # Fallback: any modality-like question
            modality_answers = [
                qa["answer"]
                for qa in grp["qas"]
                if "modality" in _norm(qa["question"]) or "kind of image" in _norm(qa["question"])
            ]
        if not modality_answers or not any(is_xr_modality_answer(a) for a in modality_answers):
            dropped_images += 1
            continue
        kept_images += 1
        image_id = f"hf2019_{sig[:12]}"
        xr_ids.add(image_id)
        for qa in grp["qas"]:
            rows.append(
                {
                    "image": grp["image"],
                    "image_id": image_id,
                    "content_sig": sig,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "split": grp["split"],
                    "year": "2019",
                    "question_category": qa["category"],
                }
            )

    filter_stats["2019"] = {
        "images_total": len(groups),
        "images_kept": kept_images,
        "images_dropped": dropped_images,
        "qa_rows_kept": len(rows),
    }
    return rows, xr_ids


def _find_qa_file(zf: zipfile.ZipFile, split: str) -> Optional[str]:
    for name in zf.namelist():
        low = name.lower()
        if "__macosx" in low or not low.endswith(".txt"):
            continue
        if "vqg" in low or "generation" in low:
            continue
        base = Path(name).name.lower()
        if "imageid" in base.replace("_", ""):
            continue
        if "question" in base and "answer" not in base and "pair" not in base:
            # standalone questions file handled in test branch
            continue
        if "qa" not in base and "pair" not in base and "validationset" not in base:
            continue
        if split == "val" and "val" not in base:
            continue
        if split == "train" and "train" not in base:
            continue
        if split == "test" and "test" not in base:
            continue
        return name
    return None


def _load_zip_vqa_year_split(
    year: str,
    split: str,
    zip_path: Path,
    limit: Optional[int],
    xr_image_ids: set,
    filter_stats: Dict,
) -> List[Dict]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    zf = _read_zip(zip_path)
    rows: List[Dict] = []
    counters = Counter()

    if split == "test":
        q_file = _find_member(zf, "questions")
        a_file = _find_member(zf, "referenceanswers")
        if not q_file or not a_file:
            raise FileNotFoundError(f"{year} test Q/A files not found in {zip_path}")
        questions: Dict[str, str] = {}
        with zf.open(q_file) as fh:
            for line in fh:
                parsed = _parse_pipe_qa(line.decode("utf-8", errors="replace"))
                if parsed:
                    questions[parsed[0]] = parsed[1]
                else:
                    raw = line.decode("utf-8", errors="replace").strip()
                    if "|" in raw:
                        i, q = raw.split("|", 1)
                        questions[i.strip()] = q.strip()
        answers: Dict[str, str] = {}
        with zf.open(a_file) as fh:
            for line in fh:
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                parts = raw.split("|")
                if len(parts) >= 2:
                    answers[parts[0].strip()] = parts[1].strip()

        inner_name = _find_member(zf, "images.zip") or _find_member(zf, "test-images.zip")
        inner_zf = zf
        if inner_name:
            with zf.open(inner_name) as outer:
                inner_zf = zipfile.ZipFile(io.BytesIO(outer.read()))

        count = 0
        for image_id, question in questions.items():
            if limit is not None and count >= limit:
                break
            answer = answers.get(image_id, "")
            verdict = classify_zip_xr(question, answer, image_id, xr_image_ids)
            counters[verdict] += 1
            if verdict != "keep":
                continue
            img = _image_from_zip(inner_zf, image_id)
            if img is None:
                counters["missing_image"] += 1
                continue
            rows.append(
                {
                    "image": img,
                    "image_id": image_id,
                    "content_sig": _signature(img),
                    "question": question,
                    "answer": answer,
                    "split": "test",
                    "year": year,
                    "question_category": infer_question_category(question),
                }
            )
            count += 1
    else:
        qa_file = _find_qa_file(zf, split)
        if qa_file is None:
            raise FileNotFoundError(f"No QA file for {year} {split} in {zip_path}")

        count = 0
        with zf.open(qa_file) as fh:
            for line in fh:
                if limit is not None and count >= limit:
                    break
                parsed = _parse_pipe_qa(line.decode("utf-8", errors="replace"))
                if not parsed:
                    continue
                image_id, question, answer = parsed
                verdict = classify_zip_xr(question, answer, image_id, xr_image_ids)
                counters[verdict] += 1
                if verdict != "keep":
                    continue
                img = _image_from_zip(zf, image_id)
                if img is None:
                    counters["missing_image"] += 1
                    continue
                rows.append(
                    {
                        "image": img,
                        "image_id": image_id,
                        "content_sig": _signature(img),
                        "question": question,
                        "answer": answer,
                        "split": split,
                        "year": year,
                        "question_category": infer_question_category(question),
                    }
                )
                count += 1

    filter_stats[f"{year}_{split}"] = dict(counters)
    filter_stats[f"{year}_{split}"]["qa_rows_kept"] = len(rows)
    return rows


def _find_local_2020_train_zip() -> Optional[Path]:
    """Manual AIcrowd drop-in: any *Train*.zip under downloads/."""
    if not DOWNLOADS_DIR.exists():
        return None
    for path in sorted(DOWNLOADS_DIR.glob("*Train*.zip")):
        if path.is_file() and path.stat().st_size > 0:
            return path
    for path in sorted(DOWNLOADS_DIR.glob("*train*.zip")):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _load_2020_train(
    limit: Optional[int],
    xr_image_ids: set,
    filter_stats: Dict,
) -> List[Dict]:
    if os.environ.get("VQA_MED_SKIP_2020_TRAIN", "").strip() in ("1", "true", "yes"):
        filter_stats["2020_train"] = {"skipped": True}
        return []
    zip_path = _find_local_2020_train_zip()
    if zip_path is None:
        filter_stats["2020_train"] = {
            "skipped": True,
            "reason": "no local train zip; download from AIcrowd and place under data/raw/vqa_med/downloads/",
        }
        return []
    return _load_zip_vqa_year_split(
        "2020", "train", zip_path, limit, xr_image_ids, filter_stats
    )


def _normalize_question(q: str) -> str:
    return _norm(q)


def merge_and_dedup(rows: List[Dict]) -> List[Dict]:
    best: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        key = (row["content_sig"], _normalize_question(row["question"]))
        existing = best.get(key)
        if existing is None:
            best[key] = row
            continue
        if SPLIT_RANK.get(row["split"], 0) > SPLIT_RANK.get(existing["split"], 0):
            best[key] = row
    return list(best.values())


def collect_vqa_med_rows(
    limit_per_split: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """Load and filter all VQA-Med years. Returns raw row dicts + filter stats."""
    filter_stats: Dict = {"modality_filter": "chest_xray"}
    all_rows: List[Dict] = []

    rows_2019, xr_ids = _load_2019_hf(limit_per_split, filter_stats)
    all_rows.extend(rows_2019)
    # Also track synpic-style IDs from 2019 if any (usually none on HF).

    for year, split, key in [
        ("2020", "val", "2020_val"),
        ("2020", "test", "2020_test"),
        ("2021", "val", "2021_val"),
        ("2021", "test", "2021_test"),
    ]:
        try:
            zip_path = _download(URLS[key], DOWNLOADS_DIR / f"{key}.zip")
            chunk = _load_zip_vqa_year_split(
                year, split, zip_path, limit_per_split, xr_ids, filter_stats
            )
            all_rows.extend(chunk)
        except Exception as exc:  # noqa: BLE001
            filter_stats[f"{year}_{split}"] = {"error": str(exc)}
            print(f"[vqa_med] warning: failed {year} {split}: {exc}")

    try:
        all_rows.extend(_load_2020_train(limit_per_split, xr_ids, filter_stats))
    except Exception as exc:  # noqa: BLE001
        filter_stats["2020_train"] = {"error": str(exc)}
        print(f"[vqa_med] warning: 2020 train failed: {exc}")

    merged = merge_and_dedup(all_rows)
    by_split = Counter(r["split"] for r in merged)
    filter_stats["merged"] = {
        "n_rows": len(merged),
        "n_images": len({r["content_sig"] for r in merged}),
        "by_split": dict(by_split),
        "by_year": dict(Counter(r["year"] for r in merged)),
    }

    LAST_FILTER_STATS.clear()
    LAST_FILTER_STATS.update(filter_stats)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(filter_stats, f, indent=2)
    return merged, filter_stats


def probe_summary() -> Dict:
    """Quick probe: HF 2019 + one zip; no full image save."""
    stats: Dict = {}
    rows_2019, xr_ids = _load_2019_hf(limit_per_split=200, filter_stats=stats)
    stats["2019_sample_rows"] = len(rows_2019)
    stats["xr_ids_from_2019"] = len(xr_ids)
    try:
        zip_path = _download(URLS["2020_val"], DOWNLOADS_DIR / "2020_val.zip")
        z = _load_zip_vqa_year_split(
            "2020", "val", zip_path, 200, xr_ids, stats
        )
        stats["2020_val_sample_rows"] = len(z)
    except Exception as exc:  # noqa: BLE001
        stats["2020_val_error"] = str(exc)
    return stats
