"""Per-task evaluation on the frozen test split + registry logging.

VQA (primary):
    - vqa_acc: exact-match accuracy over all VQA rows (normalized)
    - vqa_f1 : mean token-level F1 over open-ended VQA answers
Report (secondary):
    - report_bleu   : corpus BLEU (sacrebleu)
    - report_rougeL : mean ROUGE-L F-measure

The frozen base model is scored as a baseline (cached per base_model +
data_version for full-test runs) so every finetuned run reports a delta.

Examples
--------
    python -m src.eval --run-id <id>              # full frozen test set
    python -m src.eval --run-id <id> --num-samples 40   # quick subset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .dataset import load_rows, open_image
from .registry import RUNS_DIR
from .utils import ROOT, load_jsonl, load_yaml, rel_to_root

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(_PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(pred: str, gold: str) -> float:
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for t in p_tokens:
        if t in g_tokens:
            common[t] = min(p_tokens.count(t), g_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_metrics(preds: List[Dict]) -> Dict[str, float]:
    """preds: list of {task, answer_type, pred, gold, source?}."""
    metrics: Dict[str, float] = {}

    vqa = [p for p in preds if p["task"] == "vqa"]
    if vqa:
        correct = sum(
            1 for p in vqa if normalize_answer(p["pred"]) == normalize_answer(p["gold"])
        )
        metrics["vqa_acc"] = round(correct / len(vqa), 4)
        open_rows = [p for p in vqa if p.get("answer_type") == "open"] or vqa
        metrics["vqa_f1"] = round(
            sum(_token_f1(p["pred"], p["gold"]) for p in open_rows) / len(open_rows), 4
        )
        by_source: Dict[str, List[Dict]] = defaultdict(list)
        for p in vqa:
            by_source[p.get("source") or "unknown"].append(p)
        acc_by_src: Dict[str, float] = {}
        f1_by_src: Dict[str, float] = {}
        for src, src_rows in sorted(by_source.items()):
            acc_by_src[src] = round(
                sum(
                    1
                    for p in src_rows
                    if normalize_answer(p["pred"]) == normalize_answer(p["gold"])
                )
                / len(src_rows),
                4,
            )
            open_src = [p for p in src_rows if p.get("answer_type") == "open"] or src_rows
            f1_by_src[src] = round(
                sum(_token_f1(p["pred"], p["gold"]) for p in open_src) / len(open_src),
                4,
            )
        metrics["vqa_acc_by_source"] = acc_by_src
        metrics["vqa_f1_by_source"] = f1_by_src

    report = [p for p in preds if p["task"] == "report"]
    if report:
        preds_txt = [p["pred"] for p in report]
        golds_txt = [p["gold"] for p in report]
        metrics.update(_text_gen_metrics(preds_txt, golds_txt))
    return metrics


def _text_gen_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import sacrebleu

        out["report_bleu"] = round(
            sacrebleu.corpus_bleu(preds, [golds]).score / 100.0, 4
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] BLEU skipped: {exc}")
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(g, p)["rougeL"].fmeasure for p, g in zip(preds, golds)
        ]
        out["report_rougeL"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] ROUGE skipped: {exc}")
    return out


def _score_model(model_ref: str, rows: List[Dict], cfg: Dict) -> Dict[str, float]:
    from .infer import generate_answer, load_for_inference

    model, tokenizer = load_for_inference(
        model_ref,
        load_in_4bit=bool(cfg.get("load_in_4bit", True)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
    )
    preds: List[Dict] = []
    for i, row in enumerate(rows):
        image = open_image(row)
        max_new = 64 if row["task"] == "vqa" else 256
        pred = generate_answer(model, tokenizer, image, row["prompt"], max_new)
        preds.append(
            {
                "task": row["task"],
                "source": row.get("source"),
                "answer_type": row.get("answer_type"),
                "pred": pred,
                "gold": row["target"],
            }
        )
        if (i + 1) % 10 == 0:
            print(f"    scored {i + 1}/{len(rows)}")
    # Free GPU memory between models.
    try:
        import gc

        import torch

        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return _compute_metrics(preds)


def _subsample(rows: List[Dict], num_samples: Optional[int], seed: int) -> List[Dict]:
    if num_samples is None or num_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    # Keep a task-balanced subset when possible.
    by_task: Dict[str, List[Dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    picked: List[Dict] = []
    per = max(1, num_samples // max(1, len(by_task)))
    for task_rows in by_task.values():
        rng.shuffle(task_rows)
        picked.extend(task_rows[:per])
    rng.shuffle(picked)
    return picked[:num_samples]


def _run_meta(run_id: str) -> Dict:
    run_json = RUNS_DIR / run_id / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"No run.json for run_id={run_id}")
    with open(run_json, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_run(run_id: str, num_samples: Optional[int] = None) -> Dict[str, float]:
    from . import registry

    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.get("gpu", 0)))

    meta = _run_meta(run_id)
    base_model = meta["base_model"]
    data_version = meta.get("data_version", "unknown")
    processed_dir = ROOT / str(cfg.get("processed_dir", "data/processed"))
    seed = int(cfg.get("seed", 3407))

    test_rows = load_rows(processed_dir, "test")
    eval_rows = _subsample(test_rows, num_samples, seed)
    print(f"==> Evaluating run {run_id} on {len(eval_rows)} test rows")

    adapter_dir = RUNS_DIR / run_id / "lora"
    ft_metrics = _score_model(str(adapter_dir), eval_rows, cfg)
    print(f"    finetuned: {ft_metrics}")

    # Baseline: cache only for full-test runs (subsets differ per call).
    full_test = num_samples is None
    baseline = registry.load_baseline(base_model, data_version) if full_test else None
    if baseline is None:
        print("==> Scoring frozen base model for baseline...")
        baseline = _score_model(base_model, eval_rows, cfg)
        if full_test:
            registry.save_baseline(base_model, data_version, baseline)
    print(f"    baseline:  {baseline}")

    registry.log_metrics(
        run_id,
        metrics=ft_metrics,
        baseline=baseline,
        n_eval=len(eval_rows),
        data_version=data_version,
    )
    return ft_metrics


def evaluate_crosssite(
    run_id: str,
    eval_path: str,
    name: str = "slake_xray_en",
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Score a run on a held-out CROSS-SITE eval set (e.g. SLAKE).

    Results are written to ``outputs/runs/<run_id>/crosssite_<name>.json`` and
    merged into a ``crosssite`` field on the registry row. The in-domain
    ``metrics.json`` / ``data_version`` benchmark is never touched. The frozen
    base is scored as a baseline and cached in a namespace separate from the
    in-domain baseline (keyed by base_model + crosssite_version).
    """
    from . import registry

    cfg = load_yaml(ROOT / "configs" / "train.yaml")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.get("gpu", 0)))

    meta = _run_meta(run_id)
    base_model = meta["base_model"]
    seed = int(cfg.get("seed", 3407))

    eval_file = Path(eval_path)
    rows = load_jsonl(eval_file)
    if not rows:
        raise ValueError(f"No rows in cross-site set {eval_file}")

    # crosssite_version keys the baseline cache; read from the sibling stats.json.
    crosssite_version = "unknown"
    stats_path = eval_file.parent / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            crosssite_version = json.load(f).get("crosssite_version", "unknown")

    full = num_samples is None
    eval_rows = _subsample(rows, num_samples, seed)
    print(
        f"==> Cross-site eval [{name}] run {run_id} on {len(eval_rows)} rows "
        f"(crosssite_version={crosssite_version})"
    )

    adapter_dir = RUNS_DIR / run_id / "lora"
    ft_metrics = _score_model(str(adapter_dir), eval_rows, cfg)
    print(f"    finetuned: {ft_metrics}")

    # Baseline: cache only for full runs (subsets differ per call).
    cache_key = f"crosssite-{name}-{crosssite_version}"
    baseline = registry.load_baseline(base_model, cache_key) if full else None
    if baseline is None:
        print("==> Scoring frozen base model for cross-site baseline...")
        baseline = _score_model(base_model, eval_rows, cfg)
        if full:
            registry.save_baseline(base_model, cache_key, baseline)
    print(f"    baseline:  {baseline}")

    entry = {
        "metrics": ft_metrics,
        "baseline": baseline,
        "n": len(eval_rows),
        "crosssite_version": crosssite_version,
        "eval_path": rel_to_root(eval_file),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"crosssite_{name}.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "name": name, **entry}, f, indent=2)

    # Merge into the run's 'crosssite' dict without clobbering other names.
    existing: Dict[str, Dict] = {}
    for r in registry.load_registry():
        if r.get("run_id") == run_id:
            existing = dict(r.get("crosssite") or {})
            break
    existing[name] = entry
    registry.update_run(run_id, crosssite=existing)
    print(f"==> Logged cross-site [{name}] for {run_id}: {ft_metrics}")
    return ft_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a run on the frozen test set.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument(
        "--crosssite",
        default=None,
        help="path to a cross-site eval JSONL (e.g. data/crosssite/slake_xray_en.jsonl)",
    )
    p.add_argument("--crosssite-name", default="slake_xray_en")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.crosssite:
        evaluate_crosssite(
            run_id=args.run_id,
            eval_path=args.crosssite,
            name=args.crosssite_name,
            num_samples=args.num_samples,
        )
    else:
        evaluate_run(run_id=args.run_id, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
