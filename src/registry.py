"""Model / experiment registry.

Every training run gets its own directory under ``outputs/runs/<run_id>/`` and
one row in the append-only index ``outputs/registry.jsonl``. This keeps many
runs (different params / datasets) self-contained and comparable, and never
overwrites earlier results.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .utils import ROOT

OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
REGISTRY_PATH = OUTPUTS_DIR / "registry.jsonl"
BASELINES_DIR = OUTPUTS_DIR / "baselines"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def _short_model_name(base_model: str) -> str:
    name = base_model.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "", name)


def make_run_id(run_name: str, base_model: str, lora_r: int) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{run_name}-{_short_model_name(base_model)}-r{lora_r}-{ts}"


def load_registry() -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_registry(rows: List[Dict[str, Any]]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _upsert_row(row: Dict[str, Any]) -> None:
    """Insert or replace a registry row keyed by run_id."""
    rows = load_registry()
    by_id = {r["run_id"]: r for r in rows}
    by_id[row["run_id"]] = {**by_id.get(row["run_id"], {}), **row}
    # Preserve original ordering, appending new ids at the end.
    order = [r["run_id"] for r in rows]
    if row["run_id"] not in order:
        order.append(row["run_id"])
    _write_registry([by_id[i] for i in order])


def update_run(run_id: str, **fields: Any) -> None:
    """Patch arbitrary fields on a run's registry row."""
    _upsert_row({"run_id": run_id, **fields})


def start_run(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Create the run directory + a 'started' registry row.

    ``meta`` should include: run_name, base_model, lora_r, lora_alpha,
    learning_rate, num_train_epochs, max_steps, quant, data_mix, sources,
    data_version, n_train, seed, gpu, and 'config' (the full resolved config
    dict to snapshot).

    Returns a handle dict with ``run_id`` and ``run_dir`` (Path).
    """
    run_id = make_run_id(
        meta.get("run_name", "run"),
        meta["base_model"],
        int(meta.get("lora_r", 0)),
    )
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta.get("config", {}), f, sort_keys=False)

    run_json = {
        "run_id": run_id,
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        **{k: v for k, v in meta.items() if k != "config"},
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_json, f, indent=2)

    row = {
        "run_id": run_id,
        "status": "started",
        "base_model": meta["base_model"],
        "lora_r": meta.get("lora_r"),
        "lora_alpha": meta.get("lora_alpha"),
        "lr": meta.get("learning_rate"),
        "epochs": meta.get("num_train_epochs"),
        "max_steps": meta.get("max_steps"),
        "quant": meta.get("quant"),
        "data_mix": meta.get("data_mix"),
        "sources": meta.get("sources"),
        "data_version": meta.get("data_version"),
        "n_train": meta.get("n_train"),
        "seed": meta.get("seed"),
        "git_commit": run_json["git_commit"],
        "gpu": meta.get("gpu"),
        "adapter_path": str((run_dir / "lora").as_posix()),
    }
    _upsert_row(row)
    print(f"==> Registered run {run_id}")
    return {"run_id": run_id, "run_dir": run_dir}


def log_metrics(
    run_id: str,
    metrics: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    train_minutes: Optional[float] = None,
    n_eval: Optional[int] = None,
    data_version: Optional[str] = None,
) -> None:
    """Write metrics.json under the run dir and mark the registry row done."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "metrics": metrics,
        "baseline": baseline,
        "n_eval": n_eval,
        "data_version": data_version,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    update: Dict[str, Any] = {"run_id": run_id, "status": "done", "metrics": metrics}
    if baseline is not None:
        update["baseline"] = baseline
    if train_minutes is not None:
        update["train_minutes"] = round(train_minutes, 2)
    if n_eval is not None:
        update["n_eval"] = n_eval
    if data_version is not None:
        update["data_version"] = data_version
    _upsert_row(update)
    print(f"==> Logged metrics for {run_id}: {metrics}")


# --- baseline cache (frozen base model on a given data_version) --------------
def baseline_path(base_model: str, data_version: str) -> Path:
    key = f"{_short_model_name(base_model)}-{data_version}"
    return BASELINES_DIR / f"{key}.json"


def load_baseline(base_model: str, data_version: str) -> Optional[Dict[str, Any]]:
    p = baseline_path(base_model, data_version)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_baseline(base_model: str, data_version: str, metrics: Dict[str, Any]) -> None:
    p = baseline_path(base_model, data_version)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
