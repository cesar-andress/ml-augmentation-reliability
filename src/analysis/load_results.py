"""Load confirmatory units into a canonical analysis table (Protocol v1.2.1)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FROZEN_DATASET_IDS = (
    44,
    1067,
    1489,
    40983,
    42178,
    46911,
    46915,
    46921,
    46927,
    46952,
)
LEARNERS = ("xgboost", "catboost", "tabpfn", "tabicl")
GBDT = ("xgboost", "catboost")
TFM = ("tabpfn", "tabicl")
AUG_ARMS = ("A1", "A2", "A3")
BASE_ARM = "A0"
A0PLUS = "A0+"
EXPECTED_CELLS = 1800


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_identity_manifest(repo_root: Path) -> dict[int, dict[str, str]]:
    path = repo_root / "artifacts/manifests/dataset_content_identity_v1.csv"
    out: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["resolved_openml_id"])] = row
    return out


def minority_count_from_identity(row: dict[str, str]) -> int:
    counts = json.loads(row["class_counts_json"])
    return int(min(counts.values()))


def prevalence_from_identity(row: dict[str, str]) -> float:
    counts = json.loads(row["class_counts_json"])
    vals = list(counts.values())
    n = sum(vals)
    return float(min(vals) / n)


def build_input_manifest(repo_root: Path) -> dict[str, Any]:
    units_root = repo_root / "results/confirmatory/units"
    entries = []
    total_rows = 0
    for ds in FROZEN_DATASET_IDS:
        for repeat in (0, 1):
            for fold in range(5):
                uid = f"d{ds}_r{repeat}_f{fold}"
                root = units_root / uid
                uc = root / "status/unit_complete.json"
                rp = root / "metrics/results.parquet"
                if not uc.exists() or not rp.exists():
                    raise FileNotFoundError(f"missing complete unit artifacts: {uid}")
                ucj = json.loads(uc.read_text())
                if ucj.get("status") != "COMPLETE":
                    raise AssertionError(f"{uid} not COMPLETE")
                df = pd.read_parquet(rp)
                if len(df) != 18:
                    raise AssertionError(f"{uid} cells={len(df)}")
                if not (df["scientific_status"] == "CONFIRMATORY").all():
                    raise AssertionError(f"{uid} non-CONFIRMATORY")
                if "smoke" in df.columns and df["smoke"].astype(bool).any():
                    raise AssertionError(f"{uid} smoke flag")
                total_rows += len(df)
                entries.append(
                    {
                        "unit_id": uid,
                        "dataset_id": ds,
                        "repeat": repeat,
                        "fold": fold,
                        "unit_complete_path": str(uc.relative_to(repo_root)),
                        "results_parquet_path": str(rp.relative_to(repo_root)),
                        "results_parquet_sha256": _sha256_file(rp),
                        "n_rows": int(len(df)),
                        "scientific_mode": True,
                        "smoke": False,
                    }
                )
    if len(entries) != 100:
        raise AssertionError(f"expected 100 units, got {len(entries)}")
    if total_rows != EXPECTED_CELLS:
        raise AssertionError(f"expected {EXPECTED_CELLS} cells, got {total_rows}")
    return {
        "n_units": len(entries),
        "n_cells": total_rows,
        "dataset_ids": list(FROZEN_DATASET_IDS),
        "learners": list(LEARNERS),
        "arms": ["A0", "A1", "A2", "A3", "A0+"],
        "units": entries,
    }


def load_canonical_table(repo_root: Path) -> pd.DataFrame:
    frames = []
    for ds in FROZEN_DATASET_IDS:
        for repeat in (0, 1):
            for fold in range(5):
                uid = f"d{ds}_r{repeat}_f{fold}"
                rp = repo_root / f"results/confirmatory/units/{uid}/metrics/results.parquet"
                df = pd.read_parquet(rp)
                frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    key = ["dataset_id", "repeat", "fold", "learner", "arm"]
    if table.duplicated(subset=key).any():
        raise AssertionError("duplicate scientific cells")
    if len(table) != EXPECTED_CELLS:
        raise AssertionError(f"canonical table size {len(table)}")
    # Exclude smoke / dry_run if present
    if "scientific_status" in table.columns:
        table = table[table["scientific_status"] == "CONFIRMATORY"].copy()
    return table


def load_sidecar_metric(repo_root: Path, unit_id: str, learner: str, arm: str, key: str) -> float:
    stem = f"{learner}_{'A0plus' if arm == 'A0+' else arm}"
    path = repo_root / f"results/confirmatory/units/{unit_id}/metrics/{stem}.json"
    payload = json.loads(path.read_text())
    return float(payload[key])


def attach_sidecar_log_losses(repo_root: Path, table: pd.DataFrame) -> pd.DataFrame:
    """Attach Platt / isotonic TEST log loss from per-cell sidecar metrics."""
    platt = []
    iso = []
    for row in table.itertuples(index=False):
        platt.append(load_sidecar_metric(repo_root, row.unit_id, row.learner, row.arm, "log_loss_platt"))
        iso.append(load_sidecar_metric(repo_root, row.unit_id, row.learner, row.arm, "log_loss_isotonic"))
    out = table.copy()
    out["log_loss_platt"] = platt
    out["log_loss_isotonic"] = iso
    return out


def _positive_prob(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        if a.shape[1] == 2:
            return a[:, 1]
        if a.shape[1] == 1:
            return a[:, 0]
    return a.ravel()


def load_tfm_temperature_log_loss(
    repo_root: Path,
    *,
    unit_id: str,
    dataset_id: int,
    learner: str,
    arm: str,
    temperature_key: str = "sensitivity",
) -> float:
    """Recompute raw TEST log loss for a TFM temperature variant from stored preds."""
    from sklearn.metrics import log_loss as sk_ll

    from src.calibration.posthoc import clip_prob
    from src.data.openml_loader import binarize_labels

    unit = repo_root / f"results/confirmatory/units/{unit_id}"
    pred_path = unit / "predictions" / learner / ("A0plus" if arm == "A0+" else arm) / "test.json"
    payload = json.loads(pred_path.read_text())
    key = f"p_{temperature_key}"
    if key not in payload:
        raise KeyError(f"{pred_path} missing {key}")
    p = clip_prob(_positive_prob(payload[key]))
    row_ids = np.asarray(payload["row_ids"], dtype=int)
    y_full = pd.read_parquet(repo_root / f"data/raw/openml/{dataset_id}/y.parquet")["y"]
    y_bin, _ = binarize_labels(y_full)
    y_test = y_bin[row_ids]
    return float(sk_ll(y_test, np.column_stack([1 - p, p]), labels=[0, 1]))
