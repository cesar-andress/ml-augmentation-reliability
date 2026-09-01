"""Augmentation arms A0–A3 with shared repair."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE, RandomOverSampler

from src.preprocessing.pipeline import PreprocessMeta, categorical_feature_indices


@dataclass
class AugmentResult:
    X: pd.DataFrame
    y: np.ndarray
    repair_fraction: float
    n_synthetic: int
    method: str
    generator_fit_seconds: float = 0.0
    generator_sample_seconds: float = 0.0
    scientific_mark: str = ""
    extra: dict[str, Any] | None = None


def _class_counts(y: np.ndarray) -> tuple[int, int, int, int]:
    """Return majority_label, minority_label, n_maj, n_min."""
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) != 2:
        raise ValueError(f"expected binary y, got {vals}")
    order = np.argsort(counts)
    minority_label = int(vals[order[0]])
    majority_label = int(vals[order[1]])
    n_min = int(counts[order[0]])
    n_maj = int(counts[order[1]])
    return majority_label, minority_label, n_maj, n_min


def shared_repair(
    X_syn: pd.DataFrame,
    X_train: pd.DataFrame,
    meta: PreprocessMeta,
) -> tuple[pd.DataFrame, float]:
    """Snap categoricals, round integer-valued numerics, clip to TRAIN min/max.

    Returns (repaired_frame, row_repair_fraction).
    """
    X_out, row_frac, _cell_frac = shared_repair_with_cell_stats(X_syn, X_train, meta)
    return X_out, row_frac


def shared_repair_with_cell_stats(
    X_syn: pd.DataFrame,
    X_train: pd.DataFrame,
    meta: PreprocessMeta,
) -> tuple[pd.DataFrame, float, float]:
    """Frozen repair with both row-level and cell-level fractions."""
    if len(X_syn) == 0:
        return X_syn.copy(), 0.0, 0.0

    X_out = X_syn.copy()
    changed_rows = np.zeros(len(X_out), dtype=bool)
    cell_changed = 0
    n_cells = 0

    def _mark_cells(mask: np.ndarray) -> None:
        nonlocal cell_changed, n_cells
        mask = np.asarray(mask, dtype=bool)
        n_cells += len(mask)
        cell_changed += int(mask.sum())
        if len(mask):
            changed_rows[: len(mask)] |= mask

    # Categorical codes -> valid levels or sentinel is already a code; snap to {0..K-1, sentinel}
    for c in meta.categorical_cols:
        k = len(meta.category_levels[c])
        sentinel = meta.unknown_category_sentinel
        valid = set(range(k)) | {sentinel}
        col = X_out[c].to_numpy().astype(np.int64)
        bad = np.array([int(v) not in valid for v in col], dtype=bool)
        if bad.any():
            col = col.copy()
            col[bad] = sentinel
            X_out[c] = col
        _mark_cells(bad)
        rounded = np.rint(X_out[c].to_numpy()).astype(np.int64)
        round_changed = rounded != X_out[c].to_numpy()
        if round_changed.any():
            X_out[c] = rounded
        _mark_cells(round_changed)

    for c in meta.numeric_cols:
        train_col = X_train[c].to_numpy(dtype=np.float64)
        lo, hi = float(np.min(train_col)), float(np.max(train_col))
        vals = X_out[c].to_numpy(dtype=np.float64).copy()
        is_int_valued = np.all(np.isclose(train_col, np.rint(train_col)))
        if is_int_valued:
            rounded = np.rint(vals)
            _mark_cells(~np.isclose(rounded, vals))
            vals = rounded
        clipped = np.clip(vals, lo, hi)
        _mark_cells(~np.isclose(clipped, vals))
        X_out[c] = clipped.astype(np.float64)

    for c in meta.missing_indicator_cols:
        vals = X_out[c].to_numpy(dtype=np.float64)
        snapped = np.clip(np.rint(vals), 0, 1).astype(np.int64)
        _mark_cells(snapped != np.rint(vals).astype(np.int64))
        X_out[c] = snapped

    row_frac = float(changed_rows.mean()) if len(changed_rows) else 0.0
    cell_frac = float(cell_changed / n_cells) if n_cells else 0.0
    return X_out, row_frac, cell_frac


def augment_a0(X_train: pd.DataFrame, y_train: np.ndarray) -> AugmentResult:
    return AugmentResult(
        X=X_train.copy(),
        y=np.asarray(y_train).copy(),
        repair_fraction=0.0,
        n_synthetic=0,
        method="A0",
    )


def augment_a1(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int) -> AugmentResult:
    maj, mino, n_maj, n_min = _class_counts(y_train)
    k = n_maj - n_min
    ros = RandomOverSampler(sampling_strategy={mino: n_maj, maj: n_maj}, random_state=random_state)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    # ROS should need zero repair
    return AugmentResult(
        X=X_res,
        y=np.asarray(y_res),
        repair_fraction=0.0,
        n_synthetic=k,
        method="A1",
    )


def augment_a2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta: PreprocessMeta,
    random_state: int,
    k_neighbors: int = 5,
) -> AugmentResult:
    maj, mino, n_maj, n_min = _class_counts(y_train)
    k = n_maj - n_min
    cat_idx = categorical_feature_indices(meta)
    sampling_strategy = {mino: n_maj, maj: n_maj}
    if cat_idx:
        sampler = SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
        method = "A2_SMOTENC"
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
        method = "A2_SMOTE"
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    # Repair only synthetic minority rows (appended after original)
    n_real = len(X_train)
    X_syn = X_res.iloc[n_real:].copy()
    X_syn_rep, repair_frac = shared_repair(X_syn, X_train, meta)
    X_out = pd.concat([X_res.iloc[:n_real], X_syn_rep], ignore_index=True)
    return AugmentResult(
        X=X_out,
        y=np.asarray(y_res),
        repair_fraction=repair_frac,
        n_synthetic=k,
        method=method,
    )


def augment_a3_tabddpm_via_subprocess(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta: PreprocessMeta,
    *,
    job_dir: Path,
    synthcity_python: str,
    random_state: int,
    smoke_config: dict[str, Any],
) -> AugmentResult:
    """File-based interface to .venv_synthcity for TabDDPM."""
    maj, mino, n_maj, n_min = _class_counts(y_train)
    k = n_maj - n_min
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    train_path = job_dir / "train.parquet"
    y_path = job_dir / "y.npy"
    meta_path = job_dir / "meta.json"
    cfg_path = job_dir / "gen_config.json"
    out_path = job_dir / "synthetic_minority.parquet"
    status_path = job_dir / "status.json"

    X_train.to_parquet(train_path)
    np.save(y_path, y_train)
    with meta_path.open("w") as f:
        json.dump(
            {
                "numeric_cols": meta.numeric_cols,
                "categorical_cols": meta.categorical_cols,
                "missing_indicator_cols": meta.missing_indicator_cols,
                "category_levels": meta.category_levels,
                "unknown_category_sentinel": meta.unknown_category_sentinel,
                "output_columns": meta.output_columns,
                "majority_label": maj,
                "minority_label": mino,
                "k": k,
            },
            f,
        )
    with cfg_path.open("w") as f:
        json.dump({"random_state": random_state, **smoke_config}, f)

    worker = Path(__file__).resolve().parents[2] / "scripts" / "a3_tabddpm_worker.py"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [synthcity_python, str(worker), str(job_dir)],
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"A3 worker failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    status = json.loads(status_path.read_text())
    if status.get("status") != "ok":
        raise RuntimeError(f"A3 worker status not ok: {status}")

    X_syn = pd.read_parquet(out_path)
    X_syn_rep, repair_frac = shared_repair(X_syn, X_train, meta)
    y_syn = np.full(len(X_syn_rep), mino, dtype=np.int64)
    X_out = pd.concat([X_train.reset_index(drop=True), X_syn_rep], ignore_index=True)
    y_out = np.concatenate([y_train, y_syn])
    # Ensure exact counts
    # If generator returned != k rows, truncate/pad by resampling repaired syn rows
    if len(X_syn_rep) != k:
        if len(X_syn_rep) == 0:
            raise RuntimeError("A3 generated 0 synthetic rows")
        rng = np.random.default_rng(random_state)
        take = rng.choice(len(X_syn_rep), size=k, replace=(len(X_syn_rep) < k))
        X_syn_rep = X_syn_rep.iloc[take].reset_index(drop=True)
        X_out = pd.concat([X_train.reset_index(drop=True), X_syn_rep], ignore_index=True)
        y_out = np.concatenate([y_train, np.full(k, mino, dtype=np.int64)])

    return AugmentResult(
        X=X_out,
        y=y_out,
        repair_fraction=repair_frac,
        n_synthetic=k,
        method="A3_TabDDPM",
        generator_fit_seconds=float(status.get("fit_seconds", wall)),
        generator_sample_seconds=float(status.get("sample_seconds", 0.0)),
        scientific_mark=smoke_config.get("mark", "") if smoke_config.get("scientific_mode") else smoke_config.get("mark", "SMOKE_ONLY_NOT_SCIENTIFIC"),
        extra=status,
    )


def assert_balanced_to_majority(y: np.ndarray) -> None:
    maj, mino, n_maj, n_min = _class_counts(y)
    if n_maj != n_min:
        raise AssertionError(f"expected balanced to majority={n_maj}, got min={n_min}")
