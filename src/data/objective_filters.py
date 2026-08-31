"""Objective eligibility filters for confirmatory dataset freeze."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.feature_audit import FeatureTypeAudit
from src.data.splitting import build_split_for_fold


@dataclass
class ObjectiveResult:
    pass_all: bool
    checks: dict[str, bool]
    metrics: dict[str, Any]
    max_balanced_train_size: int | None
    prevalence_upper_bound_used: float


def tabpfn_max_balanced_train_size(
    y_bin: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 42,
) -> int:
    """Simulate planned splits; return max 2 * N_majority_TRAIN over folds."""
    y_bin = np.asarray(y_bin).astype(int)
    maxima = []
    for repeat in range(n_repeats):
        for fold in range(n_splits):
            split = build_split_for_fold(
                y_bin,
                n_splits=n_splits,
                n_repeats=n_repeats,
                seed=seed,
                repeat_index=repeat,
                fold_index=fold,
            )
            y_train = y_bin[split.train]
            # majority count in TRAIN (label 0 is majority after our binarize_labels
            # which sets minority=1; majority is the larger class among {0,1})
            n0 = int((y_train == 0).sum())
            n1 = int((y_train == 1).sum())
            n_maj = max(n0, n1)
            maxima.append(2 * n_maj)
    return int(max(maxima)) if maxima else 0


def evaluate_objective(
    *,
    y_bin: np.ndarray,
    audit: FeatureTypeAudit,
    n_rows: int,
    missing_frac: float,
    prevalence_upper: float = 0.40,
    tabpfn_row_limit: int = 10_000,
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 42,
) -> ObjectiveResult:
    y_bin = np.asarray(y_bin).astype(int)
    n_minority = int(y_bin.sum())  # positive = minority by construction
    prev = float(n_minority / n_rows) if n_rows else 0.0
    max_bal = tabpfn_max_balanced_train_size(
        y_bin, n_splits=n_splits, n_repeats=n_repeats, seed=seed
    )
    n_classes = int(len(np.unique(y_bin)))

    checks = {
        "binary": n_classes == 2,
        "rows_in_range": 500 <= n_rows <= 10_000,
        "has_continuous": len(audit.continuous_cols) >= 1,
        "prevalence_ge_0_02": prev >= 0.02,
        "prevalence_le_upper": prev <= prevalence_upper,
        "minority_count_ge_250": n_minority >= 250,
        "missing_lt_10pct": missing_frac < 0.10,
        "encoded_features_le_100": audit.n_encoded_features_est <= 100,
        "tabpfn_envelope_balanced_train_le_10000": max_bal <= tabpfn_row_limit,
    }
    metrics = {
        "n_rows": n_rows,
        "n_minority": n_minority,
        "n_majority": int(n_rows - n_minority),
        "minority_prevalence": prev,
        "missing_frac": missing_frac,
        "n_continuous": len(audit.continuous_cols),
        "n_numeric": len(audit.numeric_cols),
        "n_categorical": len(audit.categorical_cols),
        "n_encoded_features_est": audit.n_encoded_features_est,
        "missing_indicator_count": audit.missing_indicator_count,
        "max_balanced_train_size": max_bal,
        "prevalence_upper_bound_used": prevalence_upper,
    }
    return ObjectiveResult(
        pass_all=all(checks.values()),
        checks=checks,
        metrics=metrics,
        max_balanced_train_size=max_bal,
        prevalence_upper_bound_used=prevalence_upper,
    )
