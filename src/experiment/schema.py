"""Result schema helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

RESULT_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "repeat",
    "fold",
    "seed",
    "learner",
    "family",
    "arm",
    "temperature",
    "calibration",
    "n_train_real",
    "n_train_final",
    "minority_prevalence_real",
    "minority_prevalence_final",
    "log_loss",
    "auroc",
    "calibration_slope",
    "calibration_intercept",
    "conformal_set_size",
    "conformal_coverage",
    "f1_tuned",
    "brier",
    "auprc",
    "repair_fraction",
    "generator_fit_seconds",
    "generator_sample_seconds",
    "learner_fit_seconds",
    "inference_seconds",
    "peak_gpu_memory_mb",
    "status",
    "failure_reason",
    "manifest_path",
    "scientific_mark",
]


def empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def validate_result_row(row: dict[str, Any]) -> None:
    missing = [c for c in RESULT_COLUMNS if c not in row]
    if missing:
        raise AssertionError(f"result row missing columns: {missing}")


def append_result(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    validate_result_row(row)
    rows.append(row)


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    for r in rows:
        validate_result_row(r)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
