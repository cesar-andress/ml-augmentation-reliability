"""Extended confirmatory result schema."""

from __future__ import annotations

from typing import Any

import pandas as pd

CONFIRMATORY_RESULT_COLUMNS = [
    "protocol_version",
    "freeze_tag",
    "cohort_sha256",
    "unit_id",
    "dataset_id",
    "dataset_version",
    "repeat",
    "fold",
    "seed",
    "learner",
    "family",
    "arm",
    "temperature",
    "calibration",
    "scientific_status",
    "n_train_real",
    "n_train_final",
    "n_majority_real",
    "n_minority_real",
    "minority_prevalence_real",
    "minority_prevalence_final",
    "hpo_candidate_count",
    "best_hpo_candidate_index",
    "log_loss",
    "auroc",
    "calibration_slope",
    "calibration_intercept",
    "conformal_set_size",
    "conformal_coverage",
    "f1_tuned",
    "brier",
    "auprc",
    "repair_row_fraction",
    "repair_cell_fraction",
    "generator_fit_seconds",
    "generator_sample_seconds",
    "hpo_seconds",
    "learner_fit_seconds",
    "inference_seconds",
    "peak_gpu_memory_mb",
    "manifest_path",
    "status",
    "failure_reason",
]


def validate_confirmatory_result_row(row: dict[str, Any]) -> None:
    missing = [c for c in CONFIRMATORY_RESULT_COLUMNS if c not in row]
    if missing:
        raise AssertionError(f"confirmatory result row missing columns: {missing}")


def empty_confirmatory_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CONFIRMATORY_RESULT_COLUMNS)
