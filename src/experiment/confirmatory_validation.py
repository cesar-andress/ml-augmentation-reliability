"""Pre-run freeze validation for confirmatory execution."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any

from src.experiment.confirmatory_constants import (
    A3_CONFIG_SHA256,
    COHORT_SHA256,
    FREEZE_COMMIT,
    FREEZE_TAG,
    HPO_CONFIG_SHA256,
    PROTOCOL_VERSION,
    VALID_FOLDS,
    VALID_REPEATS,
)
from src.experiment.confirmatory_protocol import (
    ConfirmatoryProtocolError,
    sha256_path,
    validate_confirmatory_protocol_start,
)


class ConfirmatoryPreRunError(ConfirmatoryProtocolError):
    """Raised when pre-run validation fails."""


def _git_tag_commit(repo_root: Path, tag: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", f"{tag}^{{commit}}"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return out
    except subprocess.CalledProcessError as e:
        raise ConfirmatoryPreRunError(f"freeze tag {tag!r} not found: {e.output}") from e


def load_frozen_dataset_row(repo_root: Path, dataset_id: int) -> dict[str, str]:
    csv_path = repo_root / "artifacts/manifests/datasets_frozen_v1_2.csv"
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["resolved_openml_id"]) == dataset_id:
                return row
    raise ConfirmatoryPreRunError(f"dataset_id {dataset_id} not in frozen cohort")


def validate_pre_run(
    *,
    repo_root: Path,
    dataset_id: int,
    repeat: int,
    fold: int,
) -> dict[str, Any]:
    """Full pre-run validation before any checkpoint/dataset payload access."""
    if repeat not in VALID_REPEATS:
        raise ConfirmatoryPreRunError(f"repeat must be in {VALID_REPEATS}, got {repeat}")
    if fold not in VALID_FOLDS:
        raise ConfirmatoryPreRunError(f"fold must be in {VALID_FOLDS}, got {fold}")

    tag_commit = _git_tag_commit(repo_root, FREEZE_TAG)
    if tag_commit != FREEZE_COMMIT:
        raise ConfirmatoryPreRunError(
            f"tag {FREEZE_TAG} points to {tag_commit}, expected {FREEZE_COMMIT}"
        )

    proto = validate_confirmatory_protocol_start(repo_root=repo_root, protocol_version=PROTOCOL_VERSION)

    hpo_sha = sha256_path(repo_root / "configs/hpo_v1_2_1.yaml")
    a3_sha = sha256_path(repo_root / "configs/a3_protocol_v1_2_1.yaml")
    if hpo_sha != HPO_CONFIG_SHA256:
        raise ConfirmatoryPreRunError(f"HPO config SHA mismatch: {hpo_sha}")
    if a3_sha != A3_CONFIG_SHA256:
        raise ConfirmatoryPreRunError(f"A3 config SHA mismatch: {a3_sha}")
    if proto["cohort_sha256"] != COHORT_SHA256:
        raise ConfirmatoryPreRunError("cohort SHA mismatch")

    ds_row = load_frozen_dataset_row(repo_root, dataset_id)

    return {
        **proto,
        "freeze_tag": FREEZE_TAG,
        "freeze_commit": tag_commit,
        "hpo_config_sha256": hpo_sha,
        "a3_config_sha256": a3_sha,
        "dataset_id": dataset_id,
        "dataset_name": ds_row["dataset_name"],
        "dataset_version": int(ds_row["openml_version"]),
        "expected_raw_checksum": ds_row["raw_checksum"],
        "repeat": repeat,
        "fold": fold,
    }
