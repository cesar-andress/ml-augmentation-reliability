"""Frozen cohort canonical content identity manifest (engineering v1)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.canonical_content_hash import (
    canonical_content_sha256,
    feature_name_order_fingerprint,
    row_order_fingerprint,
)
from src.data.openml_loader import legacy_frozen_parquet_sha256, parquet_bytes_checksum

IDENTITY_MANIFEST_V1 = "artifacts/manifests/dataset_content_identity_v1.csv"
FROZEN_COHORT_CSV = "artifacts/manifests/datasets_frozen_v1_2.csv"

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


class DatasetIdentityError(Exception):
    """Raised when dataset content identity validation fails."""


def load_identity_manifest(repo_root: Path) -> dict[int, dict[str, str]]:
    path = repo_root / IDENTITY_MANIFEST_V1
    if not path.exists():
        raise DatasetIdentityError(f"missing identity manifest: {path}")
    out: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oid = int(row["resolved_openml_id"])
            out[oid] = row
    return out


def load_frozen_legacy_row(repo_root: Path, dataset_id: int) -> dict[str, str]:
    path = repo_root / FROZEN_COHORT_CSV
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["resolved_openml_id"]) == dataset_id:
                return row
    raise DatasetIdentityError(f"dataset {dataset_id} not in {FROZEN_COHORT_CSV}")


def validate_dataset_content_identity(
    *,
    repo_root: Path,
    openml_id: int,
    openml_version: int,
    dataset_name: str,
    target_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    legacy_parquet_sha256: str | None = None,
) -> dict[str, Any]:
    """Authoritative confirmatory identity check (canonical + metadata + optional legacy bytes)."""
    identity = load_identity_manifest(repo_root).get(openml_id)
    if identity is None:
        raise DatasetIdentityError(f"dataset {openml_id} missing from content identity manifest")

    if int(identity["openml_version"]) != int(openml_version):
        raise DatasetIdentityError(
            f"OpenML version mismatch for {openml_id}: got {openml_version}, "
            f"expected {identity['openml_version']}"
        )
    if str(identity["dataset_name"]) != str(dataset_name):
        raise DatasetIdentityError(
            f"dataset name mismatch for {openml_id}: got {dataset_name!r}, "
            f"expected {identity['dataset_name']!r}"
        )
    if str(identity.get("target_name", "")) and target_name:
        if str(identity["target_name"]) != str(target_name):
            raise DatasetIdentityError(
                f"target mismatch for {openml_id}: got {target_name!r}, "
                f"expected {identity['target_name']!r}"
            )

    canonical = canonical_content_sha256(X, y, target_name=target_name or "y")
    expected_canonical = identity["canonical_content_sha256"]
    if canonical != expected_canonical:
        raise DatasetIdentityError(
            f"canonical_content_sha256 mismatch for {openml_id}: got {canonical}, "
            f"expected {expected_canonical}"
        )

    legacy_expected = identity.get("legacy_frozen_parquet_sha256") or identity.get("raw_checksum")
    if legacy_parquet_sha256 and legacy_expected:
        if legacy_parquet_sha256 != legacy_expected:
            raise DatasetIdentityError(
                f"legacy_frozen_parquet_sha256 mismatch for {openml_id}: "
                f"got {legacy_parquet_sha256}, expected {legacy_expected}"
            )

    return {
        "resolved_openml_id": openml_id,
        "openml_version": openml_version,
        "dataset_name": dataset_name,
        "target_name": target_name,
        "canonical_content_sha256": canonical,
        "legacy_frozen_parquet_sha256": legacy_parquet_sha256 or legacy_expected,
        "identity_manifest_version": "v1",
    }


def compute_identity_row(
    *,
    openml_id: int,
    openml_version: int,
    dataset_name: str,
    target_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    legacy_frozen_parquet_sha256: str,
    class_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build one manifest row for dataset_content_identity_v1.csv."""
    y_s = pd.Series(y)
    if class_counts is None:
        class_counts = {str(k): int(v) for k, v in y_s.value_counts().items()}
    return {
        "resolved_openml_id": openml_id,
        "openml_version": openml_version,
        "dataset_name": dataset_name,
        "target_name": target_name,
        "n_rows": len(X),
        "n_source_features": int(X.shape[1]),
        "class_counts_json": pd.Series(class_counts).to_json(),
        "legacy_frozen_parquet_sha256": legacy_frozen_parquet_sha256,
        "canonical_content_sha256": canonical_content_sha256(X, y, target_name=target_name),
        "feature_name_order_fingerprint": feature_name_order_fingerprint(X),
        "row_order_fingerprint": row_order_fingerprint(X, y),
    }
