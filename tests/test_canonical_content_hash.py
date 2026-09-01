"""Tests for canonical dataset content identity (no model training)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IDENTITY_CSV = ROOT / "artifacts/manifests/dataset_content_identity_v1.csv"
FROZEN_CSV = ROOT / "artifacts/manifests/datasets_frozen_v1_2.csv"

from src.data.canonical_content_hash import canonical_content_sha256
from src.data.dataset_content_identity import FROZEN_DATASET_IDS, validate_dataset_content_identity
from src.data.openml_loader import (
    cache_raw_openml,
    fetch_openml_dataframe,
    legacy_frozen_parquet_sha256,
    load_frozen_openml_raw,
)


def _toy_xy():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([False, True, False])
    return X, y


def test_canonical_hash_deterministic():
    X, y = _toy_xy()
    assert canonical_content_sha256(X, y, target_name="t") == canonical_content_sha256(
        X.copy(), y.copy(), target_name="t"
    )


def test_canonical_hash_row_order_sensitive():
    X, y = _toy_xy()
    h0 = canonical_content_sha256(X, y, target_name="t")
    X2 = X.iloc[[2, 0, 1]].reset_index(drop=True)
    y2 = y.iloc[[2, 0, 1]].reset_index(drop=True)
    assert canonical_content_sha256(X2, y2, target_name="t") != h0


def test_canonical_hash_column_order_sensitive():
    X, y = _toy_xy()
    h0 = canonical_content_sha256(X, y, target_name="t")
    assert canonical_content_sha256(X[["b", "a"]], y, target_name="t") != h0


def test_canonical_hash_value_sensitive():
    X, y = _toy_xy()
    h0 = canonical_content_sha256(X, y, target_name="t")
    X2 = X.copy()
    X2.loc[0, "a"] = 9.0
    assert canonical_content_sha256(X2, y, target_name="t") != h0


def test_canonical_hash_target_sensitive():
    X, y = _toy_xy()
    h0 = canonical_content_sha256(X, y, target_name="t")
    y2 = y.copy()
    y2.iloc[1] = False
    assert canonical_content_sha256(X, y2, target_name="t") != h0


def test_canonical_bool_target_stable_across_storage():
    X, y = _toy_xy()
    h_bool = canonical_content_sha256(X, y, target_name="defects")
    h_cat = canonical_content_sha256(X, y.astype("category"), target_name="defects")
    assert h_bool == h_cat


def test_cache_writer_preserves_raw_target_dtype(tmp_path):
    X, y, _, _, target, _ = fetch_openml_dataframe(1067)
    cache_raw_openml(1067, X, y, {"openml_id": 1067}, raw_root=tmp_path, target_name=target)
    y_disk = pd.read_parquet(tmp_path / "1067" / "y.parquet")["y"]
    assert str(y_disk.dtype) == "bool"


def test_openml_1067_freeze_byte_hash_reproduced(tmp_path):
    X, y, _, _, target, _ = fetch_openml_dataframe(1067)
    expected = "6fecfff679a8168cdf32dd8a3a01a59ec1e2cbd0d8405141a971599450c5c730"
    assert cache_raw_openml(1067, X, y, None, raw_root=tmp_path, target_name=target) == expected


def test_openml_1067_canonical_matches_manifest():
    row = _identity_row(1067)
    X, y, _, _, target, _ = fetch_openml_dataframe(1067)
    assert canonical_content_sha256(X, y, target_name=target) == row["canonical_content_sha256"]


def test_future_validation_accepts_exact_1067_content():
    X, y, name, version, target, _ = fetch_openml_dataframe(1067)
    validate_dataset_content_identity(
        repo_root=ROOT,
        openml_id=1067,
        openml_version=int(version),
        dataset_name=name,
        target_name=target,
        X=X,
        y=y,
        legacy_parquet_sha256=_legacy_bytes_from_frames(X, y),
    )


def test_future_validation_rejects_altered_row_order():
    X, y, name, version, target, _ = fetch_openml_dataframe(1067)
    digest = _legacy_bytes_from_frames(X, y)
    with pytest.raises(Exception):
        validate_dataset_content_identity(
            repo_root=ROOT,
            openml_id=1067,
            openml_version=int(version),
            dataset_name=name,
            target_name=target,
            X=X.iloc[::-1].reset_index(drop=True),
            y=y.iloc[::-1].reset_index(drop=True),
            legacy_parquet_sha256=digest,
        )


def test_future_validation_rejects_altered_target():
    X, y, name, version, target, _ = fetch_openml_dataframe(1067)
    digest = _legacy_bytes_from_frames(X, y)
    y2 = y.copy()
    y2.iloc[0] = not bool(y2.iloc[0])
    with pytest.raises(Exception):
        validate_dataset_content_identity(
            repo_root=ROOT,
            openml_id=1067,
            openml_version=int(version),
            dataset_name=name,
            target_name=target,
            X=X,
            y=y2,
            legacy_parquet_sha256=digest,
        )


def test_future_validation_rejects_wrong_openml_version():
    X, y, name, _, target, _ = fetch_openml_dataframe(1067)
    with pytest.raises(Exception):
        validate_dataset_content_identity(
            repo_root=ROOT,
            openml_id=1067,
            openml_version=99,
            dataset_name=name,
            target_name=target,
            X=X,
            y=y,
            legacy_parquet_sha256=_legacy_bytes_from_frames(X, y),
        )


def test_all_ten_frozen_datasets_have_identity_entries():
    with IDENTITY_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = sorted(int(r["resolved_openml_id"]) for r in rows)
    assert ids == sorted(FROZEN_DATASET_IDS)


def test_historical_datasets_frozen_v1_2_unchanged():
    digest = hashlib.sha256(FROZEN_CSV.read_bytes()).hexdigest()
    sha_path = ROOT / "artifacts/manifests/datasets_frozen_v1_2.sha256"
    assert sha_path.read_text().strip() == digest


def test_dataset_44_scientific_files_untouched():
    pre = json.loads(
        (ROOT / "artifacts/audits/data_identity_patch/d44_pre_patch_hashes.json").read_text()
    )
    for unit, files in pre.items():
        root = ROOT / "results/confirmatory/units" / unit
        for fname, expected in files.items():
            rel = f"status/{fname}" if fname.endswith(".json") else f"metrics/{fname}"
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
            assert got == expected


def test_load_frozen_1067_from_rematerialized_cache():
    row = _identity_row(1067)
    bundle = load_frozen_openml_raw(
        1067,
        expected_raw_checksum=row["legacy_frozen_parquet_sha256"],
        expected_canonical_content_sha256=row["canonical_content_sha256"],
        expected_version=int(row["openml_version"]),
        expected_name=row["dataset_name"],
        expected_target_name=row["target_name"],
        repo_root=ROOT,
    )
    assert bundle.checksum == row["legacy_frozen_parquet_sha256"]
    assert bundle.canonical_content_sha256 == row["canonical_content_sha256"]


def test_d1067_failed_unit_preserved():
    st = json.loads(
        (ROOT / "results/confirmatory/units/d1067_r0_f0/status/unit_status.json").read_text()
    )
    assert st["phases"]["P01_LOAD_DATASET"]["status"] == "FAILED"
    assert "raw_checksum mismatch" in st["phases"]["P01_LOAD_DATASET"]["exception"]


def _identity_row(openml_id: int) -> dict[str, str]:
    with IDENTITY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["resolved_openml_id"]) == openml_id:
                return row
    raise KeyError(openml_id)


def _legacy_bytes_from_frames(X: pd.DataFrame, y: pd.Series) -> str:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        xp, yp = td_path / "X.parquet", td_path / "y.parquet"
        X.to_parquet(xp)
        pd.DataFrame({"y": y}).to_parquet(yp)
        return legacy_frozen_parquet_sha256(xp, yp)
