"""Tests for dataset freeze / confirmatory isolation (no model training)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.confirmatory_guard import NonConfirmatoryDataError, assert_confirmatory_frame, load_confirmatory_results
from src.data.candidates import build_candidate_universe, load_climb_candidates, load_tabarena_candidates
from src.data.duplicates import build_duplicate_map
from src.data.feature_audit import audit_feature_types
from src.data.objective_filters import evaluate_objective, tabpfn_max_balanced_train_size
from src.data.openml_loader import binarize_labels

ROOT = Path(__file__).resolve().parents[1]


def test_tabarena_climb_union_and_provenance():
    ta = load_tabarena_candidates()
    cl = load_climb_candidates()
    assert len(ta) >= 50
    assert len(cl) == 73
    assert set(ta["source_pool"]) == {"tabarena_v0.1"}
    assert set(cl["source_pool"]) == {"climb"}
    u = build_candidate_universe()
    assert "resolved_openml_id" in u.columns
    assert u["candidate_key"].is_unique or u["candidate_key"].nunique() == len(u)
    # curated TabArena IDs are in 46xxx range typically
    assert ta["resolved_openml_id"].min() >= 46904 or ta["in_openml_suite_457"].any()


def test_objective_boundaries_and_prevalence_and_envelope():
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 3, size=n)})
    y = np.array([1] * 300 + [0] * 700)
    audit = audit_feature_types(X)
    assert len(audit.continuous_cols) >= 1
    miss = float(X.isna().to_numpy().mean())
    y_bin = y
    obj = evaluate_objective(y_bin=y_bin, audit=audit, n_rows=n, missing_frac=miss, prevalence_upper=0.40)
    assert obj.checks["minority_count_ge_250"]
    assert obj.checks["prevalence_le_upper"]
    assert obj.metrics["minority_prevalence"] == pytest.approx(0.3)
    # too few minority
    y2 = np.array([1] * 100 + [0] * 900)
    obj2 = evaluate_objective(y_bin=y2, audit=audit, n_rows=n, missing_frac=miss, prevalence_upper=0.40)
    assert not obj2.checks["minority_count_ge_250"]
    # envelope
    mx = tabpfn_max_balanced_train_size(y_bin)
    assert mx <= 10000
    assert obj.checks["tabpfn_envelope_balanced_train_le_10000"]


def test_missing_fraction_and_encoded_count():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "c": ["x", "y", "x", None]})
    audit = audit_feature_types(X)
    assert audit.missing_indicator_count == 2
    assert audit.n_encoded_features_est == 2 + 2


def test_duplicate_map_integrity_not_dimensions_alone():
    screen = pd.DataFrame(
        [
            {
                "resolved_openml_id": 1,
                "dataset_name": "foo",
                "n_rows": 1000,
                "n_source_features": 10,
                "minority_prevalence": 0.2,
                "source_pools": "climb",
                "raw_checksum": "aaa",
                "objective_eligible": True,
            },
            {
                "resolved_openml_id": 2,
                "dataset_name": "foo",
                "n_rows": 1000,
                "n_source_features": 10,
                "minority_prevalence": 0.2,
                "source_pools": "tabarena_v0.1",
                "raw_checksum": "aaa",
                "objective_eligible": True,
            },
            {
                "resolved_openml_id": 3,
                "dataset_name": "bar",
                "n_rows": 1000,
                "n_source_features": 10,
                "minority_prevalence": 0.3,
                "source_pools": "climb",
                "raw_checksum": "bbb",
                "objective_eligible": True,
            },
            {
                "resolved_openml_id": 4,
                "dataset_name": "baz",
                "n_rows": 1000,
                "n_source_features": 10,
                "minority_prevalence": 0.25,
                "source_pools": "climb",
                "raw_checksum": "ccc",
                "objective_eligible": True,
            },
        ]
    )
    dup = build_duplicate_map(screen)
    # 1 and 2 are duplicates; retain tabarena id 2
    assert set(dup["removed_openml_id"]) == {1}
    assert set(dup["retained_openml_id"]) == {2}
    # 3 and 4 same dims different names/prevalence/checksum -> not duplicates
    assert 3 not in set(dup["removed_openml_id"]) and 4 not in set(dup["removed_openml_id"])


def test_smoke_confirmatory_separation_and_guard(tmp_path):
    smoke = ROOT / "results" / "smoke" / "smoke_results.parquet"
    assert smoke.exists()
    assert not (ROOT / "results" / "raw" / "smoke_results.parquet").exists()
    assert (ROOT / "results" / "raw" / "MIGRATION_NOTE.md").exists()

    df = pd.DataFrame(
        {
            "dataset_id": [1, 2],
            "scientific_status": ["CONFIRMATORY", "SMOKE_ONLY"],
            "log_loss": [0.1, 0.2],
        }
    )
    with pytest.raises(NonConfirmatoryDataError):
        assert_confirmatory_frame(df)
    ok = df[df["scientific_status"] == "CONFIRMATORY"]
    assert_confirmatory_frame(ok)

    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(NonConfirmatoryDataError):
        load_confirmatory_results(p)

    # refuse smoke directory
    with pytest.raises(NonConfirmatoryDataError):
        load_confirmatory_results(smoke)


def test_frozen_cohort_artifacts_if_present():
    csv_path = ROOT / "artifacts" / "manifests" / "datasets_frozen_v1.csv"
    sha_path = ROOT / "artifacts" / "manifests" / "datasets_frozen_v1.sha256"
    if not csv_path.exists():
        pytest.skip("cohort not frozen yet")
    data = csv_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert sha_path.read_text().strip() == digest
    df = pd.read_csv(csv_path)
    assert df["resolved_openml_id"].is_unique
    assert list(df["resolved_openml_id"]) == sorted(df["resolved_openml_id"].tolist())
    yaml_path = ROOT / "artifacts" / "manifests" / "confirmatory_freeze_v1.yaml"
    assert yaml_path.exists()
    text = yaml_path.read_text()
    for oid in df["resolved_openml_id"]:
        assert str(int(oid)) in text
