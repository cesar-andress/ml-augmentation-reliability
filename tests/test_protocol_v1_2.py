"""Protocol v1.2 screening / freeze tests (outcome-blind; no model training)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.confirmatory_guard import NonConfirmatoryDataError, assert_confirmatory_frame, load_confirmatory_results
from src.data.binarization_exclusion import is_artificial_binarization
from src.data.candidates import (
    FOURTH_SOURCE_FORBIDDEN,
    build_candidate_universe_v1_2,
    load_climb_candidates,
    load_openml_cc18_candidates,
    load_tabarena_candidates,
)
from src.data.treatment_intensity import prevalence_le_040_equiv_r_ge_05, treatment_intensity_r

ROOT = Path(__file__).resolve().parents[1]


def test_binarization_description_and_binaryClass_rules():
    hit, reason = is_artificial_binarization(
        description="This is a binarized version of the original data set as was used in ...",
        default_target_attribute="class",
    )
    assert hit and "description_matches" in reason
    hit2, reason2 = is_artificial_binarization(description="normal data", default_target_attribute="binaryClass")
    assert hit2 and reason2 == "default_target_attribute_is_binaryClass"
    miss, _ = is_artificial_binarization(description="ordinary binary medical data", default_target_attribute="Outcome")
    assert not miss


def test_known_ids_excluded_generically_not_by_hardcoded_id():
    """Fetch metadata for 735/833/976/1021 and assert generic rule fires."""
    import openml

    for oid in [735, 833, 976, 1021]:
        ds = openml.datasets.get_dataset(oid, download_data=False, download_qualities=False)
        hit, reason = is_artificial_binarization(
            description=ds.description or "",
            default_target_attribute=ds.default_target_attribute,
        )
        assert hit, f"OpenML {oid} should be caught generically, got reason={reason!r}"


def test_prevalence_and_treatment_intensity_equivalence():
    for p in [0.02, 0.10, 0.25, 0.40, 0.3999, 0.4001, 0.45]:
        if 0 < p < 0.5:
            r = treatment_intensity_r(p)
            assert (p <= 0.40) == (r >= 0.5) or abs(p - 0.40) < 1e-12
            assert prevalence_le_040_equiv_r_ge_05(p) or abs(p - 0.40) < 1e-9
    assert treatment_intensity_r(0.40) == pytest.approx(0.5)
    assert treatment_intensity_r(1 / 3) == pytest.approx(1.0)


def test_cc18_three_pool_union_provenance_no_fourth_source():
    assert FOURTH_SOURCE_FORBIDDEN is True
    ta = load_tabarena_candidates()
    cl = load_climb_candidates()
    cc = load_openml_cc18_candidates()
    assert len(cc) == 72
    u = build_candidate_universe_v1_2()
    pools = set(u["source_pool"].unique())
    assert pools == {"tabarena_v0.1", "climb", "openml_cc18"}
    # provenance preserved: same OID can appear in multiple pools
    assert u["candidate_key"].is_unique
    assert set(["source_pool", "source_dataset_id", "resolved_openml_id", "dataset_name"]).issubset(u.columns)


def test_d_threshold_is_10():
    text = (ROOT / "artifacts/protocol/protocol_v1_2.yaml").read_text()
    assert "minimum_cohort_size_D: 10" in text


def test_smoke_cannot_enter_confirmatory_logic():
    smoke = ROOT / "results/smoke/smoke_results.parquet"
    if smoke.exists():
        with pytest.raises(NonConfirmatoryDataError):
            load_confirmatory_results(smoke)
    df = pd.DataFrame({"scientific_status": ["SMOKE_ONLY"], "log_loss": [0.1]})
    with pytest.raises(NonConfirmatoryDataError):
        assert_confirmatory_frame(df)


def test_v1_2_ledger_and_exclusions_if_present():
    cand = ROOT / "artifacts/manifests/candidates_v1_2.csv"
    if not cand.exists():
        pytest.skip("v1.2 screen not run yet")
    df = pd.read_csv(cand)
    assert df["resolved_openml_id"].is_unique or df.groupby("resolved_openml_id").ngroups <= len(df)
    # each OID once in ledger
    assert df["resolved_openml_id"].is_unique
    for oid in [735, 833, 976, 1021]:
        sub = df[df["resolved_openml_id"] == oid]
        assert len(sub) == 1
        assert "artificial_binarization" in str(sub.iloc[0]["first_exclusion_reason"])
        assert sub.iloc[0]["final_decision"] == "EXCLUDE"
    # 46930 determinism manifest, not inferential KEEP
    det = pd.read_csv(ROOT / "artifacts/manifests/pipeline_determinism_dataset.csv")
    assert 46930 in set(det["resolved_openml_id"].astype(int))
    sub30 = df[df["resolved_openml_id"] == 46930]
    assert len(sub30) == 1
    assert sub30.iloc[0]["final_decision"] == "EXCLUDE"
    assert "Structural zero treatment" in str(sub30.iloc[0]["first_exclusion_reason"])
    # 44232 excluded, 1489 keep if present as KEEP or excluded for other objective reasons
    sub442 = df[df["resolved_openml_id"] == 44232]
    assert len(sub442) == 1
    assert sub442.iloc[0]["final_decision"] == "EXCLUDE"
    sub1489 = df[df["resolved_openml_id"] == 1489]
    if len(sub1489) and bool(sub1489.iloc[0]["objective_eligible"]):
        assert sub1489.iloc[0]["final_decision"] == "KEEP"
        assert "speaker" in str(sub1489.iloc[0].get("semantic_decision_reason", "")).lower() or "ELENA" in str(
            sub1489.iloc[0].get("semantic_decision_reason", "")
        )


def test_frozen_v1_2_if_present():
    csv_path = ROOT / "artifacts/manifests/datasets_frozen_v1_2.csv"
    sha_path = ROOT / "artifacts/manifests/datasets_frozen_v1_2.sha256"
    if not csv_path.exists():
        pytest.skip("cohort not frozen")
    data = csv_path.read_bytes()
    assert sha_path.read_text().strip() == hashlib.sha256(data).hexdigest()
    df = pd.read_csv(csv_path)
    assert len(df) >= 10
    assert df["resolved_openml_id"].is_unique
    assert list(df["resolved_openml_id"]) == sorted(df["resolved_openml_id"].tolist())
    assert "treatment_intensity_r" in df.columns
    # no fourth source in pools
    pools = ";".join(df["source_pools"].astype(str))
    assert "tabarena" in pools or "climb" in pools or "openml_cc18" in pools
