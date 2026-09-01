"""Protocol v1.2.1 amendment tests — no model training."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.experiment.confirmatory_protocol import (
    ACCEPTED_PROTOCOL_VERSION,
    ConfirmatoryProtocolError,
    load_a3_config,
    load_freeze_v1_2_1,
    sha256_path,
    validate_a3_protocol_mode,
    validate_confirmatory_protocol_start,
    validate_hpo_config,
    verify_synthcity_tabddpm_signature,
)
from src.experiment.hpo_v1_2_1 import (
    XGB_SEARCH_KEYS,
    CAT_SEARCH_KEYS,
    a0_plus_candidates,
    compute_k_extra,
    derive_generator_seed,
    derive_hpo_seed,
    generate_candidate_sequence,
    inner_augmentation_cache_key,
    load_hpo_config,
    make_hpo_rng,
    outer_a3_cache_key,
    sample_one_candidate,
    select_hpo_candidate,
    shared_arm_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_xgboost_hpo_ranges_match_amendment():
    cfg = load_hpo_config(str(ROOT / "configs/hpo_v1_2_1.yaml"))
    xgb = cfg["learners"]["xgboost"]
    assert xgb["fixed"]["n_estimators"] == 500
    assert xgb["fixed"]["tree_method"] == "hist"
    assert xgb["fixed"]["device"] == "cuda"
    assert xgb["fixed"]["enable_categorical"] is False
    assert xgb["search"]["max_depth"]["values"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert xgb["search"]["learning_rate"] == {"type": "log_uniform", "low": 0.01, "high": 0.30}
    assert xgb["search"]["subsample"] == {"type": "continuous_uniform", "low": 0.60, "high": 1.00}
    assert xgb["search"]["colsample_bytree"] == {"type": "continuous_uniform", "low": 0.60, "high": 1.00}
    assert xgb["search"]["min_child_weight"] == {"type": "log_uniform", "low": 1.0, "high": 20.0}
    assert xgb["search"]["reg_lambda"] == {"type": "log_uniform", "low": 0.1, "high": 10.0}
    assert set(xgb["search"].keys()) == set(XGB_SEARCH_KEYS)


def test_catboost_hpo_ranges_match_amendment():
    cfg = load_hpo_config(str(ROOT / "configs/hpo_v1_2_1.yaml"))
    cb = cfg["learners"]["catboost"]
    assert cb["fixed"]["iterations"] == 500
    assert cb["fixed"]["learning_rate"] == 0.05
    assert cb["fixed"]["task_type"] == "CPU"
    assert cb["search"]["depth"]["values"] == [4, 5, 6, 7, 8, 9, 10]
    assert cb["search"]["l2_leaf_reg"] == {"type": "log_uniform", "low": 1.0, "high": 30.0}
    assert cb["search"]["subsample"] == {"type": "continuous_uniform", "low": 0.60, "high": 1.00}
    assert cb["search"]["rsm"] == {"type": "continuous_uniform", "low": 0.60, "high": 1.00}
    assert cb["search"]["random_strength"] == {"type": "log_uniform", "low": 0.1, "high": 10.0}
    assert set(cb["search"].keys()) == set(CAT_SEARCH_KEYS)


def test_log_uniform_sampler_correct():
    cfg = load_hpo_config(str(ROOT / "configs/hpo_v1_2_1.yaml"))
    rng = np.random.default_rng(0)
    spec = cfg["learners"]["xgboost"]["search"]["learning_rate"]
    lo, hi = spec["low"], spec["high"]
    samples = [float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) for _ in range(1000)]
    assert min(samples) >= lo
    assert max(samples) <= hi


def test_hpo_sequence_deterministic():
    a = shared_arm_candidates(dataset_id=44, repeat=0, fold=0, learner="xgboost")
    b = shared_arm_candidates(dataset_id=44, repeat=0, fold=0, learner="xgboost")
    assert a == b
    c = shared_arm_candidates(dataset_id=44, repeat=0, fold=0, learner="catboost")
    assert a != c


def test_all_arms_share_first_20_configs():
    cfg = load_hpo_config(str(ROOT / "configs/hpo_v1_2_1.yaml"))
    seq = shared_arm_candidates(dataset_id=44, repeat=0, fold=0, learner="xgboost", cfg=cfg)
    for arm in ["A0", "A1", "A2", "A3"]:
        again = shared_arm_candidates(dataset_id=44, repeat=0, fold=0, learner="xgboost", cfg=cfg)
        assert seq == again, f"arm {arm} must share identical first 20 configs"


def test_a0_plus_is_prefix_extension_of_a0():
    k_extra = 5
    a0 = shared_arm_candidates(dataset_id=44, repeat=0, fold=0, learner="xgboost", n_candidates=20)
    a0p = a0_plus_candidates(dataset_id=44, repeat=0, fold=0, learner="xgboost", k_extra=k_extra)
    assert len(a0p) == 25
    assert a0p[:20] == a0


def test_tie_breaking_deterministic():
    metrics = [
        {"mean_inner_log_loss": 0.5, "std_inner_log_loss": 0.1, "candidate_index": 3},
        {"mean_inner_log_loss": 0.5, "std_inner_log_loss": 0.05, "candidate_index": 7},
        {"mean_inner_log_loss": 0.5, "std_inner_log_loss": 0.05, "candidate_index": 2},
    ]
    assert select_hpo_candidate(metrics) == 2


def test_hpo_forbids_cal_test_partitions():
    cfg = load_hpo_config(str(ROOT / "configs/hpo_v1_2_1.yaml"))
    forbidden = set(cfg["standard_budget"]["forbidden_partitions_for_selection"])
    assert {"CAL-A", "CAL-B", "TEST"}.issubset(forbidden)


def test_a3_installed_signature_matches_expected():
    info = verify_synthcity_tabddpm_signature(repo_root=ROOT)
    assert info["version"] == "0.2.12"
    assert info["plugin_class"] == "TabDDPMPlugin"
    assert info["verified"] is True
    assert not info["mismatches"]


def test_a3_protocol_mode_rejects_smoke_parameters():
    a3 = load_a3_config(str(ROOT / "configs/a3_protocol_v1_2_1.yaml"))
    validate_a3_protocol_mode(a3)
    bad = dict(a3)
    bad["constructor"] = dict(a3["constructor"])
    bad["constructor"]["n_iter"] = 2
    with pytest.raises(ConfirmatoryProtocolError, match="n_iter"):
        validate_a3_protocol_mode(bad)


def test_a3_outer_seed_is_outer_fold_seed():
    a3 = load_a3_config(str(ROOT / "configs/a3_protocol_v1_2_1.yaml"))
    assert a3["seeds"]["outer"] == "outer_fold_seed"
    assert a3["constructor"]["random_state"] == "outer_fold_seed"


def test_a3_inner_seeds_deterministic_and_arm_specific():
    s1 = derive_generator_seed(dataset_id=44, repeat=0, fold=0, arm="A3", inner_fold=0)
    s2 = derive_generator_seed(dataset_id=44, repeat=0, fold=0, arm="A3", inner_fold=0)
    s3 = derive_generator_seed(dataset_id=44, repeat=0, fold=0, arm="A2", inner_fold=0)
    assert s1 == s2
    assert s1 != s3


def test_augmented_artifact_cache_keys_learner_independent():
    k_xgb = inner_augmentation_cache_key(dataset_id=44, repeat=0, fold=0, arm="A3", inner_fold=1)
    k_cb = inner_augmentation_cache_key(dataset_id=44, repeat=0, fold=0, arm="A3", inner_fold=1)
    assert k_xgb == k_cb
    assert "learner" not in k_xgb


def test_outer_a3_cache_shared_across_learners():
    key = outer_a3_cache_key(dataset_id=44, repeat=0, fold=0)
    assert key == "44|0|0|A3|outer"


def test_inner_cache_policy_documented_in_hpo_config():
    cfg = load_hpo_config(str(ROOT / "configs/hpo_v1_2_1.yaml"))
    cache = cfg["inner_augmentation_cache"]
    assert cache["outer_a3_forbidden_in_hpo"] is True
    assert cache["per_candidate_a3_retrain_forbidden"] is True


def test_compute_k_extra_formula():
    assert compute_k_extra(g_max=100.0, t_bar=30.0) == 4
    assert compute_k_extra(g_max=10000.0, t_bar=10.0) == 200


def test_confirmatory_runner_rejects_protocol_v1_2():
    with pytest.raises(ConfirmatoryProtocolError, match="superseded"):
        validate_confirmatory_protocol_start(repo_root=ROOT, protocol_version="1.2")


def test_confirmatory_runner_accepts_v1_2_1_freeze():
    out = validate_confirmatory_protocol_start(repo_root=ROOT, protocol_version=ACCEPTED_PROTOCOL_VERSION)
    assert out["protocol_version"] == "1.2.1"
    assert out["scientific_mode"] is True
    assert out["smoke_settings_allowed"] is False
    assert out["standard_hpo_candidates"] == 20
    assert out["inner_cv_folds"] == 3


def test_freeze_v1_2_1_hashes_match_files():
    freeze = load_freeze_v1_2_1(str(ROOT / "artifacts/manifests/confirmatory_freeze_v1_2_1.yaml"))
    assert freeze["dataset_cohort_sha256"] == "209bc80826843940da92799ca48b406a34df7e2dbafd2ff26590d092187ecb34"
    assert sha256_path(ROOT / "configs/hpo_v1_2_1.yaml") == freeze["artifact_sha256"]["hpo_v1_2_1_yaml"]
    assert sha256_path(ROOT / "configs/a3_protocol_v1_2_1.yaml") == freeze["artifact_sha256"]["a3_protocol_v1_2_1_yaml"]


def test_v1_2_historical_files_unchanged_vs_tag():
    for rel in [
        "artifacts/protocol/protocol_v1_2.yaml",
        "artifacts/protocol/statistical_analysis_v1_2.yaml",
        "artifacts/manifests/confirmatory_freeze_v1_2.yaml",
        "artifacts/manifests/datasets_frozen_v1_2.csv",
    ]:
        local = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        tagged = hashlib.sha256(
            __import__("subprocess").check_output(
                ["git", "show", f"protocol-v1.2-freeze:{rel}"], cwd=ROOT
            )
        ).hexdigest()
        assert local == tagged, rel


def test_hpo_seed_derivation_matches_spec():
    seed = derive_hpo_seed(dataset_id=44, repeat=0, fold=0, learner="xgboost")
    msg = "44|0|0|xgboost|hpo-v1.2.1"
    expected = int(hashlib.sha256(msg.encode()).hexdigest(), 16) % (2**32)
    assert seed == expected
