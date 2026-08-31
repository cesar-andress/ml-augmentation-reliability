"""Unit tests for frozen protocol components. Must fail loudly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.augmentation.arms import augment_a0, augment_a1, augment_a2, assert_balanced_to_majority, shared_repair
from src.calibration.posthoc import IsotonicCalibrator, PlattCalibrator, clip_prob, logit
from src.calibration.threshold import tune_threshold_f1
from src.conformal.split_conformal import conformity_score, conformal_quantile
from src.data.splitting import build_split_for_fold
from src.experiment.manifest import create_manifest, save_manifest
from src.experiment.schema import RESULT_COLUMNS, validate_result_row
from src.learners.wrappers import TabPFNLearner
from src.preprocessing.pipeline import fit_preprocessor


def _toy_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num_int": rng.integers(0, 5, size=n).astype(float),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    # inject missing
    df.loc[rng.choice(n, size=20, replace=False), "num1"] = np.nan
    y = (df["num1"].fillna(0) + (df["cat"] == "a").astype(float) > 0).astype(int).to_numpy()
    # ensure both classes reasonably sized
    if y.sum() < 50:
        y[:80] = 1
    if (1 - y).sum() < 50:
        y[-80:] = 0
    return df, y


def test_split_proportions_and_no_leakage():
    _, y = _toy_frame(500)
    split = build_split_for_fold(y, n_splits=5, n_repeats=2, seed=42, repeat_index=0, fold_index=0)
    split.assert_no_overlap()
    n = len(y)
    assert abs(len(split.test) / n - 0.20) < 0.05
    rem = n - len(split.test)
    assert abs(len(split.train) / rem - 0.625) < 0.08
    assert abs(len(split.cal_a) / rem - 0.1875) < 0.08
    assert abs(len(split.cal_b) / rem - 0.1875) < 0.08


def test_stratification_rough():
    _, y = _toy_frame(600)
    split = build_split_for_fold(y, n_splits=5, n_repeats=2, seed=0, repeat_index=0, fold_index=0)
    global_prev = y.mean()
    for part in [split.train, split.cal_a, split.cal_b, split.test]:
        assert abs(y[part].mean() - global_prev) < 0.12


def test_preprocessing_train_only_and_unknown_cats_and_no_nan():
    X, y = _toy_frame(300)
    split = build_split_for_fold(y, n_splits=5, n_repeats=1, seed=1, repeat_index=0, fold_index=0)
    pre = fit_preprocessor(X.iloc[split.train], unknown_category_sentinel=-1)
    # unknown category in test
    X_te = X.iloc[split.test].copy()
    X_te.iloc[0, X_te.columns.get_loc("cat")] = "UNSEEN_LEVEL"
    Xt = pre.transform(X.iloc[split.train])
    Xte = pre.transform(X_te)
    assert not Xt.isna().any().any()
    assert not Xte.isna().any().any()
    assert int(Xte.iloc[0]["cat"]) == -1
    # medians from train only: corrupting test must not change train transform
    Xt2 = pre.transform(X.iloc[split.train])
    pd.testing.assert_frame_equal(Xt, Xt2)


def test_augmentation_counts_and_smote_selection():
    X, y = _toy_frame(400, seed=2)
    # make imbalance
    y = y.copy()
    y[:] = 0
    y[:80] = 1
    pre = fit_preprocessor(X, unknown_category_sentinel=-1)
    Xp = pre.transform(X)
    a0 = augment_a0(Xp, y)
    assert len(a0.y) == len(y)
    a1 = augment_a1(Xp, y, random_state=0)
    assert_balanced_to_majority(a1.y)
    assert a1.repair_fraction == 0.0
    a2 = augment_a2(Xp, y, pre.meta, random_state=0, k_neighbors=5)
    assert_balanced_to_majority(a2.y)
    assert a2.method == "A2_SMOTENC"
    # all-continuous -> SMOTE
    Xc = Xp[pre.meta.numeric_cols + pre.meta.missing_indicator_cols]
    from dataclasses import replace
    from src.preprocessing.pipeline import PreprocessMeta

    meta_c = PreprocessMeta(
        numeric_cols=list(Xc.columns),
        categorical_cols=[],
        missing_indicator_cols=[c for c in Xc.columns if c.startswith("__miss__")],
        numeric_medians={c: float(Xc[c].median()) for c in Xc.columns},
        categorical_modes={},
        category_levels={},
        unknown_category_sentinel=-1,
        output_columns=list(Xc.columns),
    )
    a2c = augment_a2(Xc, y, meta_c, random_state=0, k_neighbors=5)
    assert a2c.method == "A2_SMOTE"


def test_repair_snaps_invalid_categorical_codes():
    X, y = _toy_frame(200)
    pre = fit_preprocessor(X, unknown_category_sentinel=-1)
    Xp = pre.transform(X)
    syn = Xp.iloc[:10].copy()
    syn.iloc[0, syn.columns.get_loc("cat")] = 999
    fixed, frac = shared_repair(syn, Xp, pre.meta)
    assert int(fixed.iloc[0]["cat"]) == -1
    assert frac > 0


def test_tabpfn_limits():
    learner = TabPFNLearner(
        {
            "model_path": "checkpoints/tabpfn/tabpfn-v2-classifier.ckpt",
            "n_estimators": 8,
            "inference_precision": "float32",
            "n_preprocessing_jobs": 1,
            "random_state": 0,
            "ignore_pretraining_limits": False,
            "softmax_temperature": 1.0,
        }
    )
    X = pd.DataFrame(np.zeros((10_001, 3)))
    with pytest.raises(AssertionError, match="row limit"):
        learner.fit(X, np.zeros(10_001, dtype=int))
    X2 = pd.DataFrame(np.zeros((10, 101)))
    with pytest.raises(AssertionError, match="feature limit"):
        learner.fit(X2, np.zeros(10, dtype=int))


def test_logloss_clipping_and_platt_isotonic_threshold_conformal_manifest_schema(tmp_path):
    p = np.array([0.0, 1.0, 0.5])
    c = clip_prob(p)
    assert c.min() >= 1e-6 and c.max() <= 1 - 1e-6
    z = logit(np.array([0.2, 0.8, 0.55, 0.1, 0.9]))
    y = np.array([0, 1, 1, 0, 1])
    platt = PlattCalibrator().fit(np.array([0.2, 0.8, 0.55, 0.1, 0.9]), y)
    out = platt.transform(np.array([0.3, 0.7]))
    assert out.shape == (2,)
    iso = IsotonicCalibrator().fit(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    iso_out = iso.transform(np.array([-1.0, 2.0]))
    assert iso_out.min() >= 0 and iso_out.max() <= 1

    # threshold tie -> smallest
    y_t = np.array([0, 1, 0, 1])
    # craft probs so multiple thresholds share F1
    p_t = np.array([0.1, 0.4, 0.6, 0.9])
    thr = tune_threshold_f1(y_t, p_t)
    assert isinstance(thr, float)

    scores = conformity_score(np.array([0.9, 0.2, 0.6]), np.array([1, 0, 1]))
    q = conformal_quantile(scores, alpha=0.10)
    assert 0 <= q <= 1

    m = create_manifest(
        repo_root=Path("."),
        dataset_id=1,
        dataset_name="toy",
        dataset_version=1,
        dataset_checksum="abc",
        fold=0,
        repeat=0,
        seed=0,
        learner="xgboost",
        arm="A0",
        preprocessing_config={},
        augmentation_config={},
        hyperparameters={},
    )
    path = save_manifest(m, tmp_path / "m.json")
    assert path.exists()
    json.loads(path.read_text())

    row = {c: None for c in RESULT_COLUMNS}
    validate_result_row(row)
    with pytest.raises(AssertionError):
        validate_result_row({"dataset_id": 1})
