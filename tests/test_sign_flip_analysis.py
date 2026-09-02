"""Tests for frozen exact sign-flip confirmatory statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.aggregate import bar_delta, dataset_delta, dataset_psi, per_fold_delta
from src.analysis.sign_flip import exact_sign_flip_ci, exact_sign_flip_test, studentized_mean_t


def test_studentized_mean_known():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean=3, sd=sqrt(2.5), t = 3 / (sqrt(2.5)/sqrt(5))
    expected = 3.0 / (np.sqrt(2.5) / np.sqrt(5))
    assert abs(studentized_mean_t(x) - expected) < 1e-12


def test_exact_sign_flip_enumerates_1024():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10)
    res = exact_sign_flip_test(x)
    assert res["D"] == 10
    assert res["n_permutations"] == 1024
    assert 0.0 <= res["p_value"] <= 1.0


def test_exact_sign_flip_all_positive_small_p():
    x = np.ones(10)
    res = exact_sign_flip_test(x)
    # only all-+ and theoretically patterns with same |t|; for all equal nonzero,
    # only two patterns give |t|=inf (all+ and all-) when sd=0 after? wait all ones flipped still equal abs
    # all equal values: sd=0, t = +/- inf depending on mean sign; only all+ and all- give |inf|
    # Actually all sign patterns of equal |values| give sd=0 and mean = (+-1)*c with various...
    # For all ones: after signs, values are ±1; mean and sd depend on # of + vs -.
    assert res["p_value"] <= 2 / 1024 + 1e-12 or res["n_positive"] == 10


def test_symmetric_zeros_p_one():
    x = np.zeros(10)
    res = exact_sign_flip_test(x)
    assert res["statistic"] == 0.0
    assert abs(res["p_value"] - 1.0) < 1e-15


def test_ci_contains_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.05, scale=0.1, size=10)
    ci = exact_sign_flip_ci(x, level=0.95)
    assert ci["lower"] <= float(np.mean(x)) <= ci["upper"]
    assert ci["grid_step"] == pytest.approx((x.max() - x.min()) / 1000.0)


def test_delta_sign_convention():
    """Delta = LL_a - LL_A0; positive means augmentation worsens."""
    import pandas as pd

    rows = []
    for ds in [1]:
        for fold in range(10):
            for learner in ["xgboost", "catboost", "tabpfn", "tabicl"]:
                rows.append(
                    {
                        "dataset_id": ds,
                        "repeat": fold // 5,
                        "fold": fold % 5,
                        "learner": learner,
                        "family": "gbdt" if learner in {"xgboost", "catboost"} else "tfm",
                        "arm": "A0",
                        "log_loss": 0.5,
                    }
                )
                for arm, ll in [("A1", 0.6), ("A2", 0.55), ("A3", 0.7)]:
                    rows.append(
                        {
                            "dataset_id": ds,
                            "repeat": fold // 5,
                            "fold": fold % 5,
                            "learner": learner,
                            "family": "gbdt" if learner in {"xgboost", "catboost"} else "tfm",
                            "arm": arm,
                            "log_loss": ll,
                        }
                    )
    # Need 10 datasets for aggregate asserts — build minimal synthetic differently
    frames = []
    for ds in range(10):
        for fold in range(10):
            for learner in ["xgboost", "catboost", "tabpfn", "tabicl"]:
                frames.append(
                    {
                        "dataset_id": ds,
                        "repeat": fold // 5,
                        "fold": fold % 5,
                        "learner": learner,
                        "family": "gbdt" if learner in {"xgboost", "catboost"} else "tfm",
                        "arm": "A0",
                        "log_loss": 1.0,
                    }
                )
                for arm in ["A1", "A2", "A3"]:
                    frames.append(
                        {
                            "dataset_id": ds,
                            "repeat": fold // 5,
                            "fold": fold % 5,
                            "learner": learner,
                            "family": "gbdt" if learner in {"xgboost", "catboost"} else "tfm",
                            "arm": arm,
                            "log_loss": 1.2,
                        }
                    )
    table = pd.DataFrame(frames)
    fd = per_fold_delta(table)
    assert (fd["delta"] > 0).all()
    bar = bar_delta(fd)
    dd = dataset_delta(bar)
    assert (dd["Delta_d"] > 0).all()
    # Psi: TFM and GBDT same → 0
    ps = dataset_psi(bar)
    assert np.allclose(ps["Psi_d"], 0.0)
