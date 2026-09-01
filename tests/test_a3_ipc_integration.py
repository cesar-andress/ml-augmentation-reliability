"""Tiny non-scientific A3 IPC integration: main env <-> synthcity worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.arms import augment_a3_tabddpm_via_subprocess, assert_balanced_to_majority
from src.experiment.live_adapters import make_toy_dataset_bundle
from src.preprocessing.pipeline import fit_preprocessor


def test_a3_ipc_tiny_smoke_worker(tmp_path):
    """File IPC through .venv_synthcity with smoke-only settings.

    Marked NON_SCIENTIFIC_INTEGRATION_TEST — must never enter confirmatory.
    """
    synth_py = ROOT / ".venv_synthcity" / "bin" / "python"
    if not synth_py.exists():
        pytest.skip("synthcity env missing")

    bundle = make_toy_dataset_bundle(seed=1)
    # use a small balanced-enough subset
    X = bundle.X.iloc[:120].copy()
    y = (bundle.y.astype("category").cat.codes.to_numpy()[:120] == 0).astype(np.int64)
    # ensure minority size
    if y.sum() < 20:
        y[:30] = 1
    if (1 - y).sum() < 20:
        y[-30:] = 0
    # force imbalance
    y[40:] = 0
    y[:25] = 1

    pre = fit_preprocessor(X, unknown_category_sentinel=-1)
    Xp = pre.transform(X)
    job = tmp_path / "a3_ipc_job"
    aug = augment_a3_tabddpm_via_subprocess(
        Xp,
        y,
        pre.meta,
        job_dir=job,
        synthcity_python=str(synth_py),
        random_state=0,
        smoke_config={
            "n_iter": 2,
            "batch_size": 32,
            "num_timesteps": 10,
            "mark": "NON_SCIENTIFIC_INTEGRATION_TEST",
            "scientific_mode": False,
        },
    )
    assert_balanced_to_majority(aug.y)
    assert aug.n_synthetic > 0
    assert (job / "status.json").exists()
    status = json.loads((job / "status.json").read_text())
    assert status["status"] == "ok"
    assert status["mark"] == "NON_SCIENTIFIC_INTEGRATION_TEST"
    assert status["n_generated"] >= 1
    # write marker so artifacts are never confused with confirmatory
    (tmp_path / "MARK.txt").write_text("NON_SCIENTIFIC_INTEGRATION_TEST\n")
