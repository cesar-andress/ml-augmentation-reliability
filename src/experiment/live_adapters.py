"""Injectable adapters for confirmatory live execution (real or toy/mocked)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.augmentation.arms import augment_a3_tabddpm_via_subprocess
from src.data.openml_loader import DatasetBundle, load_openml_raw
from src.learners.wrappers import build_learner
from src.preprocessing.pipeline import PreprocessMeta


@dataclass
class LiveAdapters:
    """Hooks for orchestration. Defaults = real frozen implementations."""

    load_dataset: Callable[[int], DatasetBundle] | None = None
    augment_a3: Callable[..., Any] | None = None
    build_learner: Callable[..., Any] | None = None
    evaluate_hpo_fold: Callable[..., float] | None = None
    skip_freeze_validation: bool = False
    scientific: bool = True
    resource_logger: Callable[[], dict[str, Any]] | None = None
    expected_checksum: str | None = None
    expected_version: int | None = None
    expected_name: str | None = None
    hpo_candidate_limit: int | None = None  # toy only; None = full 20
    mark: str = ""

    def resolve_load_dataset(self) -> Callable[[int], DatasetBundle]:
        return self.load_dataset or load_openml_raw

    def resolve_build_learner(self) -> Callable[..., Any]:
        return self.build_learner or build_learner

    def resolve_augment_a3(self) -> Callable[..., Any]:
        return self.augment_a3 or augment_a3_tabddpm_via_subprocess

    def resource_snapshot(self) -> dict[str, Any]:
        if self.resource_logger:
            return self.resource_logger()
        out: dict[str, Any] = {}
        try:
            import psutil
            import os

            proc = psutil.Process(os.getpid())
            out["peak_rss_mb"] = proc.memory_info().rss / (1024**2)
        except Exception:
            out["peak_rss_mb"] = None
        try:
            import torch

            out["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                out["gpu_name"] = torch.cuda.get_device_name(0)
                out["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024**2))
                out["torch_version"] = torch.__version__
            else:
                out["peak_gpu_memory_mb"] = None
        except Exception as e:
            out["cuda_error"] = f"{type(e).__name__}: {e}"
        return out


def mock_prob_learner(name: str, family: str):
    """Deterministic toy learner for orchestration tests (no real model fit)."""

    class _Mock:
        def __init__(self):
            self.name = name
            self.family = family
            self._p = 0.5

        def fit(self, X, y):
            y = np.asarray(y)
            self._p = float(np.clip(y.mean(), 0.05, 0.95))
            return self

        def predict_proba(self, X):
            n = len(X)
            # slight feature dependence for non-constant probs
            if hasattr(X, "iloc"):
                base = pd.DataFrame(X).iloc[:, 0].to_numpy(dtype=np.float64)
            else:
                base = np.asarray(X)[:, 0].astype(np.float64)
            z = (base - np.nanmean(base)) / (np.nanstd(base) + 1e-6)
            p = 1.0 / (1.0 + np.exp(-(z * 0.5 + np.log(self._p / (1 - self._p)))))
            return np.clip(p, 1e-6, 1 - 1e-6)

    return _Mock


def make_toy_dataset_bundle(seed: int = 0) -> DatasetBundle:
    """Deterministic imbalanced binary fixture with numeric/categorical/missing/unknown."""
    rng = np.random.default_rng(seed)
    n = 400
    num1 = rng.normal(size=n)
    num_int = rng.integers(0, 6, size=n).astype(float)
    cat = rng.choice(["a", "b", "c"], size=n)
    # imbalance: ~20% minority
    logits = 0.8 * num1 + (cat == "a").astype(float) - 1.2
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p * 0.35).astype(int)
    if y.sum() < 60:
        y[:80] = 1
    if (1 - y).sum() < 60:
        y[-80:] = 0
    X = pd.DataFrame({"num1": num1, "num_int": num_int, "cat": cat})
    miss_idx = rng.choice(n, size=25, replace=False)
    X.loc[miss_idx, "num1"] = np.nan
    # unknown category reserved for non-train partitions (applied by tests after split if needed)
    checksum = "toy_fixture_checksum_v1"
    return DatasetBundle(
        openml_id=-1,
        name="toy_binary",
        version=1,
        X=X,
        y=pd.Series(y.astype(str)),
        checksum=checksum,
        description="NON_SCIENTIFIC toy fixture",
    )


def toy_a3_augmenter(X_train, y_train, meta, **kwargs):
    """Non-scientific A3 stand-in: noise around minority rows."""
    from src.augmentation.arms import AugmentResult, _class_counts, shared_repair_with_cell_stats

    maj, mino, n_maj, n_min = _class_counts(y_train)
    k = n_maj - n_min
    rng = np.random.default_rng(int(kwargs.get("random_state", 0)))
    X_min = X_train.loc[np.asarray(y_train) == mino]
    take = rng.choice(len(X_min), size=k, replace=True)
    X_syn = X_min.iloc[take].reset_index(drop=True).copy()
    for c in meta.numeric_cols:
        X_syn[c] = X_syn[c].to_numpy(dtype=np.float64) + rng.normal(0, 0.01, size=k)
    X_syn_rep, row_frac, cell_frac = shared_repair_with_cell_stats(X_syn, X_train, meta)
    y_out = np.concatenate([y_train, np.full(k, mino, dtype=np.int64)])
    X_out = pd.concat([X_train.reset_index(drop=True), X_syn_rep], ignore_index=True)
    return AugmentResult(
        X=X_out,
        y=y_out,
        repair_fraction=row_frac,
        n_synthetic=k,
        method="A3_TOY_MOCK",
        generator_fit_seconds=0.01,
        generator_sample_seconds=0.01,
        scientific_mark="NON_SCIENTIFIC_INTEGRATION_TEST",
        extra={"repair_cell_fraction": cell_frac, "mark": "NON_SCIENTIFIC_INTEGRATION_TEST"},
    )
