"""Post-hoc calibration from stored probabilities (no CalibratedClassifierCV)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def clip_prob(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = clip_prob(p, eps)
    return np.log(p) - np.log(1.0 - p)


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, p_cala: np.ndarray, y_cala: np.ndarray) -> "PlattCalibrator":
        z = logit(p_cala).reshape(-1, 1)
        self.model.fit(z, y_cala)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = logit(p).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1].astype(np.float64)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p_cala: np.ndarray, y_cala: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(clip_prob(p_cala), y_cala)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.model.transform(clip_prob(p)).astype(np.float64)
