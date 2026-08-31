"""Threshold tuning on CAL-A raw probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def tune_threshold_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    """Maximize F1 over unique predicted probabilities; ties -> smallest threshold."""
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=np.float64)
    candidates = np.unique(p)
    best_t = float(candidates.min()) if len(candidates) else 0.5
    best_f1 = -1.0
    for t in np.sort(candidates):
        pred = (p >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1 or (np.isclose(f1, best_f1) and t < best_t):
            best_f1 = float(f1)
            best_t = float(t)
    return best_t
