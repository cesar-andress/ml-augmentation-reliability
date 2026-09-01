"""Binary split conformal prediction."""

from __future__ import annotations

import numpy as np


def conformity_score(p_hat_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s(x,y) = 1 - p_hat(y|x)."""
    p_hat_pos = np.asarray(p_hat_pos, dtype=np.float64)
    y = np.asarray(y).astype(int)
    p_y = np.where(y == 1, p_hat_pos, 1.0 - p_hat_pos)
    return 1.0 - p_y


def conformal_quantile(scores_cal: np.ndarray, alpha: float = 0.10) -> float:
    """Marginal split-conformal quantile on CAL-B."""
    qhat, _meta = conformal_quantile_with_meta(scores_cal, alpha=alpha)
    return qhat


def conformal_quantile_with_meta(scores_cal: np.ndarray, alpha: float = 0.10) -> tuple[float, dict]:
    """Marginal split-conformal quantile plus order-statistic metadata."""
    scores = np.asarray(scores_cal, dtype=np.float64)
    n = len(scores)
    if n == 0:
        raise ValueError("empty CAL-B scores")
    # standard finite-sample correction: level = ceil((n+1)(1-alpha)) / n
    order_stat = int(np.ceil((n + 1) * (1.0 - alpha)))
    level = min(order_stat / n, 1.0)
    qhat = float(np.quantile(scores, level, method="higher"))
    return qhat, {
        "n_calibration": n,
        "alpha": float(alpha),
        "level": float(level),
        "order_statistic_index": min(order_stat, n),
        "score_checksum": float(np.sum(scores)),
    }


def prediction_sets(p_hat_pos: np.ndarray, qhat: float) -> list[set[int]]:
    """Include label y if s(x,y) <= qhat."""
    p = np.asarray(p_hat_pos, dtype=np.float64)
    sets: list[set[int]] = []
    for pi in p:
        s0 = 1.0 - (1.0 - pi)
        s1 = 1.0 - pi
        out: set[int] = set()
        if s0 <= qhat:
            out.add(0)
        if s1 <= qhat:
            out.add(1)
        sets.append(out)
    return sets


def set_metrics(sets: list[set[int]], y_true: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    sizes = np.array([len(s) for s in sets], dtype=np.float64)
    covered = np.array([int(y in s) for y, s in zip(y_true, sets)], dtype=np.float64)
    return {
        "conformal_set_size": float(sizes.mean()),
        "conformal_coverage": float(covered.mean()),
    }
