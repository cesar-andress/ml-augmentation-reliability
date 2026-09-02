"""Exact two-sided sign-flip permutation tests (Protocol v1.2).

Implements the frozen confirmatory procedure:
- statistic: one-sample studentized mean (t)
- enumerate all 2^D sign assignments
- confidence intervals by inverting the same test on a grid with
  resolution = observed_vector_range / 1000
"""

from __future__ import annotations

from typing import Any

import numpy as np


def studentized_mean_t(x: np.ndarray) -> float:
    """One-sample studentized mean: mean / (sd / sqrt(n)), sample sd (ddof=1)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError("studentized mean requires n >= 2")
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        if mean == 0.0:
            return 0.0
        return float(np.copysign(np.inf, mean))
    return mean / (sd / np.sqrt(n))


def exact_sign_flip_test(x: np.ndarray) -> dict[str, Any]:
    """Exact two-sided sign-flip test of centre symmetry about zero.

    Enumerates all 2^D sign patterns. p-value =
    (# permutations with |t_perm| >= |t_obs|) / 2^D
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    d = int(x.size)
    if d < 1:
        raise ValueError("empty observation vector")
    n_perm = 1 << d
    t_obs = studentized_mean_t(x)
    abs_obs = abs(t_obs)
    count = 0
    # Iterate all bitmasks; bit i selects sign for observation i
    for mask in range(n_perm):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(d)], dtype=np.float64)
        # Convention: bit=1 keeps +, bit=0 flips — both cover all patterns
        t_perm = studentized_mean_t(x * signs)
        if abs(t_perm) >= abs_obs - 1e-15:
            count += 1
    p = count / n_perm
    n_pos = int(np.sum(x > 0))
    n_neg = int(np.sum(x < 0))
    n_zero = int(np.sum(x == 0))
    return {
        "D": d,
        "n_permutations": n_perm,
        "statistic": float(t_obs),
        "p_value": float(p),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "values": x.tolist(),
    }


def exact_sign_flip_ci(
    x: np.ndarray,
    *,
    level: float = 0.95,
) -> dict[str, Any]:
    """Invert the exact sign-flip test for a (1-alpha) confidence interval.

    Grid resolution = (max(x) - min(x)) / 1000 as frozen in
    statistical_analysis_v1_2.yaml. If the range is zero, return a
    degenerate interval at the common value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    alpha = 1.0 - level
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    span = xmax - xmin
    if span == 0.0:
        v = float(x[0])
        return {
            "level": level,
            "method": "invert_exact_sign_flip_test",
            "lower": v,
            "upper": v,
            "grid_points": 1,
            "grid_step": 0.0,
            "observed_range": 0.0,
        }

    step = span / 1000.0
    # Search theta such that shifted vector x - theta is compatible with H0
    # at two-sided level alpha (p >= alpha).
    # Expand beyond [xmin, xmax] slightly so the interval can contain the mean.
    # Practical search window: [xmin - span, xmax + span] with same step size.
    thetas = np.arange(xmin - span, xmax + span + 0.5 * step, step, dtype=np.float64)
    accepted: list[float] = []
    for theta in thetas:
        shifted = x - theta
        # Fast reject using mean-only bound is unsafe; use full test.
        # For speed at D=10 (1024 perms × ~3000 thetas): ~3e6 ops — fine.
        res = exact_sign_flip_test(shifted)
        if res["p_value"] >= alpha - 1e-15:
            accepted.append(float(theta))
    if not accepted:
        # Fallback: report observed mean as point (should not happen for D=10)
        m = float(np.mean(x))
        return {
            "level": level,
            "method": "invert_exact_sign_flip_test",
            "lower": m,
            "upper": m,
            "grid_points": int(thetas.size),
            "grid_step": step,
            "observed_range": span,
            "warning": "no_theta_accepted",
        }
    return {
        "level": level,
        "method": "invert_exact_sign_flip_test",
        "lower": float(min(accepted)),
        "upper": float(max(accepted)),
        "grid_points": int(thetas.size),
        "grid_step": step,
        "observed_range": span,
        "n_accepted": len(accepted),
    }


def conventional_t_interval(x: np.ndarray, *, level: float = 0.95) -> dict[str, float]:
    """Supplementary conventional Student-t CI (not confirmatory)."""
    from scipy import stats

    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if n < 2 or sd == 0.0:
        return {"level": level, "lower": mean, "upper": mean, "mean": mean}
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df=n - 1))
    return {
        "level": level,
        "lower": mean - tcrit * se,
        "upper": mean + tcrit * se,
        "mean": mean,
    }
