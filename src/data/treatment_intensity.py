"""Treatment intensity r = (1-2p)/p and prevalence helpers (Protocol v1.2)."""

from __future__ import annotations


def treatment_intensity_r(p: float) -> float:
    """r = (1 - 2p) / p. Undefined/non-positive designs when p<=0 or p>=0.5."""
    if p <= 0:
        raise ValueError(f"prevalence must be > 0, got {p}")
    return (1.0 - 2.0 * p) / p


def prevalence_le_040_equiv_r_ge_05(p: float, tol: float = 1e-12) -> bool:
    """Mathematical equivalence: p <= 0.40 iff r >= 0.5 for p in (0, 0.5)."""
    if p <= 0 or p >= 0.5:
        return False
    return (p <= 0.40 + tol) == (treatment_intensity_r(p) >= 0.5 - tol)
