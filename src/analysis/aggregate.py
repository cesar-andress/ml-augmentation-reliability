"""Dataset-level aggregation for frozen confirmatory contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.load_results import AUG_ARMS, BASE_ARM, GBDT, TFM


def per_fold_delta(
    table: pd.DataFrame,
    *,
    metric: str = "log_loss",
) -> pd.DataFrame:
    """Compute Delta_{d,l,a,f} = metric(a) - metric(A0) for a in A1/A2/A3."""
    base = table[table["arm"] == BASE_ARM][
        ["dataset_id", "repeat", "fold", "learner", metric]
    ].rename(columns={metric: "metric_a0"})
    aug = table[table["arm"].isin(AUG_ARMS)][
        ["dataset_id", "repeat", "fold", "learner", "arm", "family", metric]
    ].rename(columns={metric: "metric_arm"})
    merged = aug.merge(base, on=["dataset_id", "repeat", "fold", "learner"], how="inner")
    if len(merged) != 10 * 10 * 4 * 3:  # datasets * folds * learners * arms
        raise AssertionError(f"unexpected fold-delta rows: {len(merged)}")
    merged["delta"] = merged["metric_arm"] - merged["metric_a0"]
    return merged


def bar_delta(fold_deltas: pd.DataFrame) -> pd.DataFrame:
    """Mean over 10 outer folds → barDelta_{d,l,a}."""
    g = (
        fold_deltas.groupby(["dataset_id", "learner", "arm", "family"], as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "bar_delta"})
    )
    if len(g) != 10 * 4 * 3:
        raise AssertionError(f"unexpected bar_delta rows: {len(g)}")
    return g


def dataset_delta(bar: pd.DataFrame) -> pd.DataFrame:
    """Delta_d = mean over 4 learners × 3 arms of barDelta."""
    out = bar.groupby("dataset_id", as_index=False)["bar_delta"].mean().rename(columns={"bar_delta": "Delta_d"})
    if len(out) != 10:
        raise AssertionError(f"expected D=10, got {len(out)}")
    return out.sort_values("dataset_id").reset_index(drop=True)


def dataset_psi(bar: pd.DataFrame) -> pd.DataFrame:
    """Psi_d = mean_TFM(barDelta) - mean_GBDT(barDelta)."""
    rows = []
    for ds, sub in bar.groupby("dataset_id"):
        tfm = sub[sub["learner"].isin(TFM)]["bar_delta"].mean()
        gbdt = sub[sub["learner"].isin(GBDT)]["bar_delta"].mean()
        rows.append({"dataset_id": int(ds), "Psi_d": float(tfm - gbdt), "mean_tfm": float(tfm), "mean_gbdt": float(gbdt)})
    out = pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)
    if len(out) != 10:
        raise AssertionError(f"expected D=10 Psi, got {len(out)}")
    return out


def a0plus_contrast(table: pd.DataFrame, *, metric: str = "log_loss") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory A0+ − A0 for GBDTs only, fold-averaged then dataset-level."""
    base = table[(table["arm"] == BASE_ARM) & (table["learner"].isin(GBDT))][
        ["dataset_id", "repeat", "fold", "learner", metric]
    ].rename(columns={metric: "m0"})
    plus = table[(table["arm"] == "A0+") & (table["learner"].isin(GBDT))][
        ["dataset_id", "repeat", "fold", "learner", metric]
    ].rename(columns={metric: "mplus"})
    m = plus.merge(base, on=["dataset_id", "repeat", "fold", "learner"], how="inner")
    m["delta"] = m["mplus"] - m["m0"]
    by_dl = m.groupby(["dataset_id", "learner"], as_index=False)["delta"].mean()
    by_d = m.groupby("dataset_id", as_index=False)["delta"].mean().rename(columns={"delta": "A0plus_minus_A0"})
    return by_dl, by_d


def descriptive_arm_learner(bar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in AUG_ARMS:
        v = bar.loc[bar["arm"] == arm, "bar_delta"].to_numpy()
        rows.append(_summary_row("arm", arm, v))
    for learner in ("xgboost", "catboost", "tabpfn", "tabicl"):
        v = bar.loc[bar["learner"] == learner, "bar_delta"].to_numpy()
        rows.append(_summary_row("learner", learner, v))
    return pd.DataFrame(rows)


def _summary_row(kind: str, name: str, v: np.ndarray) -> dict:
    return {
        "group": kind,
        "name": name,
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "role": "DESCRIPTIVE",
    }
