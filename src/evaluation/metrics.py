"""Evaluation metrics for the smoke protocol."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.calibration.posthoc import clip_prob, logit


def compute_metrics(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_platt: np.ndarray,
    *,
    threshold: float,
    conformal_set_size: float,
    conformal_coverage: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p_raw = clip_prob(p_raw)
    p_platt = clip_prob(p_platt)

    # PRIMARY
    ll_raw = float(log_loss(y_true, np.column_stack([1 - p_raw, p_raw]), labels=[0, 1]))
    ll_platt = float(log_loss(y_true, np.column_stack([1 - p_platt, p_platt]), labels=[0, 1]))

    # SECONDARY
    try:
        auroc = float(roc_auc_score(y_true, p_raw))
    except ValueError:
        auroc = float("nan")

    # calibration slope/intercept via logit(p) ~ y regression on TEST using Platt probs
    z = logit(p_platt).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(z, y_true)
    # Actually scientific calibration slope/intercept typically from:
    # logit(y) ~ a + b logit(p); use logistic? Protocol says calibration slope/intercept —
    # common approach: regress y on logit(p) via logistic, or linear probability.
    # We use logistic regression of y on logit(p): intercept + slope on logit scale.
    from sklearn.linear_model import LogisticRegression

    cal = LogisticRegression(solver="lbfgs")
    cal.fit(z, y_true)
    slope = float(cal.coef_.ravel()[0])
    intercept = float(cal.intercept_.ravel()[0])

    pred = (p_raw >= threshold).astype(int)
    f1 = float(f1_score(y_true, pred, zero_division=0))

    # SUPPLEMENTARY
    brier = float(brier_score_loss(y_true, p_raw))
    try:
        auprc = float(average_precision_score(y_true, p_raw))
    except ValueError:
        auprc = float("nan")

    return {
        "log_loss_raw": ll_raw,
        "log_loss": ll_platt,  # primary reported calibrated
        "log_loss_platt": ll_platt,
        "auroc": auroc,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "conformal_set_size": float(conformal_set_size),
        "conformal_coverage": float(conformal_coverage),
        "f1_tuned": f1,
        "brier": brier,
        "auprc": auprc,
    }
