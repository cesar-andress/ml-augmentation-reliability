"""Feature-type audit under the frozen preprocessing contract (deterministic rules)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FEATURE_TYPE_RULES = """
Rules (frozen, reproducible; do not change for convenience):

1. Start from observed pandas Series after OpenML dataframe load.
2. Categorical if ANY of:
   - dtype is pandas CategoricalDtype
   - dtype == object
   - dtype is boolean
3. Otherwise numeric if pandas considers the series numeric (excluding bool).
4. Continuous numeric: numeric AND >2 unique non-null values AND NOT integer-valued
   (integer-valued := all non-null values within 1e-9 of an integer).
5. Integer-valued numeric: numeric AND all non-null values near integers.
   These remain numeric under preprocessing (not ordinal-encoded) unless rule 2 applies.
6. Missingness indicators: one per source feature with any missing value in the audited
   table (TRAIN-fit in experiments; full-table estimate used for screening upper bound).
7. Encoded feature count estimate = n_numeric + n_categorical + missing_indicator_count
   (ordinal encoding keeps one column per categorical feature).
8. If OpenML categorical_indicator disagrees with rules 2–3, record
   dtype_metadata_disagreement and DO NOT silently override; screening uses observed
   rules 2–3, disagreement flagged for review.
"""


@dataclass
class FeatureTypeAudit:
    numeric_cols: list[str]
    categorical_cols: list[str]
    continuous_cols: list[str]
    integer_valued_numeric_cols: list[str]
    missing_indicator_count: int
    n_encoded_features_est: int
    openml_categorical_indicator: list[bool] | None
    dtype_metadata_disagreement: list[dict[str, Any]]
    rules_version: str = "v1_frozen_preprocess_contract"


def audit_feature_types(
    X: pd.DataFrame,
    openml_categorical_indicator: list[bool] | None = None,
) -> FeatureTypeAudit:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    continuous_cols: list[str] = []
    integer_valued_numeric_cols: list[str] = []
    disagreements: list[dict[str, Any]] = []

    for i, col in enumerate(X.columns):
        s = X[col]
        name = str(col)
        observed_cat = (
            isinstance(s.dtype, pd.CategoricalDtype)
            or s.dtype == object
            or pd.api.types.is_bool_dtype(s)
        )
        observed_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

        if openml_categorical_indicator is not None and i < len(openml_categorical_indicator):
            openml_cat = bool(openml_categorical_indicator[i])
            if openml_cat and observed_num and not observed_cat:
                disagreements.append(
                    {
                        "column": name,
                        "openml_categorical": True,
                        "observed": "numeric",
                        "action": "flag_only_use_observed_numeric",
                    }
                )
            if (not openml_cat) and observed_cat:
                disagreements.append(
                    {
                        "column": name,
                        "openml_categorical": False,
                        "observed": "categorical",
                        "action": "flag_only_use_observed_categorical",
                    }
                )

        if observed_cat:
            categorical_cols.append(name)
        elif observed_num:
            numeric_cols.append(name)
            vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            is_int_valued = bool(np.all(np.isclose(vals, np.rint(vals), atol=1e-9)))
            nunique = int(pd.unique(vals).size)
            if is_int_valued:
                integer_valued_numeric_cols.append(name)
            if (not is_int_valued) and nunique > 2:
                continuous_cols.append(name)
        else:
            categorical_cols.append(name)

    miss_count = int(X.isna().any(axis=0).sum())
    n_encoded = len(numeric_cols) + len(categorical_cols) + miss_count
    return FeatureTypeAudit(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        integer_valued_numeric_cols=integer_valued_numeric_cols,
        missing_indicator_count=miss_count,
        n_encoded_features_est=n_encoded,
        openml_categorical_indicator=openml_categorical_indicator,
        dtype_metadata_disagreement=disagreements,
    )
