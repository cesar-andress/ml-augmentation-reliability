"""Frozen preprocessing contract: TRAIN-fit only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreprocessMeta:
    numeric_cols: list[str]
    categorical_cols: list[str]
    missing_indicator_cols: list[str]
    numeric_medians: dict[str, float]
    categorical_modes: dict[str, Any]
    category_levels: dict[str, list[Any]]
    unknown_category_sentinel: int
    output_columns: list[str]
    feature_types: dict[str, str] = field(default_factory=dict)


@dataclass
class FittedPreprocessor:
    meta: PreprocessMeta

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return transform(X, self.meta)


def _is_categorical(s: pd.Series) -> bool:
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    return True


def fit_preprocessor(X_train: pd.DataFrame, unknown_category_sentinel: int = -1) -> FittedPreprocessor:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for c in X_train.columns:
        if _is_categorical(X_train[c]):
            categorical_cols.append(str(c))
        else:
            numeric_cols.append(str(c))

    # Missingness indicators determined BEFORE imputation, based on TRAIN presence of missing
    missing_source_cols = [str(c) for c in X_train.columns if X_train[c].isna().any()]
    missing_indicator_cols = [f"__miss__{c}" for c in missing_source_cols]

    numeric_medians: dict[str, float] = {}
    for c in numeric_cols:
        med = float(pd.to_numeric(X_train[c], errors="coerce").median())
        if np.isnan(med):
            med = 0.0
        numeric_medians[c] = med

    categorical_modes: dict[str, Any] = {}
    category_levels: dict[str, list[Any]] = {}
    for c in categorical_cols:
        s = X_train[c]
        mode = s.mode(dropna=True)
        mode_val = mode.iloc[0] if len(mode) else "__MISSING_MODE__"
        categorical_modes[c] = mode_val
        levels = list(pd.Series(s.dropna().unique()).tolist())
        # stable order
        levels = sorted(levels, key=lambda x: str(x))
        category_levels[c] = levels
        # sentinel must be outside TRAIN category set codes [0..K-1]
        if unknown_category_sentinel >= 0 and unknown_category_sentinel < len(levels):
            raise ValueError(
                f"unknown_category_sentinel={unknown_category_sentinel} collides with TRAIN levels for {c}"
            )

    output_columns = (
        numeric_cols
        + categorical_cols
        + missing_indicator_cols
    )
    feature_types = {c: "numeric" for c in numeric_cols}
    feature_types.update({c: "categorical" for c in categorical_cols})
    feature_types.update({c: "missing_indicator" for c in missing_indicator_cols})

    meta = PreprocessMeta(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        missing_indicator_cols=missing_indicator_cols,
        numeric_medians=numeric_medians,
        categorical_modes=categorical_modes,
        category_levels=category_levels,
        unknown_category_sentinel=int(unknown_category_sentinel),
        output_columns=output_columns,
        feature_types=feature_types,
    )
    return FittedPreprocessor(meta=meta)


def transform(X: pd.DataFrame, meta: PreprocessMeta) -> pd.DataFrame:
    pieces: dict[str, np.ndarray] = {}

    # Missing indicators first conceptually (values from pre-imputation X)
    for col, ind_name in zip(
        [c.replace("__miss__", "", 1) for c in meta.missing_indicator_cols],
        meta.missing_indicator_cols,
        strict=False,
    ):
        # recover source col from indicator name
        src = ind_name[len("__miss__") :]
        if src in X.columns:
            pieces[ind_name] = X[src].isna().astype(np.int64).to_numpy()
        else:
            pieces[ind_name] = np.zeros(len(X), dtype=np.int64)

    for c in meta.numeric_cols:
        s = pd.to_numeric(X[c], errors="coerce") if c in X.columns else pd.Series(np.nan, index=X.index)
        s = s.fillna(meta.numeric_medians[c]).astype(np.float64)
        pieces[c] = s.to_numpy(dtype=np.float64)

    for c in meta.categorical_cols:
        if c in X.columns:
            s = X[c].copy()
        else:
            s = pd.Series([np.nan] * len(X))
        s = s.where(s.notna(), meta.categorical_modes[c])
        levels = meta.category_levels[c]
        level_to_code = {lv: i for i, lv in enumerate(levels)}
        codes = []
        for v in s.tolist():
            codes.append(level_to_code.get(v, meta.unknown_category_sentinel))
        pieces[c] = np.asarray(codes, dtype=np.int64)

    out = pd.DataFrame({c: pieces[c] for c in meta.output_columns})
    if out.isna().any().any():
        raise AssertionError("NaNs remain after preprocessing")
    return out


def categorical_feature_indices(meta: PreprocessMeta) -> list[int]:
    return [meta.output_columns.index(c) for c in meta.categorical_cols]
