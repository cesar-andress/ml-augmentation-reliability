"""Deterministic canonical dataset content identity hashing.

This module implements ``canonical_content_sha256``: a serialization-independent
fingerprint of frozen OpenML raw tabular content (features + target).

Row order is preserved exactly (CV splits depend on it). Columns are hashed in
dataframe column order. No Parquet/Arrow/filesystem metadata participates.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_HASH_VERSION = 1
_NA_TOKEN = b"\xffNA"
_BOOL_FALSE = b"\x00"
_BOOL_TRUE = b"\x01"


def _logical_kind(series: pd.Series) -> str:
    """Classify a column's logical type for canonical serialization."""
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "int"
    if pd.api.types.is_numeric_dtype(series):
        return "float"
    return "string"


def _normalize_target_logical(series: pd.Series) -> tuple[str, pd.Series]:
    """Return (logical_kind, values) for target column."""
    kind = _logical_kind(series)
    if kind == "bool":
        return kind, series
    if kind == "string" or pd.api.types.is_categorical_dtype(series):
        # Binary/category targets: hash string form of each label (stable logical value).
        return "string", series.astype(str)
    if kind in {"int", "float"}:
        return kind, series
    return "string", series.astype(str)


def _write_utf8(hasher: hashlib._Hash, text: str) -> None:
    payload = text.encode("utf-8")
    hasher.update(struct.pack("<I", len(payload)))
    hasher.update(payload)


def _write_value(hasher: hashlib._Hash, value: Any, kind: str) -> None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        hasher.update(_NA_TOKEN)
        return

    if kind == "bool":
        hasher.update(_BOOL_TRUE if bool(value) else _BOOL_FALSE)
        return

    if kind == "int":
        hasher.update(struct.pack("<q", int(value)))
        return

    if kind == "float":
        fv = float(value)
        if not np.isfinite(fv):
            raise ValueError(f"non-finite float in canonical hash: {fv!r}")
        hasher.update(struct.pack("<d", fv))
        return

    # string / category logical values
    _write_utf8(hasher, str(value))


def canonical_content_sha256(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_name: str = "y",
) -> str:
    """Hash ordered tabular content (features then target).

    Algorithm (version 1):
    1. Header: version, target_name, n_rows, n_feature_columns
    2. Feature column names in order, each prefixed with logical kind
    3. For each row in index order 0..n-1, for each feature column in order,
       emit the logically normalized value
    4. Target column name + logical kind
    5. For each row in order, emit target value

    Numeric rules:
    - bool -> single byte 0x00 / 0x01
    - integer dtypes -> signed int64 little-endian
    - other numeric -> IEEE-754 float64 little-endian
    - strings/categories -> UTF-8 length-prefixed (uint32 LE length)
    - missing -> token b\"\\xffNA\"
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if len(X) != len(y):
        raise ValueError(f"row count mismatch: X={len(X)} y={len(y)}")

    hasher = hashlib.sha256()
    hasher.update(struct.pack("<I", CANONICAL_HASH_VERSION))
    _write_utf8(hasher, target_name)
    hasher.update(struct.pack("<Q", len(X)))
    hasher.update(struct.pack("<Q", X.shape[1]))

    feature_kinds: list[str] = []
    for col in X.columns:
        kind = _logical_kind(X[col])
        feature_kinds.append(kind)
        _write_utf8(hasher, str(col))
        _write_utf8(hasher, kind)

    for row_idx in range(len(X)):
        hasher.update(struct.pack("<Q", row_idx))
        for col, kind in zip(X.columns, feature_kinds, strict=True):
            _write_value(hasher, X[col].iloc[row_idx], kind)

    target_kind, y_logical = _normalize_target_logical(y)
    _write_utf8(hasher, target_name)
    _write_utf8(hasher, target_kind)
    for row_idx in range(len(y_logical)):
        hasher.update(struct.pack("<Q", row_idx))
        _write_value(hasher, y_logical.iloc[row_idx], target_kind)

    return hasher.hexdigest()


def feature_name_order_fingerprint(X: pd.DataFrame) -> str:
    """SHA256 of ordered feature names (audit helper)."""
    payload = "|".join(map(str, X.columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_order_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash of per-row content fingerprints in order (audit helper, not canonical)."""
    hasher = hashlib.sha256()
    for i in range(len(X)):
        row_hasher = hashlib.sha256()
        for col in X.columns:
            _write_value(row_hasher, X[col].iloc[i], _logical_kind(X[col]))
        _write_value(row_hasher, y.iloc[i], _normalize_target_logical(y)[0])
        hasher.update(row_hasher.digest())
    return hasher.hexdigest()
