"""OpenML data loading and eligibility screening (Stage A + Stage B manifest)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd

from src.data.canonical_content_hash import canonical_content_sha256


@dataclass
class DatasetBundle:
    openml_id: int
    name: str
    version: Any
    X: pd.DataFrame
    y: pd.Series
    checksum: str
    description: str
    canonical_content_sha256: str = ""
    target_name: str = ""


def _frame_checksum(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()


def legacy_frozen_parquet_sha256(x_path: str | Path, y_path: str | Path) -> str:
    """Legacy freeze checksum: SHA256(X.parquet bytes || y.parquet bytes)."""
    x_path = Path(x_path)
    y_path = Path(y_path)
    h = hashlib.sha256()
    h.update(x_path.read_bytes())
    h.update(y_path.read_bytes())
    return h.hexdigest()


# Backward-compatible alias used throughout the codebase.
parquet_bytes_checksum = legacy_frozen_parquet_sha256


def fetch_openml_dataframe(openml_id: int) -> tuple[pd.DataFrame, pd.Series, str, Any, str, str]:
    """Download OpenML dataset without preprocessing-oriented dtype coercion."""
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, _categorical_indicator, attribute_names = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=attribute_names)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    description = (ds.description or "")[:2000]
    return (
        X,
        y,
        ds.name,
        getattr(ds, "version", None),
        ds.default_target_attribute,
        description,
    )


def cache_raw_openml(
    openml_id: int,
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict[str, Any] | None = None,
    *,
    raw_root: str | Path | None = None,
    target_name: str = "y",
) -> str:
    """Persist raw parquet cache exactly as Protocol v1.2 freeze did.

    Raw cache preserves OpenML-native dtypes (e.g. bool targets stay bool).
    """
    raw_root = Path(raw_root) if raw_root else Path("data/raw/openml")
    d = raw_root / str(openml_id)
    d.mkdir(parents=True, exist_ok=True)
    x_path = d / "X.parquet"
    y_path = d / "y.parquet"
    X.to_parquet(x_path)
    pd.DataFrame({"y": y}).to_parquet(y_path)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    digest = legacy_frozen_parquet_sha256(x_path, y_path)
    (d / "raw_checksum.sha256").write_text(digest + "\n")
    canonical = canonical_content_sha256(X, y, target_name=target_name)
    (d / "canonical_content_sha256").write_text(canonical + "\n")
    return digest


def load_openml_raw(openml_id: int) -> DatasetBundle:
    X, y_raw, name, version, target_name, desc = fetch_openml_dataframe(openml_id)
    # Screening path may use category semantics for binarization checks.
    y = y_raw.astype("category")
    return DatasetBundle(
        openml_id=int(openml_id),
        name=name,
        version=version,
        X=X,
        y=y,
        checksum=_frame_checksum(X, y),
        description=desc,
        canonical_content_sha256=canonical_content_sha256(X, y_raw, target_name=target_name),
        target_name=target_name,
    )


def load_frozen_openml_raw(
    openml_id: int,
    *,
    expected_raw_checksum: str | None = None,
    expected_canonical_content_sha256: str | None = None,
    expected_version: int | None = None,
    expected_name: str | None = None,
    expected_target_name: str | None = None,
    raw_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    validate_identity: bool = True,
) -> DatasetBundle:
    """Load confirmatory dataset with legacy byte + canonical content identity checks."""
    repo_root = Path(repo_root) if repo_root else Path(".")
    raw_root = Path(raw_root) if raw_root else repo_root / "data" / "raw" / "openml"
    d = raw_root / str(openml_id)
    x_path = d / "X.parquet"
    y_path = d / "y.parquet"

    if x_path.exists() and y_path.exists():
        X = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)["y"]
        legacy_digest = legacy_frozen_parquet_sha256(x_path, y_path)
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        name = meta.get("name")
        version = meta.get("version")
        description = meta.get("description", "")
        target_name = meta.get("target") or meta.get("default_target_attribute") or "y"
        if name is None or version is None:
            ds = openml.datasets.get_dataset(openml_id, download_data=False)
            name = name or ds.name
            version = version if version is not None else getattr(ds, "version", None)
            description = description or (ds.description or "")[:2000]
            if target_name in {"", "from_openml", "y"}:
                target_name = ds.default_target_attribute
    else:
        X, y, name, version, target_name, description = fetch_openml_dataframe(openml_id)
        meta = {
            "openml_id": openml_id,
            "name": name,
            "version": version,
            "description": description,
            "target": target_name,
            "retrieval_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        }
        legacy_digest = cache_raw_openml(
            openml_id,
            X,
            y,
            meta,
            raw_root=raw_root,
            target_name=target_name,
        )

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    canonical = canonical_content_sha256(X, y, target_name=target_name)

    if expected_version is not None and int(version) != int(expected_version):
        raise AssertionError(f"dataset version mismatch: got {version}, expected {expected_version}")
    if expected_name is not None and str(name) != str(expected_name):
        raise AssertionError(f"dataset name mismatch: got {name}, expected {expected_name}")
    if expected_target_name is not None and str(target_name) != str(expected_target_name):
        raise AssertionError(f"target name mismatch: got {target_name}, expected {expected_target_name}")

    if expected_raw_checksum and legacy_digest != expected_raw_checksum:
        raise AssertionError(
            "dataset legacy_frozen_parquet_sha256 mismatch (parquet-bytes): "
            f"got {legacy_digest}, expected {expected_raw_checksum}"
        )
    if expected_canonical_content_sha256 and canonical != expected_canonical_content_sha256:
        raise AssertionError(
            "dataset canonical_content_sha256 mismatch: "
            f"got {canonical}, expected {expected_canonical_content_sha256}"
        )

    if validate_identity and repo_root is not None:
        from src.data.dataset_content_identity import validate_dataset_content_identity

        validate_dataset_content_identity(
            repo_root=repo_root,
            openml_id=int(openml_id),
            openml_version=int(version),
            dataset_name=str(name),
            target_name=str(target_name),
            X=X,
            y=y,
            legacy_parquet_sha256=legacy_digest,
        )

    return DatasetBundle(
        openml_id=int(openml_id),
        name=str(name),
        version=version,
        X=X,
        y=y,
        checksum=legacy_digest,
        description=str(description),
        canonical_content_sha256=canonical,
        target_name=str(target_name),
    )


def binarize_labels(y: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    cats = list(pd.Series(y).astype("category").cat.categories)
    if len(cats) != 2:
        raise ValueError(f"expected binary target, got {len(cats)} classes: {cats}")
    codes = pd.Series(y).astype("category").cat.codes.to_numpy()
    # minority = positive (1)
    counts = np.bincount(codes, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    y_bin = (codes == minority).astype(np.int64)
    meta = {
        "class_labels": [str(c) for c in cats],
        "minority_label": str(cats[minority]),
        "majority_label": str(cats[majority]),
        "minority_code_original": minority,
    }
    return y_bin, meta


def objective_eligibility(X: pd.DataFrame, y_bin: np.ndarray) -> dict[str, Any]:
    n_rows = int(X.shape[0])
    n_features_raw = int(X.shape[1])
    missing_frac = float(X.isna().to_numpy().mean()) if n_rows * n_features_raw else 0.0
    n_minority = int(y_bin.sum())
    prev = n_minority / n_rows if n_rows else 0.0

    # Continuous feature heuristic: numeric dtype and >2 unique non-null values
    cont_mask = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.dropna().nunique()
            cont_mask.append(nunique > 2)
        else:
            cont_mask.append(False)
    n_continuous = int(sum(cont_mask))

    # Encoding estimate: ordinal cats + missingness indicators for features with missing
    n_missing_indicators = int(X.isna().any(axis=0).sum())
    n_features_after_encoding_est = n_features_raw + n_missing_indicators

    checks = {
        "binary": True,  # caller ensures
        "rows_in_range": 500 <= n_rows <= 10_000,
        "features_after_encoding_le_100": n_features_after_encoding_est <= 100,
        "has_continuous": n_continuous >= 1,
        "prevalence_in_range": 0.02 <= prev <= 0.40,
        "minority_count_ge_250": n_minority >= 250,
        "missing_lt_10pct": missing_frac < 0.10,
    }
    return {
        "n_rows": n_rows,
        "n_features_raw": n_features_raw,
        "n_features_after_encoding_est": n_features_after_encoding_est,
        "n_continuous": n_continuous,
        "n_minority": n_minority,
        "minority_prevalence": prev,
        "missing_frac": missing_frac,
        "checks": checks,
        "pass_objective": all(checks.values()),
    }


# Known semantic flags for Stage B — NEVER auto-decide; only surface suspects.
SEMANTIC_SUSPECT_KEYWORDS = {
    "synthetic": ["synthetic", "simulated", "artificially generated", "toy dataset"],
    "fictional": ["fictional", "made-up", "fake patients"],
    "one_vs_rest": ["one-vs-rest", "one vs rest", "ovr binarization", "rest vs"],
    "non_iid": ["time series", "temporal", "longitudinal", "grouped by", "patient id over time"],
}


def suspect_semantic_issues(name: str, description: str) -> list[str]:
    text = f"{name}\n{description}".lower()
    hits = []
    for issue, kws in SEMANTIC_SUSPECT_KEYWORDS.items():
        if any(k in text for k in kws):
            hits.append(issue)
    return hits


def screen_candidates(
    openml_ids: list[int],
    source_pool: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage A objective screen + Stage B semantic-review rows (decision blank)."""
    cand_rows = []
    review_rows = []
    for oid in openml_ids:
        try:
            bundle = load_openml_raw(oid)
            y_bin, y_meta = binarize_labels(bundle.y)
            elig = objective_eligibility(bundle.X, y_bin)
            row = {
                "openml_id": oid,
                "name": bundle.name,
                "source_pool": source_pool,
                "version": bundle.version,
                "checksum": bundle.checksum,
                **{f"obj_{k}": v for k, v in elig.items() if k != "checks"},
                **{f"check_{k}": v for k, v in elig["checks"].items()},
                "pass_objective": elig["pass_objective"],
                "status": "ok",
                "error": "",
            }
            cand_rows.append(row)
            suspects = suspect_semantic_issues(bundle.name, bundle.description)
            if elig["pass_objective"] or suspects:
                review_rows.append(
                    {
                        "openml_id": oid,
                        "name": bundle.name,
                        "source_pool": source_pool,
                        "description_excerpt": bundle.description[:500].replace("\n", " "),
                        "suspected_issue": ";".join(suspects) if suspects else "needs_human_review",
                        "decision": "",
                        "decision_reason": "",
                    }
                )
        except Exception as e:
            cand_rows.append(
                {
                    "openml_id": oid,
                    "name": "",
                    "source_pool": source_pool,
                    "version": None,
                    "checksum": "",
                    "pass_objective": False,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return pd.DataFrame(cand_rows), pd.DataFrame(review_rows)
