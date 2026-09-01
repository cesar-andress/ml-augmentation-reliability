"""OpenML data loading and eligibility screening (Stage A + Stage B manifest)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd


@dataclass
class DatasetBundle:
    openml_id: int
    name: str
    version: Any
    X: pd.DataFrame
    y: pd.Series
    checksum: str
    description: str


def _frame_checksum(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()


def parquet_bytes_checksum(x_path: str | Path, y_path: str | Path) -> str:
    """Freeze-compatible checksum: SHA256(X.parquet bytes || y.parquet bytes)."""
    from pathlib import Path

    x_path = Path(x_path)
    y_path = Path(y_path)
    h = hashlib.sha256()
    h.update(x_path.read_bytes())
    h.update(y_path.read_bytes())
    return h.hexdigest()


def cache_raw_openml(
    openml_id: int,
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict[str, Any] | None = None,
    *,
    raw_root: str | Path | None = None,
) -> str:
    """Persist raw parquet cache exactly as Protocol v1.2 freeze did."""
    from pathlib import Path
    import json

    raw_root = Path(raw_root) if raw_root else Path("data/raw/openml")
    d = raw_root / str(openml_id)
    d.mkdir(parents=True, exist_ok=True)
    x_path = d / "X.parquet"
    y_path = d / "y.parquet"
    X.to_parquet(x_path)
    pd.DataFrame({"y": y}).to_parquet(y_path)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    digest = parquet_bytes_checksum(x_path, y_path)
    (d / "raw_checksum.sha256").write_text(digest + "\n")
    return digest


def load_openml_raw(openml_id: int) -> DatasetBundle:
    ds = openml.datasets.get_dataset(openml_id, download_data=True)
    X, y, categorical_indicator, attribute_names = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=attribute_names)
    y = pd.Series(y)
    # Map binary labels to {0,1} with minority as positive if needed later handled by caller.
    # Keep original categories but ensure two classes.
    y = y.astype("category")
    desc = (ds.description or "")[:2000]
    return DatasetBundle(
        openml_id=int(openml_id),
        name=ds.name,
        version=getattr(ds, "version", None),
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        checksum=_frame_checksum(X, y),
        description=desc,
    )


def load_frozen_openml_raw(
    openml_id: int,
    *,
    expected_raw_checksum: str | None = None,
    expected_version: int | None = None,
    raw_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> DatasetBundle:
    """Load confirmatory dataset using freeze-compatible parquet-byte checksum.

    Prefer existing data/raw/openml/<id>/{X,y}.parquet when present.
    Otherwise download, cache with the freeze writer, and verify checksum.
    """
    from pathlib import Path

    repo_root = Path(repo_root) if repo_root else Path(".")
    raw_root = Path(raw_root) if raw_root else repo_root / "data" / "raw" / "openml"
    d = raw_root / str(openml_id)
    x_path = d / "X.parquet"
    y_path = d / "y.parquet"

    retrieval = "cache"
    if x_path.exists() and y_path.exists():
        X = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)["y"]
        digest = parquet_bytes_checksum(x_path, y_path)
        # recover metadata
        import json

        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        name = meta.get("name")
        version = meta.get("version")
        description = meta.get("description", "")
        if name is None or version is None:
            ds = openml.datasets.get_dataset(openml_id, download_data=False)
            name = name or ds.name
            version = version if version is not None else getattr(ds, "version", None)
            description = description or (ds.description or "")[:2000]
    else:
        retrieval = "openml_download_then_cache"
        bundle = load_openml_raw(openml_id)
        meta = {
            "openml_id": openml_id,
            "name": bundle.name,
            "version": bundle.version,
            "description": bundle.description,
            "default_target_attribute": "from_openml",
        }
        digest = cache_raw_openml(openml_id, bundle.X, bundle.y, meta, raw_root=raw_root)
        X, y = bundle.X, bundle.y
        name, version, description = bundle.name, bundle.version, bundle.description

    if expected_raw_checksum and digest != expected_raw_checksum:
        raise AssertionError(
            f"dataset raw_checksum mismatch (parquet-bytes): got {digest}, expected {expected_raw_checksum}"
        )
    if expected_version is not None and int(version) != int(expected_version):
        raise AssertionError(f"dataset version mismatch: got {version}, expected {expected_version}")

    return DatasetBundle(
        openml_id=int(openml_id),
        name=str(name),
        version=version,
        X=X.reset_index(drop=True),
        y=pd.Series(y).reset_index(drop=True).astype("category"),
        checksum=digest,
        description=str(description),
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
            # Always include objective-pass datasets for human semantic review;
            # also include any with keyword hits even if objective fail.
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
