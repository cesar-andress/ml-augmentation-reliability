#!/usr/bin/env python
"""Re-materialize OpenML 1067 raw cache with freeze-identical writer (engineering)."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.canonical_content_hash import canonical_content_sha256
from src.data.dataset_content_identity import load_identity_manifest
from src.data.openml_loader import (
    cache_raw_openml,
    fetch_openml_dataframe,
    legacy_frozen_parquet_sha256,
)

FROZEN_BYTE = "6fecfff679a8168cdf32dd8a3a01a59ec1e2cbd0d8405141a971599450c5c730"
FORENSIC_ADHOC_CANONICAL = "ff59d4f5c038b8b69cff2fc75e762feecedc1dc980f3b7720b30cfcbcc112b9c"
RAW = ROOT / "data/raw/openml/1067"
ARCHIVE = ROOT / "artifacts/audits/openml_1067_checksum_forensics/pre_patch_cache"


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in ["X.parquet", "y.parquet", "meta.json", "raw_checksum.sha256"]:
        src = RAW / name
        if src.exists():
            shutil.copy2(src, ARCHIVE / name)

    X, y, name, version, target_name, description = fetch_openml_dataframe(1067)
    meta = {
        "openml_id": 1067,
        "version": version,
        "name": name,
        "target": target_name,
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "y_meta": {
            "class_labels": [str(c) for c in pd.Series(y).astype("category").cat.categories],
        },
    }

    digest = cache_raw_openml(1067, X, y, meta, raw_root=ROOT / "data/raw/openml", target_name=target_name)
    canonical = canonical_content_sha256(X, y, target_name=target_name)
    identity = load_identity_manifest(ROOT)[1067]
    expected_canonical = identity["canonical_content_sha256"]

    if digest != FROZEN_BYTE:
        raise SystemExit(f"byte checksum failed: got {digest}, expected {FROZEN_BYTE}")
    if canonical != expected_canonical:
        raise SystemExit(f"canonical v1 failed: got {canonical}, expected {expected_canonical}")

    report = {
        "rematerialized_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_frozen_parquet_sha256": digest,
        "canonical_content_sha256_v1": canonical,
        "forensic_adhoc_canonical_reference": FORENSIC_ADHOC_CANONICAL,
        "y_dtype": str(y.dtype),
        "archive_path": str(ARCHIVE),
    }
    out = ROOT / "artifacts/audits/openml_1067_checksum_forensics/rematerialization_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
