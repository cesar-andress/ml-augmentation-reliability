#!/usr/bin/env python
"""Build dataset_content_identity_v1.csv for frozen cohort (no modelling)."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.dataset_content_identity import FROZEN_DATASET_IDS, compute_identity_row
from src.data.openml_loader import fetch_openml_dataframe, legacy_frozen_parquet_sha256

FROZEN_CSV = ROOT / "artifacts/manifests/datasets_frozen_v1_2.csv"
OUT_CSV = ROOT / "artifacts/manifests/dataset_content_identity_v1.csv"


def _load_frozen_legacy() -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with FROZEN_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["resolved_openml_id"])] = row
    return out


def main() -> None:
    legacy = _load_frozen_legacy()
    rows = []
    for oid in FROZEN_DATASET_IDS:
        leg = legacy[oid]
        X, y, name, version, target_name, _desc = fetch_openml_dataframe(oid)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            xp, yp = td_path / "X.parquet", td_path / "y.parquet"
            X.to_parquet(xp)
            pd.DataFrame({"y": y}).to_parquet(yp)
            legacy_bytes = legacy_frozen_parquet_sha256(xp, yp)

        expected_legacy = leg["raw_checksum"]
        if legacy_bytes != expected_legacy:
            raise SystemExit(
                f"legacy byte checksum mismatch for {oid}: got {legacy_bytes}, expected {expected_legacy}"
            )

        row = compute_identity_row(
            openml_id=oid,
            openml_version=int(version),
            dataset_name=str(name),
            target_name=str(target_name),
            X=X,
            y=y,
            legacy_frozen_parquet_sha256=legacy_bytes,
        )
        rows.append(row)

    rows.sort(key=lambda r: int(r["resolved_openml_id"]))
    fieldnames = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {OUT_CSV} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
