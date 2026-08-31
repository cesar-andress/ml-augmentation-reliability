"""Confirmatory result readers must reject non-CONFIRMATORY rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_STATUS = "CONFIRMATORY"
SMOKE_DIR = Path("results/smoke")
CONFIRMATORY_DIR = Path("results/confirmatory")


class NonConfirmatoryDataError(ValueError):
    """Raised when confirmatory analysis is offered smoke or unmarked rows."""


def assert_confirmatory_frame(df: pd.DataFrame, *, path: str | Path | None = None) -> pd.DataFrame:
    if "scientific_status" not in df.columns:
        raise NonConfirmatoryDataError(
            f"missing scientific_status column{f' in {path}' if path else ''}; "
            "confirmatory analysis refuses unmarked rows"
        )
    bad = df["scientific_status"] != REQUIRED_STATUS
    if bad.any():
        n_bad = int(bad.sum())
        examples = df.loc[bad, "scientific_status"].astype(str).unique()[:5].tolist()
        raise NonConfirmatoryDataError(
            f"{n_bad} row(s) have scientific_status != {REQUIRED_STATUS!r} "
            f"(examples={examples}){f' in {path}' if path else ''}"
        )
    return df


def load_confirmatory_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if SMOKE_DIR.resolve() in path.resolve().parents or path.resolve().parent == SMOKE_DIR.resolve():
        raise NonConfirmatoryDataError(f"refusing smoke path for confirmatory load: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported results format: {path}")
    return assert_confirmatory_frame(df, path=path)
