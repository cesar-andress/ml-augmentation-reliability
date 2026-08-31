"""Candidate universe builders: TabArena v0.1 ∪ CLIMB ∪ OpenML-CC18."""

from __future__ import annotations

import json
from pathlib import Path

import openml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "artifacts" / "manifests" / "sources"


def load_tabarena_candidates(meta_csv: Path | None = None) -> pd.DataFrame:
    meta_csv = meta_csv or (SOURCES / "tabarena_dataset_metadata.csv")
    meta = pd.read_csv(meta_csv)
    suite = openml.study.get_suite(457)
    suite_ids = set(int(x) for x in suite.data)
    rows = []
    for _, r in meta.iterrows():
        oid = int(r["dataset_id"])
        rows.append(
            {
                "source_pool": "tabarena_v0.1",
                "source_dataset_id": str(oid),
                "resolved_openml_id": oid,
                "dataset_name": str(r.get("dataset_name") or r.get("openml_dataset_name")),
                "in_openml_suite_457": oid in suite_ids,
            }
        )
    present = {r["resolved_openml_id"] for r in rows}
    for oid in sorted(suite_ids - present):
        rows.append(
            {
                "source_pool": "tabarena_v0.1",
                "source_dataset_id": str(oid),
                "resolved_openml_id": oid,
                "dataset_name": f"suite457_{oid}",
                "in_openml_suite_457": True,
            }
        )
    return pd.DataFrame(rows)


def load_climb_candidates(meta_csv: Path | None = None) -> pd.DataFrame:
    meta_csv = meta_csv or (SOURCES / "climb_openml_datainfo.csv")
    meta = pd.read_csv(meta_csv)
    rows = []
    for _, r in meta.iterrows():
        oid = int(r["openml_id"])
        rows.append(
            {
                "source_pool": "climb",
                "source_dataset_id": str(oid),
                "resolved_openml_id": oid,
                "dataset_name": str(r["dataset_name"]),
            }
        )
    return pd.DataFrame(rows)


def load_openml_cc18_candidates(ids_json: Path | None = None) -> pd.DataFrame:
    ids_json = ids_json or (SOURCES / "openml_cc18_suite99_ids.json")
    if ids_json.exists():
        payload = json.loads(ids_json.read_text())
        ids = [int(x) for x in payload["dataset_ids"]]
    else:
        suite = openml.study.get_suite(99)
        ids = sorted(int(x) for x in suite.data)
    return pd.DataFrame(
        [
            {
                "source_pool": "openml_cc18",
                "source_dataset_id": str(oid),
                "resolved_openml_id": oid,
                "dataset_name": f"cc18_{oid}",
            }
            for oid in ids
        ]
    )


def build_candidate_universe() -> pd.DataFrame:
    """v1.1 two-pool universe (backward compatible)."""
    u = pd.concat([load_tabarena_candidates(), load_climb_candidates()], ignore_index=True, sort=False)
    u["candidate_key"] = u["source_pool"] + ":" + u["resolved_openml_id"].astype(str)
    return u


def build_candidate_universe_v1_2() -> pd.DataFrame:
    """Protocol v1.2 three-pool universe. No fourth source."""
    allowed = {"tabarena_v0.1", "climb", "openml_cc18"}
    u = pd.concat(
        [load_tabarena_candidates(), load_climb_candidates(), load_openml_cc18_candidates()],
        ignore_index=True,
        sort=False,
    )
    unexpected = set(u["source_pool"].unique()) - allowed
    if unexpected:
        raise AssertionError(f"fourth/unknown source_pool not permitted: {unexpected}")
    u["candidate_key"] = u["source_pool"] + ":" + u["resolved_openml_id"].astype(str)
    return u


FOURTH_SOURCE_FORBIDDEN = True
