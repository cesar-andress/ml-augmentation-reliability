"""Duplicate / variant detection (multi-signal; no auto-dup on dimensions alone)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _norm_name(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def build_duplicate_map(screen_df: pd.DataFrame) -> pd.DataFrame:
    """
    screen_df rows are unique resolved_openml_id after per-ID collapse.
    Required columns: resolved_openml_id, dataset_name, n_rows, n_source_features,
    minority_prevalence, source_pools, raw_checksum, objective_eligible
    """
    rows: list[dict[str, Any]] = []
    eligible = screen_df[screen_df["objective_eligible"] == True].copy()  # noqa: E712
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "removed_openml_id",
                "removed_name",
                "retained_openml_id",
                "retained_name",
                "reason",
                "signals",
            ]
        )

    # Group by normalized name
    eligible["_norm"] = eligible["dataset_name"].map(_norm_name)
    for norm, g in eligible.groupby("_norm"):
        if len(g) < 2 or not norm:
            continue
        # Multi-signal: same/near name already; require at least one more strong signal
        # among checksum equality OR (same n_rows AND same n_features AND close prevalence)
        ids = g["resolved_openml_id"].tolist()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a = g.iloc[i]
                b = g.iloc[j]
                signals = ["near_identical_name"]
                strong = False
                if (
                    pd.notna(a.get("raw_checksum"))
                    and pd.notna(b.get("raw_checksum"))
                    and a["raw_checksum"] == b["raw_checksum"]
                ):
                    signals.append("identical_checksum")
                    strong = True
                same_dim = (
                    int(a["n_rows"]) == int(b["n_rows"])
                    and int(a["n_source_features"]) == int(b["n_source_features"])
                )
                if same_dim:
                    signals.append("identical_dimensions")
                prev_close = abs(float(a["minority_prevalence"]) - float(b["minority_prevalence"])) < 1e-6
                if same_dim and prev_close:
                    signals.append("identical_target_prevalence")
                    strong = True
                # documented dual-pool same ID handled elsewhere
                if not strong:
                    # dimensions alone insufficient
                    continue

                # retention rule
                a_pools = str(a.get("source_pools", ""))
                b_pools = str(b.get("source_pools", ""))
                a_tab = "tabarena_v0.1" in a_pools
                b_tab = "tabarena_v0.1" in b_pools
                if a_tab and not b_tab:
                    retain, remove = a, b
                    reason = "prefer_tabarena_curated_reupload"
                elif b_tab and not a_tab:
                    retain, remove = b, a
                    reason = "prefer_tabarena_curated_reupload"
                elif int(a["n_rows"]) != int(b["n_rows"]):
                    retain, remove = (a, b) if int(a["n_rows"]) > int(b["n_rows"]) else (b, a)
                    reason = "prefer_larger_n"
                else:
                    # tie-break: smaller OpenML ID
                    retain, remove = (a, b) if int(a["resolved_openml_id"]) < int(b["resolved_openml_id"]) else (b, a)
                    reason = "tiebreak_smaller_openml_id"

                rows.append(
                    {
                        "removed_openml_id": int(remove["resolved_openml_id"]),
                        "removed_name": remove["dataset_name"],
                        "retained_openml_id": int(retain["resolved_openml_id"]),
                        "retained_name": retain["dataset_name"],
                        "reason": reason,
                        "signals": ";".join(signals),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates()
