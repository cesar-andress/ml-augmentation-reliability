#!/usr/bin/env python
"""Protocol v1.2 outcome-blind full rescreen + optional cohort freeze.

Does NOT train models. Does NOT read smoke metrics.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openml
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.binarization_exclusion import is_artificial_binarization
from src.data.candidates import build_candidate_universe_v1_2
from src.data.duplicates import build_duplicate_map
from src.data.feature_audit import audit_feature_types
from src.data.objective_filters import evaluate_objective
from src.data.openml_loader import binarize_labels
from src.data.semantic_review import propose_semantic_row, scan_description
from src.data.treatment_intensity import treatment_intensity_r
from src.logging_utils import write_json

MANIFESTS = ROOT / "artifacts" / "manifests"
RAW = ROOT / "data" / "raw" / "openml"
PROTOCOL = ROOT / "artifacts" / "protocol"
SPLIT_SEED = 42
N_SPLITS = 5
N_REPEATS = 2
PREVALENCE_UPPER = 0.40  # v1.2 restored
MIN_D = 10

# Human decisions carried forward
HUMAN = {
    44232: {
        "decision": "EXCLUDE",
        "decision_reason": (
            "Probable subset/variant of the 5,000-row classic telecom churn dataset; "
            "excluded conservatively to preserve dataset-level independence."
        ),
        "review_status": "HUMAN_RESOLVED_V1_1",
        "suspected_issue": "A_duplicate_derivative",
    },
    1489: {
        "decision": "KEEP",
        "decision_reason": (
            "ELENA phoneme representation contains no usable speaker/group identifier; "
            "conventionally evaluated as IID in tabular benchmarking."
        ),
        "review_status": "METHODOLOGICAL_KEEP_RECORDED",
        "suspected_issue": "",
    },
}


def cache_raw(openml_id: int, X: pd.DataFrame, y: pd.Series, meta: dict) -> str:
    d = RAW / str(openml_id)
    d.mkdir(parents=True, exist_ok=True)
    x_path = d / "X.parquet"
    y_path = d / "y.parquet"
    X.to_parquet(x_path)
    pd.DataFrame({"y": y}).to_parquet(y_path)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    h = hashlib.sha256()
    h.update(x_path.read_bytes())
    h.update(y_path.read_bytes())
    digest = h.hexdigest()
    (d / "raw_checksum.sha256").write_text(digest + "\n")
    return digest


def screen_one(oid: int, provenance: pd.DataFrame) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    source_pools = sorted(provenance["source_pool"].unique().tolist())
    names = provenance["dataset_name"].dropna().astype(str).tolist()
    row = {
        "candidate_id": f"openml:{oid}",
        "resolved_openml_id": oid,
        "source_pools": ";".join(source_pools),
        "source_dataset_ids": ";".join(f"{r.source_pool}:{r.source_dataset_id}" for r in provenance.itertuples()),
        "dataset_name": names[0] if names else str(oid),
        "retrieval_timestamp_utc": ts,
        "objective_eligible": False,
        "semantic_eligible": False,
        "duplicate_status": "none",
        "first_exclusion_reason": "",
        "final_decision": "PENDING",
    }

    def exclude(reason: str, **extra):
        row.update(extra)
        if not row["first_exclusion_reason"]:
            row["first_exclusion_reason"] = reason
        row["final_decision"] = "EXCLUDE"
        row["objective_eligible"] = False
        row["semantic_eligible"] = False
        return row

    try:
        ds = openml.datasets.get_dataset(oid, download_data=False, download_qualities=True)
        row["openml_version"] = ds.version
        row["dataset_name"] = ds.name
        row["target_name"] = ds.default_target_attribute
        desc = ds.description or ""
        row["description_excerpt"] = desc[:800]

        art, art_reason = is_artificial_binarization(
            description=desc, default_target_attribute=ds.default_target_attribute
        )
        if art:
            return exclude(f"artificial_binarization:{art_reason}", openml_version=ds.version)

        q = ds.qualities or {}
        n_q = q.get("NumberOfInstances")
        n_cls = q.get("NumberOfClasses")
        if n_q is not None and (n_q < 500 or n_q > 10_000):
            return exclude(f"prefilter_rows_out_of_range:{n_q}", n_rows=n_q)
        if n_cls is not None and n_cls != 2:
            return exclude(f"prefilter_not_binary:{n_cls}")

        ds = openml.datasets.get_dataset(oid, download_data=True, download_qualities=True)
        X, y, categorical_indicator, attribute_names = ds.get_data(
            target=ds.default_target_attribute, dataset_format="dataframe"
        )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=attribute_names)
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Re-check binarization on loaded metadata
        art, art_reason = is_artificial_binarization(
            description=ds.description or "", default_target_attribute=ds.default_target_attribute
        )
        if art:
            return exclude(f"artificial_binarization:{art_reason}")

        n_unique = int(y.nunique(dropna=True))
        if n_unique != 2:
            return exclude(f"not_binary_observed:{n_unique}", n_rows=len(X), n_source_features=X.shape[1])

        y_bin, y_meta = binarize_labels(y)
        audit = audit_feature_types(
            X,
            openml_categorical_indicator=list(categorical_indicator)
            if categorical_indicator is not None
            else None,
        )
        missing_frac = float(X.isna().to_numpy().mean()) if X.size else 0.0
        n_rows = int(len(X))
        n_min = int(y_bin.sum())
        p = float(n_min / n_rows) if n_rows else 0.0

        # Structural zero treatment (exactly balanced)
        if abs(p - 0.5) < 1e-12 or (y_bin == 0).sum() == (y_bin == 1).sum():
            row.update(
                {
                    "n_rows": n_rows,
                    "n_source_features": int(X.shape[1]),
                    "n_encoded_features": audit.n_encoded_features_est,
                    "n_continuous": len(audit.continuous_cols),
                    "n_categorical": len(audit.categorical_cols),
                    "n_minority": n_min,
                    "n_majority": n_rows - n_min,
                    "minority_prevalence": p,
                    "treatment_intensity_r": 0.0,
                    "missing_fraction": missing_frac,
                    "minority_label": y_meta["minority_label"],
                    "majority_label": y_meta["majority_label"],
                    "openml_version": ds.version,
                    "pipeline_determinism_candidate": True,
                }
            )
            return exclude(
                "Structural zero treatment: minority prevalence=0.5 implies k=N_majority-N_minority=0."
            )

        try:
            r = treatment_intensity_r(p)
        except ValueError:
            return exclude(f"invalid_prevalence_for_r:{p}")

        obj = evaluate_objective(
            y_bin=y_bin,
            audit=audit,
            n_rows=n_rows,
            missing_frac=missing_frac,
            prevalence_upper=PREVALENCE_UPPER,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            seed=SPLIT_SEED,
        )
        # Explicit r check (equivalent to p<=0.40)
        checks = dict(obj.checks)
        checks["treatment_intensity_r_ge_0_5"] = r >= 0.5
        pass_obj = all(checks.values())

        checksum = cache_raw(
            oid,
            X,
            y,
            {
                "openml_id": oid,
                "version": ds.version,
                "name": ds.name,
                "target": ds.default_target_attribute,
                "retrieval_timestamp_utc": ts,
                "y_meta": y_meta,
            },
        )

        row.update(
            {
                "openml_version": ds.version,
                "dataset_name": ds.name,
                "target_name": ds.default_target_attribute,
                "description_excerpt": (ds.description or "")[:800],
                "class_labels": "|".join(y_meta["class_labels"]),
                "minority_label": y_meta["minority_label"],
                "majority_label": y_meta["majority_label"],
                "n_rows": n_rows,
                "n_source_features": int(X.shape[1]),
                "n_numeric": len(audit.numeric_cols),
                "n_categorical": len(audit.categorical_cols),
                "n_continuous": len(audit.continuous_cols),
                "n_encoded_features": audit.n_encoded_features_est,
                "missing_indicator_count": audit.missing_indicator_count,
                "n_minority": n_min,
                "n_majority": n_rows - n_min,
                "minority_prevalence": p,
                "treatment_intensity_r": r,
                "missing_fraction": missing_frac,
                "max_balanced_train_size": obj.max_balanced_train_size,
                "raw_checksum": checksum,
                "objective_eligible": bool(pass_obj),
                "objective_checks_json": json.dumps(checks),
                "dtype_disagreement": bool(audit.dtype_metadata_disagreement),
            }
        )
        for k, v in checks.items():
            row[f"check_{k}"] = v

        if not pass_obj:
            failed = [k for k, v in checks.items() if not v]
            return exclude("objective_fail:" + ",".join(failed))

        # Semantic keyword scan (no LLM)
        hits = scan_description(desc)
        hit_codes = {h[0] for h in hits}
        strong = hit_codes & {
            "C_simulated_or_synthetic",
            "D_fictional",
            "B_artificial_ovr",
            "E_non_iid",
            "F_repeated_measurements",
            "G_grouped_observations",
            "H_time_series_temporal",
            "I_subject_leakage_risk",
        }

        # Human overrides
        if oid in HUMAN:
            h = HUMAN[oid]
            row["semantic_eligible"] = h["decision"] == "KEEP"
            row["semantic_decision"] = h["decision"]
            row["semantic_decision_reason"] = h["decision_reason"]
            row["review_status"] = h["review_status"]
            row["suspected_issue"] = h.get("suspected_issue", "")
            if h["decision"] == "EXCLUDE":
                return exclude("human_exclude:" + h["decision_reason"][:120])
            row["final_decision"] = "KEEP"
            row["semantic_eligible"] = True
            return row

        if strong:
            row["review_status"] = "HUMAN_REVIEW_REQUIRED"
            row["suspected_issue"] = ";".join(sorted(strong))
            row["evidence"] = " | ".join(f"{c}:{s}" for c, s in hits)[:2000]
            row["proposed_decision"] = "EXCLUDE"
            row["semantic_decision"] = ""
            row["final_decision"] = "HUMAN_REVIEW_REQUIRED"
            row["semantic_eligible"] = False
            if not row["first_exclusion_reason"]:
                row["first_exclusion_reason"] = "semantic_flag:" + row["suspected_issue"]
            return row

        # Thin description -> human review for NEW pools especially CC18
        if len(desc.strip()) < 40:
            row["review_status"] = "HUMAN_REVIEW_REQUIRED"
            row["suspected_issue"] = "J_unclear_provenance"
            row["evidence"] = "description_too_short_or_missing"
            row["proposed_decision"] = "KEEP"
            row["semantic_decision"] = ""
            row["final_decision"] = "HUMAN_REVIEW_REQUIRED"
            row["semantic_eligible"] = False
            row["first_exclusion_reason"] = "semantic_flag:J_unclear_provenance"
            return row

        # Clear keep
        if "tabarena_v0.1" in source_pools:
            reason = "tabarena_v0.1_curated_iid_no_semantic_keyword_hits"
        elif "openml_cc18" in source_pools:
            reason = "openml_cc18_curated_benchmark_no_semantic_keyword_hits"
        else:
            reason = "climb_official_list_no_semantic_keyword_hits"
        row["review_status"] = "PROPOSED_KEEP_METADATA_CLEAR"
        row["semantic_decision"] = "KEEP"
        row["semantic_decision_reason"] = reason
        row["semantic_eligible"] = True
        row["final_decision"] = "KEEP"
        return row

    except Exception as e:
        return exclude(f"exception:{type(e).__name__}:{e}", traceback=traceback.format_exc()[-1500:])


def apply_duplicates(ledger: pd.DataFrame) -> pd.DataFrame:
    """TabArena precedence + duplicate map; never silently drop — mark decisions."""
    elig = ledger[ledger["final_decision"] == "KEEP"].copy()
    # Adapt columns for existing duplicate builder
    screen = elig.rename(columns={})
    screen["objective_eligible"] = True
    screen["source_pools"] = screen["source_pools"]
    dup = build_duplicate_map(screen)
    # Always prefer TabArena when same checksum/name strong signals already handled.
    # Additionally: if same normalized name across pools and TabArena present, prefer TabArena
    # only when strong signals exist (checksum or dim+prev) — already in build_duplicate_map.

    # Human mapping 44232->46915 already EXCLUDE on 44232
    extra = pd.DataFrame(
        [
            {
                "removed_openml_id": 44232,
                "removed_name": "UCI_churn",
                "retained_openml_id": 46915,
                "retained_name": "churn",
                "reason": "prefer_tabarena_curated_reupload_human_confirmed_subset_variant",
                "signals": "human_review_v1_1",
            }
        ]
    )
    if dup.empty:
        dup = extra
    else:
        dup = pd.concat([dup, extra], ignore_index=True).drop_duplicates()
    dup.to_csv(MANIFESTS / "dataset_duplicate_map_v1_2.csv", index=False)

    removed = set(int(x) for x in dup["removed_openml_id"].tolist())
    for oid in removed:
        mask = ledger["resolved_openml_id"] == oid
        if (ledger.loc[mask, "final_decision"] == "KEEP").any():
            ledger.loc[mask, "duplicate_status"] = "removed_duplicate_variant"
            ledger.loc[mask, "final_decision"] = "EXCLUDE"
            ledger.loc[mask, "semantic_eligible"] = False
            if not ledger.loc[mask, "first_exclusion_reason"].astype(str).iloc[0]:
                counterpart = dup.loc[dup["removed_openml_id"] == oid, "retained_openml_id"]
                ret = int(counterpart.iloc[0]) if len(counterpart) else None
                ledger.loc[mask, "first_exclusion_reason"] = f"duplicate_of:{ret}"
            else:
                ledger.loc[mask, "first_exclusion_reason"] = ledger.loc[mask, "first_exclusion_reason"].astype(str) + f"|duplicate"
    return ledger


def write_determinism_manifest(ledger: pd.DataFrame) -> None:
    m = ledger[
        ledger["first_exclusion_reason"].astype(str).str.contains("Structural zero treatment", na=False)
        | (ledger["resolved_openml_id"] == 46930)
    ][
        [
            c
            for c in [
                "resolved_openml_id",
                "dataset_name",
                "openml_version",
                "n_rows",
                "n_minority",
                "n_majority",
                "minority_prevalence",
                "first_exclusion_reason",
                "raw_checksum",
            ]
            if c in ledger.columns
        ]
    ].drop_duplicates(subset=["resolved_openml_id"])
    # Ensure 46930 present
    if 46930 not in set(m["resolved_openml_id"].tolist() if len(m) else []):
        # try from ledger row even if columns sparse
        sub = ledger[ledger["resolved_openml_id"] == 46930]
        if len(sub):
            m = pd.concat([m, sub[m.columns.intersection(sub.columns)]], ignore_index=True)
    m = m.copy()
    m["exclusion_reason"] = (
        "Structural zero treatment: minority prevalence=0.5 implies k=N_majority-N_minority=0."
    )
    m["inferential_cohort"] = False
    m["determinism_assertion_deferred"] = True
    m.to_csv(MANIFESTS / "pipeline_determinism_dataset.csv", index=False)


def freeze_if_ready(ledger: pd.DataFrame) -> tuple[str, Path | None, str | None]:
    human = ledger[ledger["final_decision"] == "HUMAN_REVIEW_REQUIRED"]
    if len(human):
        # write review package
        lines = [
            "# DATASET_REVIEW_V1_2 — unresolved human semantic decisions",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "No model outcomes included.",
            "",
        ]
        for _, r in human.drop_duplicates("resolved_openml_id").sort_values("resolved_openml_id").iterrows():
            lines += [
                f"## OpenML {int(r.resolved_openml_id)} — {r.dataset_name}",
                f"- source_pools: {r.source_pools}",
                f"- suspected_issue: {r.get('suspected_issue','')}",
                f"- evidence: {r.get('evidence','')}",
                f"- proposed: {r.get('proposed_decision','')}",
                "- decision: _(blank)_",
                "",
            ]
        (MANIFESTS / "DATASET_REVIEW_V1_2.md").write_text("\n".join(lines))
        return "HUMAN_REVIEW_REQUIRED", None, None

    keep = ledger[ledger["final_decision"] == "KEEP"].copy()
    keep = keep.drop_duplicates(subset=["resolved_openml_id"]).sort_values("resolved_openml_id")
    d = len(keep)
    if d < MIN_D:
        write_json(
            MANIFESTS / "dataset_freeze_status_v1_2.json",
            {
                "outcome": "DATASET_COHORT_FINAL_NO_GO",
                "D": d,
                "min_D": MIN_D,
                "ids": keep["resolved_openml_id"].tolist(),
            },
        )
        return "FINAL_NO_GO", None, None

    rows = []
    for _, r in keep.iterrows():
        rows.append(
            {
                "cohort_version": "v1.2",
                "resolved_openml_id": int(r["resolved_openml_id"]),
                "openml_version": r.get("openml_version"),
                "dataset_name": r.get("dataset_name"),
                "source_pools": r.get("source_pools"),
                "n_rows": int(r["n_rows"]),
                "n_source_features": int(r["n_source_features"]),
                "n_encoded_features": int(r["n_encoded_features"]),
                "n_continuous": int(r["n_continuous"]),
                "n_categorical": int(r["n_categorical"]),
                "minority_label": r.get("minority_label"),
                "majority_label": r.get("majority_label"),
                "n_minority": int(r["n_minority"]),
                "n_majority": int(r["n_majority"]),
                "minority_prevalence": float(r["minority_prevalence"]),
                "treatment_intensity_r": float(r["treatment_intensity_r"]),
                "missing_fraction": float(r["missing_fraction"]),
                "raw_checksum": r.get("raw_checksum"),
                "semantic_decision": r.get("semantic_decision"),
                "semantic_decision_reason": r.get("semantic_decision_reason"),
            }
        )
    frozen = pd.DataFrame(rows).sort_values("resolved_openml_id")
    out = MANIFESTS / "datasets_frozen_v1_2.csv"
    csv_bytes = frozen.to_csv(index=False).encode("utf-8")
    out.write_bytes(csv_bytes)
    digest = hashlib.sha256(csv_bytes).hexdigest()
    (MANIFESTS / "datasets_frozen_v1_2.sha256").write_text(digest + "\n")
    return "GO", out, digest


def main() -> int:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)

    universe = build_candidate_universe_v1_2()
    # enrich openml_version later per id; write provenance universe
    u_out = universe.copy()
    u_out["openml_version"] = None
    u_out.to_csv(MANIFESTS / "dataset_candidate_universe_v1_2.csv", index=False)

    unique_ids = sorted(universe["resolved_openml_id"].unique().tolist())
    rows = []
    for i, oid in enumerate(unique_ids):
        prov = universe[universe["resolved_openml_id"] == oid]
        print(f"[{i+1}/{len(unique_ids)}] v1.2 screen {oid}", flush=True)
        rows.append(screen_one(oid, prov))

    ledger = pd.DataFrame(rows)
    # fill universe versions from ledger
    ver_map = dict(zip(ledger["resolved_openml_id"], ledger.get("openml_version", pd.Series(dtype=object))))
    u_out["openml_version"] = u_out["resolved_openml_id"].map(ver_map)
    # better names
    name_map = dict(zip(ledger["resolved_openml_id"], ledger["dataset_name"]))
    u_out["dataset_name"] = u_out["resolved_openml_id"].map(name_map).fillna(u_out["dataset_name"])
    u_out.to_csv(MANIFESTS / "dataset_candidate_universe_v1_2.csv", index=False)

    ledger = apply_duplicates(ledger)
    write_determinism_manifest(ledger)

    # Ensure every candidate appears once in candidates_v1_2
    ledger.to_csv(MANIFESTS / "candidates_v1_2.csv", index=False)

    # Semantic review CSV for eligible + human
    sem_cols = [
        "resolved_openml_id",
        "dataset_name",
        "source_pools",
        "objective_eligible",
        "suspected_issue",
        "evidence",
        "proposed_decision",
        "semantic_decision",
        "semantic_decision_reason",
        "review_status",
        "final_decision",
    ]
    for c in sem_cols:
        if c not in ledger.columns:
            ledger[c] = ""
    ledger[sem_cols].to_csv(MANIFESTS / "dataset_semantic_review_v1_2.csv", index=False)

    outcome, frozen_path, digest = freeze_if_ready(ledger)
    keep_ids = ledger.loc[ledger["final_decision"] == "KEEP", "resolved_openml_id"].astype(int).tolist()
    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "v1.2",
        "outcome": outcome,
        "n_universe_rows": int(len(universe)),
        "n_unique_openml_ids": int(len(unique_ids)),
        "n_objective_eligible": int(ledger["objective_eligible"].sum()),
        "n_keep": len(set(keep_ids)),
        "keep_ids": sorted(set(keep_ids)),
        "n_human_review": int((ledger["final_decision"] == "HUMAN_REVIEW_REQUIRED").sum()),
        "cohort_sha256": digest,
        "frozen_path": str(frozen_path) if frozen_path else None,
    }
    write_json(MANIFESTS / "dataset_freeze_status_v1_2.json", status)
    print(json.dumps(status, indent=2))
    return 0 if outcome == "GO" else (2 if outcome == "HUMAN_REVIEW_REQUIRED" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
