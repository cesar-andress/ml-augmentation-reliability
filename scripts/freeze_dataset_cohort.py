#!/usr/bin/env python
"""Confirmatory dataset cohort screening and freeze (no model training)."""

from __future__ import annotations

import hashlib
import json
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

from src.data.candidates import build_candidate_universe
from src.data.duplicates import build_duplicate_map
from src.data.feature_audit import FEATURE_TYPE_RULES, audit_feature_types
from src.data.objective_filters import evaluate_objective
from src.data.openml_loader import binarize_labels
from src.data.semantic_review import propose_semantic_row
from src.logging_utils import write_json

MANIFESTS = ROOT / "artifacts" / "manifests"
RAW = ROOT / "data" / "raw" / "openml"
SPLIT_SEED = 42
N_SPLITS = 5
N_REPEATS = 2


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_raw(openml_id: int, X: pd.DataFrame, y: pd.Series, meta: dict) -> str:
    d = RAW / str(openml_id)
    d.mkdir(parents=True, exist_ok=True)
    x_path = d / "X.parquet"
    y_path = d / "y.parquet"
    meta_path = d / "meta.json"
    X.to_parquet(x_path)
    pd.DataFrame({"y": y}).to_parquet(y_path)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    # checksum over X+y parquet bytes
    h = hashlib.sha256()
    h.update(x_path.read_bytes())
    h.update(y_path.read_bytes())
    digest = h.hexdigest()
    (d / "raw_checksum.sha256").write_text(digest + "\n")
    return digest


def openml_qualities_prefilter(oid: int) -> dict:
    """Cheap metadata gate to avoid huge downloads. Not a substitute for objective screen."""
    out = {"openml_id": oid, "skip_download": False, "reason": "", "qualities": {}}
    try:
        ds = openml.datasets.get_dataset(oid, download_data=False, download_qualities=True)
        q = ds.qualities or {}
        out["qualities"] = {k: q.get(k) for k in [
            "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses",
            "MinorityClassSize", "MajorityClassSize", "NumberOfMissingValues",
        ] if k in q}
        n = q.get("NumberOfInstances")
        n_cls = q.get("NumberOfClasses")
        if n is not None and (n < 500 or n > 10_000):
            out["skip_download"] = True
            out["reason"] = f"quality_rows_out_of_range:{n}"
        if n_cls is not None and n_cls != 2:
            out["skip_download"] = True
            out["reason"] = f"quality_not_binary:{n_cls}"
        out["name"] = ds.name
        out["version"] = ds.version
        out["description"] = (ds.description or "")[:5000]
        out["default_target"] = ds.default_target_attribute
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def load_and_screen_one(oid: int, provenance_rows: pd.DataFrame, prevalence_upper: float) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    pre = openml_qualities_prefilter(oid)
    source_pools = sorted(provenance_rows["source_pool"].unique().tolist())
    names = provenance_rows["dataset_name"].dropna().astype(str).tolist()
    base = {
        "resolved_openml_id": oid,
        "source_pools": ";".join(source_pools),
        "source_dataset_ids": ";".join(
            f"{r.source_pool}:{r.source_dataset_id}" for r in provenance_rows.itertuples()
        ),
        "dataset_name": pre.get("name") or (names[0] if names else str(oid)),
        "openml_version": pre.get("version"),
        "retrieval_timestamp_utc": ts,
        "prefilter_skip": pre.get("skip_download", False),
        "prefilter_reason": pre.get("reason", ""),
        "description_excerpt": (pre.get("description") or "")[:800],
        "objective_eligible": False,
        "error": pre.get("error", ""),
    }
    if pre.get("skip_download"):
        base["fail_stage"] = "prefilter"
        return base

    try:
        ds = openml.datasets.get_dataset(oid, download_data=True, download_qualities=True)
        X, y, categorical_indicator, attribute_names = ds.get_data(
            target=ds.default_target_attribute, dataset_format="dataframe"
        )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=attribute_names)
        y = pd.Series(y)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # binary check before binarize
        n_unique = y.nunique(dropna=True)
        if n_unique != 2:
            base.update(
                {
                    "fail_stage": "not_binary",
                    "n_rows": int(len(X)),
                    "n_source_features": int(X.shape[1]),
                    "n_classes_observed": int(n_unique),
                    "openml_version": ds.version,
                    "dataset_name": ds.name,
                    "description_excerpt": (ds.description or "")[:800],
                }
            )
            return base

        y_bin, y_meta = binarize_labels(y)
        audit = audit_feature_types(
            X, openml_categorical_indicator=list(categorical_indicator) if categorical_indicator is not None else None
        )
        missing_frac = float(X.isna().to_numpy().mean()) if X.size else 0.0
        obj = evaluate_objective(
            y_bin=y_bin,
            audit=audit,
            n_rows=int(len(X)),
            missing_frac=missing_frac,
            prevalence_upper=prevalence_upper,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            seed=SPLIT_SEED,
        )
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
        base.update(
            {
                "openml_version": ds.version,
                "dataset_name": ds.name,
                "description_excerpt": (ds.description or "")[:800],
                "target_name": ds.default_target_attribute,
                "class_labels": "|".join(y_meta["class_labels"]),
                "minority_label": y_meta["minority_label"],
                "majority_label": y_meta["majority_label"],
                "n_rows": int(len(X)),
                "n_source_features": int(X.shape[1]),
                "n_numeric": len(audit.numeric_cols),
                "n_categorical": len(audit.categorical_cols),
                "n_continuous": len(audit.continuous_cols),
                "n_integer_valued_numeric": len(audit.integer_valued_numeric_cols),
                "missing_indicator_count": audit.missing_indicator_count,
                "n_encoded_features": audit.n_encoded_features_est,
                "n_minority": obj.metrics["n_minority"],
                "n_majority": obj.metrics["n_majority"],
                "minority_prevalence": obj.metrics["minority_prevalence"],
                "missing_fraction": missing_frac,
                "max_balanced_train_size": obj.max_balanced_train_size,
                "raw_checksum": checksum,
                "objective_eligible": bool(obj.pass_all),
                "objective_checks_json": json.dumps(obj.checks),
                "dtype_disagreement": bool(audit.dtype_metadata_disagreement),
                "dtype_disagreement_json": json.dumps(audit.dtype_metadata_disagreement)[:2000],
                "fail_stage": "" if obj.pass_all else "objective",
                "prevalence_upper_bound_used": prevalence_upper,
            }
        )
        # per-check columns
        for k, v in obj.checks.items():
            base[f"check_{k}"] = v
        return base
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"
        base["fail_stage"] = "exception"
        base["traceback"] = traceback.format_exc()[-2000:]
        return base


def run_screen(prevalence_upper: float = 0.40) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = build_candidate_universe()
    universe.to_csv(MANIFESTS / "dataset_candidate_universe.csv", index=False)

    # Unique OpenML IDs for download (prefer collecting all provenance)
    unique_ids = sorted(universe["resolved_openml_id"].unique().tolist())
    screen_rows = []
    for i, oid in enumerate(unique_ids):
        prov = universe[universe["resolved_openml_id"] == oid]
        print(f"[{i+1}/{len(unique_ids)}] screening OpenML {oid} ...", flush=True)
        screen_rows.append(load_and_screen_one(oid, prov, prevalence_upper=prevalence_upper))

    screen = pd.DataFrame(screen_rows)
    screen.to_csv(MANIFESTS / "dataset_objective_screen.csv", index=False)

    # Semantic review for all screened (emphasize objective-eligible)
    sem_rows = []
    for _, r in screen.iterrows():
        pools = str(r.get("source_pools", "")).split(";")
        # one semantic row per source pool membership for provenance clarity
        for pool in pools:
            if not pool:
                continue
            ta_meta = None
            cl_meta = None
            if pool == "tabarena_v0.1":
                ta_meta = {"pool": "tabarena_v0.1"}
            if pool == "climb":
                cl_meta = {"pool": "climb"}
            sem_rows.append(
                propose_semantic_row(
                    resolved_openml_id=int(r["resolved_openml_id"]),
                    dataset_name=str(r.get("dataset_name", "")),
                    source_pool=pool,
                    objective_eligible=bool(r.get("objective_eligible", False)),
                    description=str(r.get("description_excerpt") or ""),
                    tabarena_meta=ta_meta,
                    climb_meta=cl_meta,
                    dtype_disagreement=bool(r.get("dtype_disagreement", False)),
                )
            )
    semantic = pd.DataFrame(sem_rows)
    semantic.to_csv(MANIFESTS / "dataset_semantic_review.csv", index=False)

    dup = build_duplicate_map(screen)
    if dup.empty:
        dup = pd.DataFrame(
            columns=[
                "removed_openml_id",
                "removed_name",
                "retained_openml_id",
                "retained_name",
                "reason",
                "signals",
            ]
        )
    dup.to_csv(MANIFESTS / "dataset_duplicate_map.csv", index=False)
    return universe, screen, semantic


def write_review_package(semantic: pd.DataFrame) -> list[dict]:
    amb = semantic[semantic["review_status"] == "HUMAN_REVIEW_REQUIRED"].copy()
    # unique by openml id
    amb = amb.drop_duplicates(subset=["resolved_openml_id", "suspected_issue"])
    lines = [
        "# DATASET_REVIEW — human semantic decisions required",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This document lists **only** datasets needing human semantic decisions.",
        "No model results are included.",
        "",
    ]
    items = []
    if amb.empty:
        lines.append("No ambiguous cases. `HUMAN_REVIEW_REQUIRED` count = 0.")
    else:
        lines.append(f"Ambiguous count (rows): {len(amb)}")
        lines.append("")
        for _, r in amb.sort_values("resolved_openml_id").iterrows():
            item = {
                "resolved_openml_id": int(r["resolved_openml_id"]),
                "dataset_name": r["dataset_name"],
                "source_pool": r["source_pool"],
                "suspected_issue": r["suspected_issue"],
                "evidence": r["evidence"],
                "proposed_decision": r["proposed_decision"],
            }
            items.append(item)
            lines += [
                f"## OpenML {r['resolved_openml_id']} — {r['dataset_name']}",
                "",
                f"- **source_pool:** {r['source_pool']}",
                f"- **suspected_issue:** {r['suspected_issue']}",
                f"- **evidence:** {r['evidence']}",
                f"- **why it matters:** Semantic exclusion {r['suspected_issue']} would remove the dataset from the confirmatory cohort if confirmed.",
                f"- **proposed:** {r['proposed_decision'] or '(none)'}",
                f"- **confidence:** MEDIUM",
                f"- **decision:** _(blank — human required)_",
                "",
            ]
    (MANIFESTS / "DATASET_REVIEW.md").write_text("\n".join(lines))
    return items


def freeze_cohort(screen: pd.DataFrame, semantic: pd.DataFrame, dup: pd.DataFrame, amendment_used: bool, before: int, after: int) -> Path | None:
    removed = set(int(x) for x in dup["removed_openml_id"].tolist()) if len(dup) else set()
    # Keep decision from semantic: for each oid, if any row has EXCLUDE with decision set, exclude;
    # require all objective-eligible oids to have a resolved KEEP decision and no HUMAN_REVIEW_REQUIRED
    human = semantic[semantic["review_status"] == "HUMAN_REVIEW_REQUIRED"]
    if len(human):
        return None

    keep_ids = set()
    for oid, g in semantic.groupby("resolved_openml_id"):
        if not bool(screen.loc[screen["resolved_openml_id"] == oid, "objective_eligible"].any()):
            continue
        if int(oid) in removed:
            continue
        decisions = set(g["decision"].fillna("").tolist())
        if "EXCLUDE" in decisions and "KEEP" not in decisions:
            continue
        if any(d == "" for d in g["decision"].tolist() if g.loc[g["decision"] == d].shape):
            # unresolved blank decisions on eligible
            if (g["decision"] == "").any() and (g["objective_eligible"] == True).any():  # noqa: E712
                return None
        if "KEEP" in decisions:
            keep_ids.add(int(oid))

    rows = []
    for oid in sorted(keep_ids):
        r = screen.loc[screen["resolved_openml_id"] == oid].iloc[0]
        sem = semantic[semantic["resolved_openml_id"] == oid].iloc[0]
        # prefer tabarena as source_pool label if present
        pools = str(r["source_pools"]).split(";")
        source_pool = "tabarena_v0.1" if "tabarena_v0.1" in pools else pools[0]
        rows.append(
            {
                "cohort_version": "v1",
                "resolved_openml_id": int(oid),
                "openml_version": r.get("openml_version"),
                "dataset_name": r.get("dataset_name"),
                "source_pool": source_pool,
                "n_rows": r.get("n_rows"),
                "n_source_features": r.get("n_source_features"),
                "n_encoded_features": r.get("n_encoded_features"),
                "n_continuous": r.get("n_continuous"),
                "n_categorical": r.get("n_categorical"),
                "minority_label": r.get("minority_label"),
                "majority_label": r.get("majority_label"),
                "n_minority": r.get("n_minority"),
                "n_majority": r.get("n_majority"),
                "minority_prevalence": r.get("minority_prevalence"),
                "missing_fraction": r.get("missing_fraction"),
                "raw_checksum": r.get("raw_checksum"),
                "semantic_decision": sem.get("decision"),
                "semantic_decision_reason": sem.get("decision_reason"),
                "prevalence_amendment_used": amendment_used,
            }
        )
    frozen = pd.DataFrame(rows).sort_values("resolved_openml_id")
    out = MANIFESTS / "datasets_frozen_v1.csv"
    # deterministic CSV bytes
    csv_bytes = frozen.to_csv(index=False).encode("utf-8")
    out.write_bytes(csv_bytes)
    digest = _sha256_bytes(csv_bytes)
    (MANIFESTS / "datasets_frozen_v1.sha256").write_text(digest + "\n")

    amendment_path = MANIFESTS / "prevalence_amendment.json"
    write_json(
        amendment_path,
        {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "used": amendment_used,
            "reason": "final_qualifying_count_below_15" if amendment_used else "not_needed",
            "number_before_amendment": before,
            "number_after_amendment": after if amendment_used else before,
            "rule_changed": "minority_prevalence_upper_0.40_to_0.50" if amendment_used else None,
        },
    )
    return out


def write_protocol_snapshot(frozen_csv: Path, git_commit: str) -> None:
    digest = (MANIFESTS / "datasets_frozen_v1.sha256").read_text().strip()
    frozen = pd.read_csv(frozen_csv)
    snap = {
        "freeze_name": "confirmatory_freeze_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset_cohort_csv": str(frozen_csv.relative_to(ROOT)),
        "dataset_cohort_sha256": digest,
        "dataset_ids": [
            {"resolved_openml_id": int(r.resolved_openml_id), "openml_version": r.openml_version, "dataset_name": r.dataset_name}
            for r in frozen.itertuples()
        ],
        "eligibility_rules": {
            "binary": True,
            "n_rows": [500, 10000],
            "n_continuous_ge": 1,
            "minority_prevalence_min": 0.02,
            "minority_prevalence_max": float(frozen["prevalence_amendment_used"].iloc[0] and 0.50 or 0.40)
            if "prevalence_amendment_used" in frozen.columns
            else 0.40,
            "n_minority_ge": 250,
            "missing_fraction_lt": 0.10,
            "n_encoded_features_le": 100,
            "tabpfn_max_balanced_train_le": 10000,
            "feature_type_rules": FEATURE_TYPE_RULES.strip(),
        },
        "semantic_exclusion_codes": [
            "A_duplicate_derivative",
            "B_artificial_ovr",
            "C_simulated_or_synthetic",
            "D_fictional",
            "E_non_iid",
            "F_repeated_measurements",
            "G_grouped_observations",
            "H_time_series_temporal",
            "I_subject_leakage_risk",
            "J_unclear_provenance",
        ],
        "duplicate_selection_rule": [
            "prefer_tabarena_curated_reupload",
            "else_prefer_larger_n",
            "else_tiebreak_smaller_openml_id",
            "never_duplicate_on_dimensions_alone",
        ],
        "split_design": {
            "outer": "RepeatedStratifiedKFold n_splits=5 n_repeats=2",
            "inner_of_remainder": {"TRAIN": 0.625, "CAL_A": 0.1875, "CAL_B": 0.1875},
            "approx_global": {"TRAIN": 0.50, "CAL_A": 0.15, "CAL_B": 0.15, "TEST": 0.20},
            "seed": SPLIT_SEED,
        },
        "learner_identities": {
            "xgboost": "3.4.1 device=cuda tree_method=hist",
            "catboost": "1.2.10",
            "tabpfn": "8.5.0 Prior-Labs/TabPFN-v2-clf/tabpfn-v2-classifier.ckpt",
            "tabicl": "2.1.1 tabicl-classifier-v1-20250208.ckpt",
        },
        "augmentation_arms": ["A0", "A1", "A2", "A3", "A0+"],
        "augmentation_target": "balance minority count to majority count on TRAIN only",
        "primary_endpoint": "Platt-calibrated TEST log loss (also report raw TEST log loss)",
        "secondary_endpoints": [
            "AUROC",
            "calibration_slope",
            "calibration_intercept",
            "conformal_average_set_size",
            "conformal_marginal_coverage",
            "F1_at_tuned_threshold",
        ],
        "seed_policy": "global seed 42; fold_seed = seed + 1000*repeat + fold",
        "hpo_search_spaces": "20-config HPO for XGBoost and CatBoost (deferred; not executed in freeze)",
        "software_version_policy": "pin scientific package identities; two-env synthcity isolation",
        "results_policy": {
            "smoke_path": "results/smoke/",
            "confirmatory_path": "results/confirmatory/",
            "scientific_status_required": "CONFIRMATORY",
        },
    }
    # fix prevalence max from amendment file
    am = json.loads((MANIFESTS / "prevalence_amendment.json").read_text())
    snap["eligibility_rules"]["minority_prevalence_max"] = 0.50 if am.get("used") else 0.40
    snap["prevalence_amendment"] = am
    (MANIFESTS / "confirmatory_freeze_v1.yaml").write_text(yaml.safe_dump(snap, sort_keys=False))


def count_qualifying(screen: pd.DataFrame, semantic: pd.DataFrame, dup: pd.DataFrame) -> int:
    removed = set(int(x) for x in dup["removed_openml_id"].tolist()) if len(dup) else set()
    if (semantic["review_status"] == "HUMAN_REVIEW_REQUIRED").any():
        # unresolved — count provisional KEEP proposals among objective eligible excluding dups
        pass
    n = 0
    for oid in screen.loc[screen["objective_eligible"] == True, "resolved_openml_id"]:  # noqa: E712
        oid = int(oid)
        if oid in removed:
            continue
        g = semantic[semantic["resolved_openml_id"] == oid]
        if (g["review_status"] == "HUMAN_REVIEW_REQUIRED").any():
            continue
        if (g["decision"] == "KEEP").any() and not (g["decision"] == "EXCLUDE").all():
            n += 1
    return n


def main() -> int:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    print("=== pass1 prevalence_upper=0.40 ===", flush=True)
    universe, screen, semantic = run_screen(prevalence_upper=0.40)
    dup = pd.read_csv(MANIFESTS / "dataset_duplicate_map.csv") if (MANIFESTS / "dataset_duplicate_map.csv").exists() else pd.DataFrame()
    ambiguous = write_review_package(semantic)
    before = count_qualifying(screen, semantic, dup)

    status = {
        "n_universe_rows": int(len(universe)),
        "n_unique_openml_ids": int(universe["resolved_openml_id"].nunique()),
        "n_objective_eligible": int(screen["objective_eligible"].sum()),
        "n_human_review_required": int((semantic["review_status"] == "HUMAN_REVIEW_REQUIRED").sum()),
        "n_qualifying_pass1": before,
        "ambiguous": ambiguous,
    }
    write_json(MANIFESTS / "dataset_freeze_status.json", status)

    if ambiguous:
        print("STOP: HUMAN_REVIEW_REQUIRED", flush=True)
        print(json.dumps(status, indent=2, default=str))
        return 2

    amendment_used = False
    after = before
    if before < 15:
        print("=== pass2 prevalence_upper=0.50 (authorized amendment) ===", flush=True)
        universe, screen, semantic = run_screen(prevalence_upper=0.50)
        dup = pd.read_csv(MANIFESTS / "dataset_duplicate_map.csv")
        ambiguous = write_review_package(semantic)
        after = count_qualifying(screen, semantic, dup)
        amendment_used = True
        status.update(
            {
                "n_objective_eligible": int(screen["objective_eligible"].sum()),
                "n_human_review_required": int((semantic["review_status"] == "HUMAN_REVIEW_REQUIRED").sum()),
                "n_qualifying_pass2": after,
                "ambiguous": ambiguous,
                "amendment_used": True,
            }
        )
        write_json(MANIFESTS / "dataset_freeze_status.json", status)
        if ambiguous:
            print("STOP: HUMAN_REVIEW_REQUIRED after amendment", flush=True)
            return 2
        if after < 15:
            write_json(
                MANIFESTS / "prevalence_amendment.json",
                {
                    "date_utc": datetime.now(timezone.utc).isoformat(),
                    "used": True,
                    "reason": "final_qualifying_count_below_15",
                    "number_before_amendment": before,
                    "number_after_amendment": after,
                    "outcome": "DATASET_COHORT_NO_GO",
                },
            )
            print("DATASET_COHORT_NO_GO", flush=True)
            return 3

    out = freeze_cohort(screen, semantic, dup, amendment_used, before, after)
    if out is None:
        print("STOP: unresolved semantic decisions", flush=True)
        return 2
    import subprocess

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parents[1], text=True).strip()
    write_protocol_snapshot(out, git_commit=git_commit)
    print(f"FROZEN {out} n={len(pd.read_csv(out))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
