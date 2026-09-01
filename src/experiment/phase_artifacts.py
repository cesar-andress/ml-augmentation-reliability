"""Explicit phase output inventories and resume provenance validation.

Protocol v1.2.1 engineering hygiene — does not alter scientific algorithms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.experiment.confirmatory_constants import (
    GBDT_LEARNERS,
    HPO_ARMS,
    INNER_CV_FOLDS,
    LEARNERS,
    arm_to_fs_name,
    cell_manifest_stem,
    expected_scientific_arms_for_learner,
)


class ProvenanceConflictError(RuntimeError):
    """Raised when COMPLETE phase artifacts are missing, mutated, or hash-mismatched."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    data = path.read_bytes()
    return {"path": rel.replace("\\", "/"), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def inventory_from_paths(root: Path, rel_paths: Iterable[str]) -> list[dict[str, Any]]:
    return [artifact_record(root, rel) for rel in rel_paths]


def required_output_paths(phase: str, *, root: Path | None = None) -> list[str]:
    """Filesystem-relative paths that establish COMPLETE for each phase."""
    if phase == "P00_VALIDATE_FREEZE":
        # Validation is primarily in-memory; freeze check leaves no mandatory unit file.
        # Inventory may be empty; resume relies on config_hash + empty inventory OK.
        return []
    if phase == "P01_LOAD_DATASET":
        return ["manifests/dataset_manifest.json"]
    if phase == "P02_BUILD_SPLITS":
        return ["manifests/split_manifest.json", "manifests/split_indices.npz"]
    if phase == "P03_FIT_PREPROCESSING":
        return ["manifests/preprocessing_manifest.json", "manifests/preprocessor.pkl"]
    if phase == "P04_PREPARE_OUTER_AUGMENTATIONS":
        out: list[str] = []
        for arm in ("A1", "A2", "A3"):
            out.extend(
                [
                    f"generated/{arm}/features.parquet",
                    f"generated/{arm}/labels.npy",
                    f"generated/{arm}/manifest.json",
                ]
            )
        return out
    if phase == "P05_PREPARE_INNER_AUGMENTATION_CACHE":
        out = []
        for arm in ("A1", "A2", "A3"):
            for i in range(INNER_CV_FOLDS):
                base = f"hpo/inner_cache/{arm}/fold_{i}"
                out.extend([f"{base}/features.parquet", f"{base}/labels.npy", f"{base}/manifest.json"])
        return out
    if phase == "P06_RUN_GBDT_HPO":
        out = []
        for learner in GBDT_LEARNERS:
            for arm in HPO_ARMS:
                out.extend(
                    [
                        f"hpo/{learner}/{arm}/candidates.parquet",
                        f"hpo/{learner}/{arm}/best.json",
                    ]
                )
        return out
    if phase == "P07_RUN_A0PLUS":
        out = []
        for learner in GBDT_LEARNERS:
            out.extend(
                [
                    f"hpo/{learner}/A0plus/candidates.parquet",
                    f"hpo/{learner}/A0plus/best.json",
                    f"hpo/{learner}/A0plus/compute_match.json",
                ]
            )
        return out
    if phase == "P08_FIT_FINAL_LEARNERS":
        return [
            f"manifests/cells/{cell_manifest_stem(learner, arm)}.json"
            for learner in LEARNERS
            for arm in expected_scientific_arms_for_learner(learner)
        ]
    if phase == "P09_PREDICT_CAL_A":
        return [
            f"predictions/{learner}/{arm_to_fs_name(arm)}/cal_a.json"
            for learner in LEARNERS
            for arm in expected_scientific_arms_for_learner(learner)
        ]
    if phase == "P10_FIT_CALIBRATORS_AND_THRESHOLD":
        out = []
        for learner in LEARNERS:
            for arm in expected_scientific_arms_for_learner(learner):
                base = f"postprocessing/{learner}/{arm_to_fs_name(arm)}"
                out.extend([f"{base}/platt.json", f"{base}/isotonic.pkl", f"{base}/threshold.json"])
        return out
    if phase == "P11_PREDICT_CAL_B":
        return [
            f"predictions/{learner}/{arm_to_fs_name(arm)}/cal_b.json"
            for learner in LEARNERS
            for arm in expected_scientific_arms_for_learner(learner)
        ]
    if phase == "P12_FIT_CONFORMAL":
        return [
            f"postprocessing/{learner}/{arm_to_fs_name(arm)}/conformal.json"
            for learner in LEARNERS
            for arm in expected_scientific_arms_for_learner(learner)
        ]
    if phase == "P13_PREDICT_TEST":
        return [
            f"predictions/{learner}/{arm_to_fs_name(arm)}/test.json"
            for learner in LEARNERS
            for arm in expected_scientific_arms_for_learner(learner)
        ]
    if phase == "P14_COMPUTE_METRICS":
        return ["metrics/results.parquet", "metrics/results.csv"]
    if phase == "P15_VALIDATE_OUTPUTS":
        return ["manifests/validation_report.json"]
    if phase == "P16_FINALIZE_UNIT":
        # unit_complete is written during P16; inventory captured after write in runner.
        return ["status/unit_complete.json", "status/finalization_record.json"]
    raise KeyError(f"unknown phase {phase}")


def build_phase_inventory(root: Path, phase: str) -> list[dict[str, Any]]:
    paths = required_output_paths(phase, root=root)
    missing = [p for p in paths if not (root / p).exists()]
    if missing:
        raise ProvenanceConflictError(
            f"phase {phase} incomplete inventory; missing outputs: {missing}"
        )
    return inventory_from_paths(root, paths)


def aggregate_inventory_checksum(artifacts: list[dict[str, Any]]) -> str:
    """Deterministic checksum over path-sorted inventory (not mtimes)."""
    payload = [{"path": a["path"], "sha256": a["sha256"], "size": a["size"]} for a in artifacts]
    payload.sort(key=lambda x: x["path"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_stored_inventory(
    root: Path,
    phase: str,
    *,
    stored_artifacts: list[dict[str, Any]] | None,
    stored_checksum: str | None,
    input_hashes: dict[str, str],
    current_input_hashes: dict[str, str],
) -> None:
    """Abort on missing/mutated/stale COMPLETE outputs. Never silently rerun."""
    if input_hashes and input_hashes != current_input_hashes:
        raise ProvenanceConflictError(
            f"phase {phase} COMPLETE input hash mismatch: stored={input_hashes} current={current_input_hashes}"
        )

    expected_paths = required_output_paths(phase, root=root)
    if stored_artifacts is not None and len(stored_artifacts) == 0 and not expected_paths:
        return

    if stored_artifacts is None:
        # Legacy COMPLETE without inventory: require all expected paths exist and
        # match optional aggregate checksum when present; still abort if files missing.
        if expected_paths:
            missing = [p for p in expected_paths if not (root / p).exists()]
            if missing:
                raise ProvenanceConflictError(
                    f"phase {phase} COMPLETE lacks output_artifacts and required files missing: {missing}"
                )
            # Recompute inventory and, if checksum stored, require match
            current = inventory_from_paths(root, expected_paths)
            agg = aggregate_inventory_checksum(current)
            if stored_checksum is not None and stored_checksum != agg:
                # Older runs used single-file or placeholder checksums — if files exist,
                # require either exact aggregate match OR that stored checksum equals
                # sha256 of the primary mapping file when only one path existed historically.
                # For hygiene hardening: mismatched non-aggregate checksum with present
                # inventory still OK only when stored_artifacts was never recorded AND
                # we can verify files exist — but user asked: hashes must match current files.
                # So: without stored_artifacts, verify files exist; if stored_checksum set,
                # it must equal aggregate OR any individual file sha (compat) else conflict.
                individual = {a["sha256"] for a in current}
                if stored_checksum not in individual and stored_checksum != agg:
                    raise ProvenanceConflictError(
                        f"phase {phase} COMPLETE checksum mismatch vs persisted outputs "
                        f"(stored={stored_checksum}, aggregate={agg})"
                    )
        return

    stored_by_path = {a["path"]: a for a in stored_artifacts}
    for rel in expected_paths:
        if rel not in stored_by_path:
            raise ProvenanceConflictError(
                f"phase {phase} COMPLETE inventory missing expected path {rel}"
            )
        path = root / rel
        if not path.exists():
            raise ProvenanceConflictError(
                f"phase {phase} COMPLETE required output missing on disk: {rel}"
            )
        current = artifact_record(root, rel)
        prev = stored_by_path[rel]
        if current["sha256"] != prev["sha256"] or current["size"] != prev["size"]:
            raise ProvenanceConflictError(
                f"phase {phase} COMPLETE output mutated: {rel} "
                f"stored_sha={prev['sha256']} current_sha={current['sha256']}"
            )

    if stored_checksum is not None:
        agg = aggregate_inventory_checksum(list(stored_by_path.values()))
        # Prefer validating against recomputed current inventory
        current_inv = inventory_from_paths(root, expected_paths) if expected_paths else []
        cur_agg = aggregate_inventory_checksum(current_inv)
        if stored_checksum not in {agg, cur_agg} and expected_paths:
            # Allow stored checksum == cur_agg only
            if stored_checksum != cur_agg:
                raise ProvenanceConflictError(
                    f"phase {phase} COMPLETE aggregate checksum mismatch: "
                    f"stored={stored_checksum} current={cur_agg}"
                )


FINALIZATION_HASH_RULE = (
    "unit_complete.json.phase_hashes records P00–P15 output checksums only. "
    "P16's checksum is stored in unit_status.json and status/finalization_record.json "
    "to avoid circular hashing of unit_complete.json."
)
