"""Dry-run planning for confirmatory units — no model/data/CUDA access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiment.confirmatory_constants import (
    A3_WORKER,
    ARMS,
    CHECKPOINTS,
    GBDT_LEARNERS,
    HPO_ARMS,
    INNER_CV_FOLDS,
    LEARNERS,
    MAIN_PYTHON,
    PHASES,
    PROTOCOL_VERSION,
    SCIENTIFIC_STATUS_PLANNED,
    STANDARD_HPO_CANDIDATES,
    SYNTHCITY_PYTHON,
    TEMPERATURES,
    unit_id,
    unit_root,
)
from src.experiment.hpo_v1_2_1 import (
    derive_generator_seed,
    derive_hpo_seed,
    inner_augmentation_cache_key,
    outer_a3_cache_key,
    shared_arm_candidates,
)


def expected_final_cells() -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    for learner in LEARNERS:
        for arm in ARMS:
            cells.append({"learner": learner, "arm": arm, "kind": "standard"})
    for learner in GBDT_LEARNERS:
        cells.append({"learner": learner, "arm": "A0+", "kind": "a0plus"})
    return cells


def build_dry_run_plan(
    *,
    repo_root: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    dataset_id = int(validation["dataset_id"])
    repeat = int(validation["repeat"])
    fold = int(validation["fold"])
    uid = unit_id(dataset_id=dataset_id, repeat=repeat, fold=fold)
    root = unit_root(repo_root, uid)

    hpo_candidates = {
        learner: {
            arm: len(shared_arm_candidates(dataset_id=dataset_id, repeat=repeat, fold=fold, learner=learner))
            for arm in HPO_ARMS
        }
        for learner in GBDT_LEARNERS
    }

    inner_cache_paths = {
        arm: [
            str(root / "hpo" / "inner_cache" / arm / f"fold_{i}")
            for i in range(INNER_CV_FOLDS)
        ]
        for arm in ("A1", "A2", "A3")
    }

    outer_gen_paths = {
        arm: str(root / "generated" / arm)
        for arm in ("A1", "A2", "A3")
    }

    hpo_paths = {
        learner: {
            arm: {
                "candidates": str(root / "hpo" / learner / arm / "candidates.parquet"),
                "best": str(root / "hpo" / learner / arm / "best.json"),
            }
            for arm in HPO_ARMS
        }
        for learner in GBDT_LEARNERS
    }

    a0plus_paths = {
        learner: str(root / "hpo" / learner / "A0plus" / "compute_match.json")
        for learner in GBDT_LEARNERS
    }

    postprocessing_paths = {
        learner: {
            arm: {
                "platt": str(root / "postprocessing" / learner / arm / "platt.json"),
                "isotonic": str(root / "postprocessing" / learner / arm / "isotonic.pkl"),
                "threshold": str(root / "postprocessing" / learner / arm / "threshold.json"),
                "conformal": str(root / "postprocessing" / learner / arm / "conformal.json"),
            }
            for arm in list(ARMS) + ["A0+"]
            if arm in ARMS or (arm == "A0+" and learner in GBDT_LEARNERS)
        }
        for learner in LEARNERS
    }
    # Fix A0+ postprocessing only for GBDT
    postprocessing_paths = {
        l: {a: p for a, p in arms.items() if a != "A0+" or l in GBDT_LEARNERS}
        for l, arms in postprocessing_paths.items()
    }

    seeds = {
        "hpo": {
            learner: derive_hpo_seed(dataset_id=dataset_id, repeat=repeat, fold=fold, learner=learner)
            for learner in GBDT_LEARNERS
        },
        "outer_a3": "outer_fold_seed_at_P02",
        "inner_augmentation": {
            arm: {
                i: derive_generator_seed(
                    dataset_id=dataset_id, repeat=repeat, fold=fold, arm=arm, inner_fold=i
                )
                for i in range(INNER_CV_FOLDS)
            }
            for arm in ("A1", "A2", "A3")
        },
    }

    checkpoints = {
        name: {"path": str(repo_root / rel), "exists": (repo_root / rel).exists()}
        for name, rel in CHECKPOINTS.items()
    }

    environments = {
        "main_python": str(repo_root / MAIN_PYTHON),
        "synthcity_python": str(repo_root / SYNTHCITY_PYTHON),
        "a3_worker": str(repo_root / A3_WORKER),
        "main_exists": (repo_root / MAIN_PYTHON).exists(),
        "synthcity_exists": (repo_root / SYNTHCITY_PYTHON).exists(),
    }

    expected_outputs = {
        "unit_manifest": str(root / "manifests" / "unit_manifest.json"),
        "split_manifest": str(root / "manifests" / "split_manifest.json"),
        "preprocessing_manifest": str(root / "manifests" / "preprocessing_manifest.json"),
        "dry_run_plan": str(root / "manifests" / "dry_run_plan.json"),
        "unit_status": str(root / "status" / "unit_status.json"),
        "result_rows": str(root / "metrics" / "unit_results.parquet"),
        "cell_manifests": [
            str(root / "manifests" / "cells" / f"{l}_{a}.json")
            for l in LEARNERS
            for a in ARMS
        ]
        + [str(root / "manifests" / "cells" / f"{l}_A0plus.json") for l in GBDT_LEARNERS],
    }

    return {
        "scientific_status": SCIENTIFIC_STATUS_PLANNED,
        "protocol_version": PROTOCOL_VERSION,
        "freeze_tag": validation["freeze_tag"],
        "freeze_commit": validation["freeze_commit"],
        "cohort_sha256": validation["cohort_sha256"],
        "hpo_config_sha256": validation["hpo_config_sha256"],
        "a3_config_sha256": validation["a3_config_sha256"],
        "unit_id": uid,
        "dataset_id": dataset_id,
        "dataset_name": validation["dataset_name"],
        "dataset_version": validation["dataset_version"],
        "expected_raw_checksum": validation["expected_raw_checksum"],
        "repeat": repeat,
        "fold": fold,
        "unit_root": str(root),
        "phase_graph": list(PHASES),
        "learners": list(LEARNERS),
        "arms": list(ARMS),
        "a0plus_learners": list(GBDT_LEARNERS),
        "expected_final_cell_count": len(expected_final_cells()),
        "expected_final_cells": expected_final_cells(),
        "hpo": {
            "standard_candidates": STANDARD_HPO_CANDIDATES,
            "inner_cv_folds": INNER_CV_FOLDS,
            "learners": list(GBDT_LEARNERS),
            "arms": list(HPO_ARMS),
            "candidate_counts_by_learner_arm": hpo_candidates,
            "shared_sequence_across_arms": True,
        },
        "a0plus": {
            "learners": list(GBDT_LEARNERS),
            "compute_match_manifest": a0plus_paths,
            "formula": "k_extra = min(200, ceil(g_max / t_bar)); total = 20 + k_extra",
        },
        "tfm_temperatures": TEMPERATURES,
        "paths": {
            "outer_generated": outer_gen_paths,
            "inner_cache": inner_cache_paths,
            "hpo": hpo_paths,
            "postprocessing": postprocessing_paths,
        },
        "seeds": seeds,
        "cache_keys": {
            "outer_a3": outer_a3_cache_key(dataset_id=dataset_id, repeat=repeat, fold=fold),
            "inner_examples": {
                arm: inner_augmentation_cache_key(
                    dataset_id=dataset_id, repeat=repeat, fold=fold, arm=arm, inner_fold=0
                )
                for arm in ("A1", "A2", "A3")
            },
        },
        "checkpoints": checkpoints,
        "environments": environments,
        "expected_outputs": expected_outputs,
        "a3_ipc": {
            "transport": ["parquet", "npy", "json"],
            "worker_script": environments["a3_worker"],
            "validation": "schema_and_checksum_before_accept",
        },
        "dry_run_guarantees": {
            "model_initialization": False,
            "cuda_initialization": False,
            "dataset_payload_download": False,
            "checkpoint_load": False,
            "metric_computation": False,
        },
    }
