"""Frozen confirmatory runner constants — Protocol v1.2.1."""

from __future__ import annotations

PROTOCOL_VERSION = "1.2.1"
FREEZE_TAG = "protocol-v1.2.1-freeze"
FREEZE_COMMIT = "c346440ba6da683762da92a8fb126d57e2bcab6c"

COHORT_SHA256 = "209bc80826843940da92799ca48b406a34df7e2dbafd2ff26590d092187ecb34"
HPO_CONFIG_SHA256 = "092219107a9dadffedf8c663198b3d7a6da1a91bb3496d11dc797fb05eba5c7d"
A3_CONFIG_SHA256 = "1193b79df9f43b7a7356c27e9ddf77ddde68300731c34ec3ddfbfdb7b6782151"

VALID_REPEATS = (0, 1)
VALID_FOLDS = (0, 1, 2, 3, 4)

LEARNERS = ("xgboost", "catboost", "tabpfn", "tabicl")
GBDT_LEARNERS = ("xgboost", "catboost")
TFM_LEARNERS = ("tabpfn", "tabicl")
ARMS = ("A0", "A1", "A2", "A3")
AUGMENT_ARMS = ("A1", "A2", "A3")
HPO_ARMS = ("A0", "A1", "A2", "A3")
INNER_CV_FOLDS = 3
STANDARD_HPO_CANDIDATES = 20

PHASES = (
    "P00_VALIDATE_FREEZE",
    "P01_LOAD_DATASET",
    "P02_BUILD_SPLITS",
    "P03_FIT_PREPROCESSING",
    "P04_PREPARE_OUTER_AUGMENTATIONS",
    "P05_PREPARE_INNER_AUGMENTATION_CACHE",
    "P06_RUN_GBDT_HPO",
    "P07_RUN_A0PLUS",
    "P08_FIT_FINAL_LEARNERS",
    "P09_PREDICT_CAL_A",
    "P10_FIT_CALIBRATORS_AND_THRESHOLD",
    "P11_PREDICT_CAL_B",
    "P12_FIT_CONFORMAL",
    "P13_PREDICT_TEST",
    "P14_COMPUTE_METRICS",
    "P15_VALIDATE_OUTPUTS",
    "P16_FINALIZE_UNIT",
)

PHASE_STATUS_VALUES = ("PLANNED", "RUNNING", "COMPLETE", "FAILED", "BLOCKED")

SCIENTIFIC_STATUS_CONFIRMATORY = "CONFIRMATORY"
SCIENTIFIC_STATUS_PLANNED = "CONFIRMATORY_PLANNED_NOT_EXECUTED"

MAIN_PYTHON = ".venv_main/bin/python"
SYNTHCITY_PYTHON = ".venv_synthcity/bin/python"
A3_WORKER = "scripts/a3_tabddpm_worker.py"

CHECKPOINTS = {
    "tabpfn": "checkpoints/tabpfn/tabpfn-v2-classifier.ckpt",
    "tabicl": "checkpoints/tabicl/tabicl-classifier-v1-20250208.ckpt",
}

# Full-file SHA256 of frozen TFM checkpoints (protocol identity).
CHECKPOINT_SHA256 = {
    "tabpfn": "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f",
    "tabicl": "04c5c1d261c1f782dc9b263990dcfa5152b1949ed8451ad16c010cdafbea07e0",
}

TEMPERATURES = {"primary": 1.0, "sensitivity": 0.9}


def arm_to_fs_name(arm: str) -> str:
    """Map scientific arm label to a single filesystem-safe directory name.

    Scientific tables/manifests keep 'A0+'; on-disk directories use 'A0plus'.
    """
    if arm == "A0+":
        return "A0plus"
    return arm


def cell_manifest_stem(learner: str, arm: str) -> str:
    return f"{learner}_{arm_to_fs_name(arm)}"


def expected_scientific_arms_for_learner(learner: str) -> tuple[str, ...]:
    if learner in GBDT_LEARNERS:
        return ("A0", "A1", "A2", "A3", "A0+")
    return ("A0", "A1", "A2", "A3")


def unit_id(*, dataset_id: int, repeat: int, fold: int) -> str:
    return f"d{dataset_id}_r{repeat}_f{fold}"


def unit_root(repo_root, unit: str) -> "Path":
    from pathlib import Path

    return Path(repo_root) / "results" / "confirmatory" / "units" / unit
