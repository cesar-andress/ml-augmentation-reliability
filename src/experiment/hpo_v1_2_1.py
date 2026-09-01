"""Protocol v1.2.1 HPO sampling — deterministic shared candidate sequences."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import yaml

HPO_VERSION = "1.2.1"
HPO_CONFIG_PATH = "configs/hpo_v1_2_1.yaml"

XGB_SEARCH_KEYS = (
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
)
CAT_SEARCH_KEYS = ("depth", "l2_leaf_reg", "subsample", "rsm", "random_strength")


class HPOConfigError(ValueError):
    """Raised when HPO configuration is invalid or incomplete."""


def sha256_file_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_hpo_config(path: str = HPO_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("protocol_version") != HPO_VERSION:
        raise HPOConfigError(f"expected protocol_version {HPO_VERSION!r}, got {cfg.get('protocol_version')!r}")
    return cfg


def derive_hpo_seed(*, dataset_id: int, repeat: int, fold: int, learner: str) -> int:
    msg = f"{dataset_id}|{repeat}|{fold}|{learner}|hpo-v1.2.1"
    digest = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


def derive_generator_seed(*, dataset_id: int, repeat: int, fold: int, arm: str, inner_fold: int | None = None) -> int:
    if inner_fold is None:
        raise ValueError("inner_fold required for inner augmentation seeds")
    msg = f"{dataset_id}|{repeat}|{fold}|{inner_fold}|{arm}"
    digest = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31)


def make_hpo_rng(*, dataset_id: int, repeat: int, fold: int, learner: str) -> np.random.Generator:
    seed = derive_hpo_seed(dataset_id=dataset_id, repeat=repeat, fold=fold, learner=learner)
    return np.random.Generator(np.random.PCG64(seed))


def _sample_param(rng: np.random.Generator, spec: dict[str, Any]) -> Any:
    ptype = spec["type"]
    if ptype == "integer_discrete_uniform":
        return int(rng.choice(spec["values"]))
    if ptype == "continuous_uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if ptype == "log_uniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    raise HPOConfigError(f"unknown parameter type: {ptype!r}")


def sample_one_candidate(rng: np.random.Generator, learner: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_hpo_config()
    if learner not in cfg["learners"]:
        raise HPOConfigError(f"unknown learner: {learner!r}")
    lcfg = cfg["learners"][learner]
    params: dict[str, Any] = dict(lcfg["fixed"])
    search = lcfg["search"]
    if learner == "xgboost":
        keys = XGB_SEARCH_KEYS
    elif learner == "catboost":
        keys = CAT_SEARCH_KEYS
    else:
        raise HPOConfigError(f"HPO sampling only defined for GBDT learners, not {learner!r}")
    for key in keys:
        params[key] = _sample_param(rng, search[key])
    return params


def generate_candidate_sequence(
    *,
    dataset_id: int,
    repeat: int,
    fold: int,
    learner: str,
    n_candidates: int,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or load_hpo_config()
    rng = make_hpo_rng(dataset_id=dataset_id, repeat=repeat, fold=fold, learner=learner)
    return [sample_one_candidate(rng, learner, cfg) for _ in range(n_candidates)]


def shared_arm_candidates(
    *,
    dataset_id: int,
    repeat: int,
    fold: int,
    learner: str,
    n_candidates: int = 20,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """First n_candidates configs are identical for A0/A1/A2/A3."""
    return generate_candidate_sequence(
        dataset_id=dataset_id,
        repeat=repeat,
        fold=fold,
        learner=learner,
        n_candidates=n_candidates,
        cfg=cfg,
    )


def a0_plus_candidates(
    *,
    dataset_id: int,
    repeat: int,
    fold: int,
    learner: str,
    k_extra: int,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """A0+ is a strict prefix extension of A0 on the same sequence."""
    standard = int((cfg or load_hpo_config())["standard_budget"]["n_candidates"])
    total = standard + k_extra
    return generate_candidate_sequence(
        dataset_id=dataset_id,
        repeat=repeat,
        fold=fold,
        learner=learner,
        n_candidates=total,
        cfg=cfg,
    )


def compute_k_extra(*, g_max: float, t_bar: float, cap: int = 200) -> int:
    if t_bar <= 0:
        raise ValueError("t_bar must be positive")
    return min(cap, int(math.ceil(g_max / t_bar)))


def select_hpo_candidate(candidate_metrics: list[dict[str, Any]]) -> int:
    """Return candidate index using frozen tie-breaking hierarchy."""

    def sort_key(item: dict[str, Any]) -> tuple[float, float, int]:
        return (
            float(item["mean_inner_log_loss"]),
            float(item["std_inner_log_loss"]),
            int(item["candidate_index"]),
        )

    best = min(candidate_metrics, key=sort_key)
    return int(best["candidate_index"])


def inner_augmentation_cache_key(
    *,
    dataset_id: int,
    repeat: int,
    fold: int,
    arm: str,
    inner_fold: int,
) -> str:
    """Cache key is learner-independent."""
    return f"{dataset_id}|{repeat}|{fold}|{arm}|inner{inner_fold}"


def outer_a3_cache_key(*, dataset_id: int, repeat: int, fold: int) -> str:
    return f"{dataset_id}|{repeat}|{fold}|A3|outer"
