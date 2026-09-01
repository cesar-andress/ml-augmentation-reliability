"""Live confirmatory phase implementations — Protocol v1.2.1 orchestration."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from src.augmentation.arms import (
    assert_balanced_to_majority,
    augment_a0,
    augment_a1,
    augment_a2,
    shared_repair_with_cell_stats,
)
from src.calibration.posthoc import IsotonicCalibrator, PlattCalibrator
from src.calibration.threshold import tune_threshold_f1
from src.conformal.split_conformal import (
    conformity_score,
    conformal_quantile_with_meta,
    prediction_sets,
    set_metrics,
)
from src.data.openml_loader import binarize_labels
from src.data.splitting import build_split_for_fold
from src.evaluation.metrics import compute_metrics
from src.experiment.confirmatory_constants import (
    ARMS,
    A3_CONFIG_SHA256,
    CHECKPOINT_SHA256,
    CHECKPOINTS,
    FREEZE_TAG,
    GBDT_LEARNERS,
    HPO_ARMS,
    INNER_CV_FOLDS,
    LEARNERS,
    PROTOCOL_VERSION,
    SCIENTIFIC_STATUS_CONFIRMATORY,
    STANDARD_HPO_CANDIDATES,
    SYNTHCITY_PYTHON,
    TEMPERATURES,
    arm_to_fs_name,
    cell_manifest_stem,
)
from src.experiment.phase_artifacts import FINALIZATION_HASH_RULE
from src.experiment.confirmatory_plan import expected_final_cells
from src.experiment.confirmatory_schema import CONFIRMATORY_RESULT_COLUMNS
from src.experiment.hpo_v1_2_1 import (
    a0_plus_candidates,
    compute_k_extra,
    derive_generator_seed,
    load_hpo_config,
    select_hpo_candidate,
    shared_arm_candidates,
)
from src.experiment.live_adapters import LiveAdapters
from src.experiment.test_access_guard import UnitTestAccessRegistry
from src.logging_utils import write_json
from src.logging_utils.timing import timed_gpu
from src.preprocessing.pipeline import fit_preprocessor


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _sha256_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


@dataclass
class LiveContext:
    repo_root: Path
    root: Path
    dataset_id: int
    repeat: int
    fold: int
    adapters: LiveAdapters = field(default_factory=LiveAdapters)
    validation: dict[str, Any] | None = None

    # filled by phases
    bundle: Any = None
    y_bin: np.ndarray | None = None
    y_meta: dict[str, Any] | None = None
    row_ids: np.ndarray | None = None
    split: Any = None
    fold_seed: int = 0
    pre: Any = None
    X_parts: dict[str, pd.DataFrame] = field(default_factory=dict)
    y_parts: dict[str, np.ndarray] = field(default_factory=dict)
    outer_aug: dict[str, Any] = field(default_factory=dict)
    inner_cache: dict[str, dict[int, Any]] = field(default_factory=dict)
    hpo_best: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    a0plus_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    fitted: dict[tuple[str, str], Any] = field(default_factory=dict)
    cal_a_preds: dict[tuple[str, str], dict[str, np.ndarray]] = field(default_factory=dict)
    post: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    cal_b_preds: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    conformal: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    test_preds: dict[tuple[str, str], dict[str, np.ndarray]] = field(default_factory=dict)
    result_rows: list[dict[str, Any]] = field(default_factory=list)
    guard: UnitTestAccessRegistry = field(default_factory=UnitTestAccessRegistry)
    generator_costs: dict[str, float] = field(default_factory=dict)
    a0_hpo_timings: dict[str, list[float]] = field(default_factory=dict)


class LivePhaseEngine:
    def __init__(self, ctx: LiveContext):
        self.ctx = ctx

    def hydrate_skipped_phase(self, phase: str) -> None:
        """Reload in-memory context from persisted artifacts (READ-ONLY; never rewrite)."""
        if phase == "P00_VALIDATE_FREEZE":
            if self.ctx.validation is None and not self.ctx.adapters.skip_freeze_validation:
                from src.experiment.confirmatory_validation import validate_pre_run

                self.ctx.validation = validate_pre_run(
                    repo_root=self.ctx.repo_root,
                    dataset_id=self.ctx.dataset_id,
                    repeat=self.ctx.repeat,
                    fold=self.ctx.fold,
                )
        elif phase == "P01_LOAD_DATASET":
            self._hydrate_p01_readonly()
        elif phase == "P02_BUILD_SPLITS":
            if self.ctx.bundle is None:
                self._hydrate_p01_readonly()
            self._hydrate_p02_readonly()
        elif phase == "P03_FIT_PREPROCESSING":
            if self.ctx.split is None:
                self.hydrate_skipped_phase("P02_BUILD_SPLITS")
            self._hydrate_p03_readonly()
        elif phase == "P04_PREPARE_OUTER_AUGMENTATIONS":
            if not self.ctx.X_parts:
                self.hydrate_skipped_phase("P03_FIT_PREPROCESSING")
            self._hydrate_p04_readonly()
        elif phase == "P05_PREPARE_INNER_AUGMENTATION_CACHE":
            if not self.ctx.X_parts:
                self.hydrate_skipped_phase("P03_FIT_PREPROCESSING")
            self._hydrate_p05_readonly()
        elif phase == "P06_RUN_GBDT_HPO":
            self._hydrate_p06_readonly()
        elif phase == "P07_RUN_A0PLUS":
            self._hydrate_p07_readonly()

    def _hydrate_p06_readonly(self) -> None:
        for learner in GBDT_LEARNERS:
            for arm in HPO_ARMS:
                best_path = self.ctx.root / "hpo" / learner / arm / "best.json"
                if not best_path.exists():
                    raise FileNotFoundError(f"cannot hydrate P06: missing {best_path}")
                self.ctx.hpo_best[(learner, arm)] = json.loads(best_path.read_text())

    def _hydrate_p07_readonly(self) -> None:
        for learner in GBDT_LEARNERS:
            best_path = self.ctx.root / "hpo" / learner / "A0plus" / "best.json"
            match_path = self.ctx.root / "hpo" / learner / "A0plus" / "compute_match.json"
            if not best_path.exists():
                raise FileNotFoundError(f"cannot hydrate P07: missing {best_path}")
            self.ctx.hpo_best[(learner, "A0+")] = json.loads(best_path.read_text())
            if match_path.exists():
                self.ctx.a0plus_info[learner] = json.loads(match_path.read_text())

    def _hydrate_p01_readonly(self) -> None:
        """Load dataset into memory without rewriting dataset_manifest.json."""
        man_path = self.ctx.root / "manifests" / "dataset_manifest.json"
        if not man_path.exists():
            raise FileNotFoundError("cannot hydrate P01: manifests/dataset_manifest.json missing")
        # Load payload via same loaders; do not write scientific manifests.
        self._load_dataset_bundle(write_manifest=False)

    def _hydrate_p02_readonly(self) -> None:
        from src.data.splitting import SplitIndices

        man_path = self.ctx.root / "manifests" / "split_manifest.json"
        idx_path = self.ctx.root / "manifests" / "split_indices.npz"
        if not man_path.exists() or not idx_path.exists():
            raise FileNotFoundError("cannot hydrate P02: split manifests missing")
        man = json.loads(man_path.read_text())
        idx = np.load(idx_path)
        split = SplitIndices(
            train=np.asarray(idx["train"]),
            cal_a=np.asarray(idx["cal_a"]),
            cal_b=np.asarray(idx["cal_b"]),
            test=np.asarray(idx["test"]),
            repeat=int(man["repeat"]),
            fold=int(man["fold"]),
            seed=int(man["seed"]),
        )
        split.assert_no_overlap()
        self.ctx.split = split
        self.ctx.fold_seed = int(split.seed)

    def _hydrate_p03_readonly(self) -> None:
        pre_path = self.ctx.root / "manifests" / "preprocessor.pkl"
        if not pre_path.exists():
            raise FileNotFoundError("cannot hydrate P03: preprocessor.pkl missing")
        if self.ctx.bundle is None or self.ctx.split is None or self.ctx.y_bin is None:
            raise RuntimeError("cannot hydrate P03: dataset/split context missing")
        with pre_path.open("rb") as f:
            pre = pickle.load(f)
        X = self.ctx.bundle.X
        split = self.ctx.split
        self.ctx.pre = pre
        self.ctx.X_parts = {
            "train": pre.transform(X.iloc[split.train]),
            "cal_a": pre.transform(X.iloc[split.cal_a]),
            "cal_b": pre.transform(X.iloc[split.cal_b]),
            "test": pre.transform(X.iloc[split.test]),
        }
        self.ctx.y_parts = {
            "train": self.ctx.y_bin[split.train],
            "cal_a": self.ctx.y_bin[split.cal_a],
            "cal_b": self.ctx.y_bin[split.cal_b],
            "test": self.ctx.y_bin[split.test],
        }
        self.ctx.fold_seed = int(split.seed)

    def _hydrate_p04_readonly(self) -> None:
        from src.augmentation.arms import AugmentResult

        self.ctx.outer_aug["A0"] = augment_a0(self.ctx.X_parts["train"], self.ctx.y_parts["train"])
        for arm in ("A1", "A2", "A3"):
            feat = self.ctx.root / "generated" / arm / "features.parquet"
            lab = self.ctx.root / "generated" / arm / "labels.npy"
            man = self.ctx.root / "generated" / arm / "manifest.json"
            if not (feat.exists() and lab.exists() and man.exists()):
                raise FileNotFoundError(f"cannot hydrate P04: missing generated/{arm} artifacts")
            m = json.loads(man.read_text())
            self.ctx.outer_aug[arm] = AugmentResult(
                X=pd.read_parquet(feat),
                y=np.load(lab),
                repair_fraction=float(m.get("repair_row_fraction", 0.0)),
                n_synthetic=int(m.get("n_synthetic", 0)),
                method=str(m.get("method", arm)),
                generator_fit_seconds=float(m.get("generator_fit_seconds", 0.0)),
                generator_sample_seconds=float(m.get("generator_sample_seconds", 0.0)),
                scientific_mark=str(m.get("scientific_mark", "")),
                extra={
                    "repair_cell_fraction": float(m.get("repair_cell_fraction", 0.0)),
                    **(m.get("extra") or {}),
                },
            )
            self.ctx.generator_costs[arm] = float(
                m.get("generator_fit_seconds", 0.0) + m.get("generator_sample_seconds", 0.0)
            )

    def _hydrate_p05_readonly(self) -> None:
        Xtr = self.ctx.X_parts["train"]
        ytr = self.ctx.y_parts["train"]
        skf = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=self.ctx.fold_seed)
        self.ctx.inner_cache = {arm: {} for arm in ("A0", "A1", "A2", "A3")}
        for inner_i, (tr_loc, va_loc) in enumerate(skf.split(Xtr, ytr)):
            X_v = Xtr.iloc[va_loc]
            y_v = ytr[va_loc]
            self.ctx.inner_cache["A0"][inner_i] = {
                "X_train": Xtr.iloc[tr_loc],
                "y_train": ytr[tr_loc],
                "X_val": X_v,
                "y_val": y_v,
            }
            for arm in ("A1", "A2", "A3"):
                path = self.ctx.root / "hpo" / "inner_cache" / arm / f"fold_{inner_i}"
                if not (path / "features.parquet").exists():
                    raise FileNotFoundError(f"cannot hydrate P05: missing {path}")
                self.ctx.inner_cache[arm][inner_i] = {
                    "X_train": pd.read_parquet(path / "features.parquet"),
                    "y_train": np.load(path / "labels.npy"),
                    "X_val": X_v,
                    "y_val": y_v,
                    "seed": json.loads((path / "manifest.json").read_text()).get("seed"),
                }

    # ---- P01 ----
    def p01_load_dataset(self) -> None:
        self._load_dataset_bundle(write_manifest=True)

    def _load_dataset_bundle(self, *, write_manifest: bool) -> None:
        from src.data.openml_loader import load_frozen_openml_raw

        expected_checksum = self.ctx.adapters.expected_checksum
        expected_version = self.ctx.adapters.expected_version
        expected_name = self.ctx.adapters.expected_name
        if self.ctx.validation:
            expected_checksum = expected_checksum or self.ctx.validation.get("expected_raw_checksum")
            expected_version = expected_version if expected_version is not None else self.ctx.validation.get("dataset_version")
            expected_name = expected_name or self.ctx.validation.get("dataset_name")

        if self.ctx.adapters.load_dataset is not None:
            bundle = self.ctx.adapters.load_dataset(int(self.ctx.dataset_id))
            if expected_checksum and bundle.checksum != expected_checksum:
                raise AssertionError(
                    f"dataset checksum mismatch: got {bundle.checksum}, expected {expected_checksum}"
                )
            retrieval_source = "adapter"
        else:
            bundle = load_frozen_openml_raw(
                int(self.ctx.dataset_id),
                expected_raw_checksum=expected_checksum,
                expected_version=expected_version,
                repo_root=self.ctx.repo_root,
            )
            retrieval_source = "frozen_parquet_cache_or_openml"

        if expected_version is not None and int(bundle.version) != int(expected_version):
            raise AssertionError(f"dataset version mismatch: got {bundle.version}, expected {expected_version}")
        if expected_name is not None and str(bundle.name) != str(expected_name) and self.ctx.adapters.scientific:
            raise AssertionError(f"dataset name mismatch: got {bundle.name}, expected {expected_name}")

        if self.ctx.adapters.scientific and int(bundle.openml_id) != int(self.ctx.dataset_id):
            raise AssertionError("OpenML ID mismatch")

        y_bin, y_meta = binarize_labels(bundle.y)
        row_ids = np.arange(len(bundle.X), dtype=np.int64)
        vals, counts = np.unique(y_bin, return_counts=True)
        class_counts = {int(v): int(c) for v, c in zip(vals, counts)}

        self.ctx.bundle = bundle
        self.ctx.y_bin = y_bin
        self.ctx.y_meta = y_meta
        self.ctx.row_ids = row_ids

        if write_manifest:
            manifest = {
                "dataset_id": int(bundle.openml_id),
                "dataset_version": bundle.version,
                "dataset_name": bundle.name,
                "target": y_meta,
                "raw_checksum": bundle.checksum,
                "checksum_algorithm": "sha256(X.parquet_bytes||y.parquet_bytes)",
                "n_rows": int(len(bundle.X)),
                "n_features": int(bundle.X.shape[1]),
                "class_counts": class_counts,
                "retrieval_source": retrieval_source,
                "row_id_policy": "stable_integer_index_0_n_minus_1_before_split",
                "scientific": self.ctx.adapters.scientific,
                "mark": self.ctx.adapters.mark,
            }
            write_json(self.ctx.root / "manifests" / "dataset_manifest.json", manifest)

    # ---- P02 ----
    def p02_build_splits(self) -> None:
        assert self.ctx.y_bin is not None and self.ctx.row_ids is not None
        split = build_split_for_fold(
            self.ctx.y_bin,
            n_splits=5,
            n_repeats=2,
            seed=42,
            repeat_index=self.ctx.repeat,
            fold_index=self.ctx.fold,
        )
        split.assert_no_overlap()
        self.ctx.split = split
        self.ctx.fold_seed = int(split.seed)

        def _part_hash(idx: np.ndarray) -> str:
            return _sha256_array(np.sort(idx.astype(np.int64)))

        manifest = {
            "repeat": split.repeat,
            "fold": split.fold,
            "seed": split.seed,
            "counts": {
                "train": int(len(split.train)),
                "cal_a": int(len(split.cal_a)),
                "cal_b": int(len(split.cal_b)),
                "test": int(len(split.test)),
            },
            "row_id_hashes": {
                "train": _part_hash(split.train),
                "cal_a": _part_hash(split.cal_a),
                "cal_b": _part_hash(split.cal_b),
                "test": _part_hash(split.test),
            },
            "row_ids": {
                "train": split.train.tolist(),
                "cal_a": split.cal_a.tolist(),
                "cal_b": split.cal_b.tolist(),
                "test": split.test.tolist(),
            },
        }
        write_json(self.ctx.root / "manifests" / "split_manifest.json", manifest)
        np.savez_compressed(
            self.ctx.root / "manifests" / "split_indices.npz",
            train=split.train,
            cal_a=split.cal_a,
            cal_b=split.cal_b,
            test=split.test,
        )

    # ---- P03 ----
    def p03_fit_preprocessing(self) -> None:
        assert self.ctx.bundle is not None and self.ctx.split is not None and self.ctx.y_bin is not None
        X = self.ctx.bundle.X
        split = self.ctx.split
        pre = fit_preprocessor(X.iloc[split.train], unknown_category_sentinel=-1)
        parts = {
            "train": pre.transform(X.iloc[split.train]),
            "cal_a": pre.transform(X.iloc[split.cal_a]),
            "cal_b": pre.transform(X.iloc[split.cal_b]),
            "test": pre.transform(X.iloc[split.test]),
        }
        for name, frame in parts.items():
            if frame.isna().any().any():
                raise AssertionError(f"NaNs after preprocessing in {name}")
        if len(pre.meta.output_columns) > 100:
            raise AssertionError("encoded feature count exceeds frozen limit 100")

        self.ctx.pre = pre
        self.ctx.X_parts = parts
        self.ctx.y_parts = {
            "train": self.ctx.y_bin[split.train],
            "cal_a": self.ctx.y_bin[split.cal_a],
            "cal_b": self.ctx.y_bin[split.cal_b],
            "test": self.ctx.y_bin[split.test],
        }

        meta = pre.meta
        meta_payload = {
            "numeric_cols": meta.numeric_cols,
            "categorical_cols": meta.categorical_cols,
            "missing_indicator_cols": meta.missing_indicator_cols,
            "numeric_medians": meta.numeric_medians,
            "categorical_modes": {k: str(v) for k, v in meta.categorical_modes.items()},
            "category_levels": {k: [str(x) for x in v] for k, v in meta.category_levels.items()},
            "unknown_category_sentinel": meta.unknown_category_sentinel,
            "output_columns": meta.output_columns,
            "n_output_features": len(meta.output_columns),
            "fitted_on": "OUTER_TRAIN_only",
        }
        meta_hash = _sha256_bytes(json.dumps(meta_payload, sort_keys=True).encode())
        meta_payload["metadata_hash"] = meta_hash
        write_json(self.ctx.root / "manifests" / "preprocessing_manifest.json", meta_payload)
        with (self.ctx.root / "manifests" / "preprocessor.pkl").open("wb") as f:
            pickle.dump(pre, f)

    # ---- P04 ----
    def p04_outer_augmentations(self) -> None:
        Xtr = self.ctx.X_parts["train"]
        ytr = self.ctx.y_parts["train"]
        meta = self.ctx.pre.meta
        seed = self.ctx.fold_seed

        self.ctx.outer_aug["A0"] = augment_a0(Xtr, ytr)

        a1 = augment_a1(Xtr, ytr, random_state=seed)
        assert_balanced_to_majority(a1.y)
        self.ctx.outer_aug["A1"] = a1
        self._persist_aug("A1", a1, seed)

        a2 = augment_a2(Xtr, ytr, meta, random_state=seed, k_neighbors=5)
        assert_balanced_to_majority(a2.y)
        n_real = len(ytr)
        if a2.n_synthetic:
            _, row_f, cell_f = shared_repair_with_cell_stats(a2.X.iloc[n_real:], Xtr, meta)
            a2.extra = {**(a2.extra or {}), "repair_row_fraction": row_f, "repair_cell_fraction": cell_f}
            a2.repair_fraction = row_f
        self.ctx.outer_aug["A2"] = a2
        self._persist_aug("A2", a2, seed)
        self.ctx.generator_costs["A2"] = float(a2.generator_fit_seconds + a2.generator_sample_seconds)

        a3_fn = self.ctx.adapters.resolve_augment_a3()
        job_dir = self.ctx.root / "generated" / "A3" / "job"
        if self.ctx.adapters.augment_a3 is not None:
            a3 = a3_fn(Xtr, ytr, meta, random_state=seed, job_dir=job_dir)
        else:
            a3_cfg = self._a3_protocol_kwargs(seed)
            a3 = a3_fn(
                Xtr,
                ytr,
                meta,
                job_dir=job_dir,
                synthcity_python=str(self.ctx.repo_root / SYNTHCITY_PYTHON),
                random_state=seed,
                smoke_config=a3_cfg,
            )
            if a3.scientific_mark and "SMOKE_ONLY" in str(a3.scientific_mark) and self.ctx.adapters.scientific:
                raise AssertionError("A3 returned smoke mark in confirmatory scientific mode")
        assert_balanced_to_majority(a3.y)
        if a3.extra is None:
            a3.extra = {}
        if "repair_cell_fraction" not in a3.extra and a3.n_synthetic:
            _, row_f, cell_f = shared_repair_with_cell_stats(a3.X.iloc[n_real:], Xtr, meta)
            a3.extra["repair_row_fraction"] = row_f
            a3.extra["repair_cell_fraction"] = cell_f
            a3.repair_fraction = row_f
        self.ctx.outer_aug["A3"] = a3
        self._persist_aug("A3", a3, seed)
        self.ctx.generator_costs["A3"] = float(a3.generator_fit_seconds + a3.generator_sample_seconds)

    def _a3_protocol_kwargs(self, seed: int) -> dict[str, Any]:
        return {
            "is_classification": True,
            "n_iter": 1000,
            "lr": 0.002,
            "weight_decay": 0.0001,
            "batch_size": 1024,
            "num_timesteps": 1000,
            "gaussian_loss_type": "mse",
            "scheduler": "cosine",
            "model_type": "mlp",
            "model_params": {},
            "dim_embed": 128,
            "continuous_encoder": "quantile",
            "cont_encoder_params": {},
            "validation_size": 0,
            "compress_dataset": False,
            "sampling_patience": 500,
            "device": "cuda",
            "random_state": seed,
            "scientific_mode": True,
            "config_sha256": A3_CONFIG_SHA256,
            "mark": "",
        }

    def _persist_aug(self, arm: str, aug, seed: int) -> None:
        out = self.ctx.root / "generated" / arm
        out.mkdir(parents=True, exist_ok=True)
        aug.X.to_parquet(out / "features.parquet")
        np.save(out / "labels.npy", aug.y)
        maj_unused = None  # silence lint; counts_map below is source of truth
        del maj_unused
        vals, counts = np.unique(aug.y, return_counts=True)
        counts_map = {int(v): int(c) for v, c in zip(vals, counts)}
        cell_frac = (aug.extra or {}).get("repair_cell_fraction", 0.0)
        man = {
            "arm": arm,
            "method": aug.method,
            "seed": seed,
            "checksum_features": _sha256_frame(aug.X),
            "checksum_labels": _sha256_array(np.asarray(aug.y)),
            "n_synthetic": int(aug.n_synthetic),
            "class_counts": counts_map,
            "repair_row_fraction": float(aug.repair_fraction),
            "repair_cell_fraction": float(cell_frac),
            "generator_fit_seconds": float(aug.generator_fit_seconds),
            "generator_sample_seconds": float(aug.generator_sample_seconds),
            "scientific_mark": aug.scientific_mark,
            "shared_across_learners": list(LEARNERS),
            "extra": aug.extra or {},
        }
        write_json(out / "manifest.json", man)

    # ---- P05 ----
    def p05_inner_cache(self) -> None:
        Xtr = self.ctx.X_parts["train"]
        ytr = self.ctx.y_parts["train"]
        meta = self.ctx.pre.meta
        skf = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=self.ctx.fold_seed)
        self.ctx.inner_cache = {arm: {} for arm in ("A0", "A1", "A2", "A3")}

        for inner_i, (tr_loc, va_loc) in enumerate(skf.split(Xtr, ytr)):
            X_i = Xtr.iloc[tr_loc]
            y_i = ytr[tr_loc]
            X_v = Xtr.iloc[va_loc]
            y_v = ytr[va_loc]

            self.ctx.inner_cache["A0"][inner_i] = {
                "X_train": X_i,
                "y_train": y_i,
                "X_val": X_v,
                "y_val": y_v,
            }

            for arm in ("A1", "A2", "A3"):
                seed = derive_generator_seed(
                    dataset_id=self.ctx.dataset_id,
                    repeat=self.ctx.repeat,
                    fold=self.ctx.fold,
                    arm=arm,
                    inner_fold=inner_i,
                )
                if arm == "A1":
                    aug = augment_a1(X_i, y_i, random_state=seed)
                elif arm == "A2":
                    aug = augment_a2(X_i, y_i, meta, random_state=seed, k_neighbors=5)
                else:
                    a3_fn = self.ctx.adapters.resolve_augment_a3()
                    job_dir = self.ctx.root / "hpo" / "inner_cache" / "A3" / f"fold_{inner_i}" / "job"
                    if self.ctx.adapters.augment_a3 is not None:
                        aug = a3_fn(X_i, y_i, meta, random_state=seed, job_dir=job_dir)
                    else:
                        aug = a3_fn(
                            X_i,
                            y_i,
                            meta,
                            job_dir=job_dir,
                            synthcity_python=str(self.ctx.repo_root / SYNTHCITY_PYTHON),
                            random_state=seed,
                            smoke_config=self._a3_protocol_kwargs(seed),
                        )
                assert_balanced_to_majority(aug.y)
                path = self.ctx.root / "hpo" / "inner_cache" / arm / f"fold_{inner_i}"
                path.mkdir(parents=True, exist_ok=True)
                aug.X.to_parquet(path / "features.parquet")
                np.save(path / "labels.npy", aug.y)
                write_json(
                    path / "manifest.json",
                    {
                        "arm": arm,
                        "inner_fold": inner_i,
                        "seed": seed,
                        "checksum": _sha256_frame(aug.X),
                        "shared_across_gbdt_learners": True,
                        "n_train": int(len(aug.y)),
                    },
                )
                self.ctx.inner_cache[arm][inner_i] = {
                    "X_train": aug.X,
                    "y_train": aug.y,
                    "X_val": X_v,
                    "y_val": y_v,
                    "seed": seed,
                }

    # ---- P06 ----
    def p06_gbdt_hpo(self) -> None:
        hpo_cfg = load_hpo_config(str(self.ctx.repo_root / "configs/hpo_v1_2_1.yaml"))
        n_cand = STANDARD_HPO_CANDIDATES
        if self.ctx.adapters.hpo_candidate_limit is not None:
            n_cand = int(self.ctx.adapters.hpo_candidate_limit)

        for learner in GBDT_LEARNERS:
            seq = shared_arm_candidates(
                dataset_id=self.ctx.dataset_id,
                repeat=self.ctx.repeat,
                fold=self.ctx.fold,
                learner=learner,
                n_candidates=n_cand,
                cfg=hpo_cfg,
            )
            self.ctx.a0_hpo_timings[learner] = []
            for arm in HPO_ARMS:
                rows = []
                metrics_for_select = []
                for idx, params in enumerate(seq):
                    fold_scores = []
                    t0 = time.perf_counter()
                    for inner_i in range(INNER_CV_FOLDS):
                        cache = self.ctx.inner_cache[arm][inner_i]
                        score = self._eval_hpo_fold(learner, params, cache)
                        fold_scores.append(score)
                    elapsed = time.perf_counter() - t0
                    if arm == "A0":
                        self.ctx.a0_hpo_timings[learner].append(elapsed)
                    mean_ll = float(np.mean(fold_scores))
                    std_ll = float(np.std(fold_scores, ddof=0))
                    rows.append(
                        {
                            "candidate_index": idx,
                            "hyperparameters": json.dumps(params, sort_keys=True),
                            "inner_fold_scores": json.dumps(fold_scores),
                            "mean_log_loss": mean_ll,
                            "std_log_loss": std_ll,
                            "timings": elapsed,
                            "status": "ok",
                        }
                    )
                    metrics_for_select.append(
                        {
                            "candidate_index": idx,
                            "mean_inner_log_loss": mean_ll,
                            "std_inner_log_loss": std_ll,
                        }
                    )
                best_idx = select_hpo_candidate(metrics_for_select)
                best_params = seq[best_idx]
                out_dir = self.ctx.root / "hpo" / learner / arm
                out_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(rows)
                df.to_parquet(out_dir / "candidates.parquet")
                best = {
                    "learner": learner,
                    "arm": arm,
                    "best_candidate_index": best_idx,
                    "best_hyperparameters": best_params,
                    "n_candidates": len(seq),
                    "selection": ["min_mean", "min_std", "min_index"],
                }
                write_json(out_dir / "best.json", best)
                self.ctx.hpo_best[(learner, arm)] = best

    def _eval_hpo_fold(self, learner: str, params: dict[str, Any], cache: dict[str, Any]) -> float:
        if self.ctx.adapters.evaluate_hpo_fold is not None:
            return float(self.ctx.adapters.evaluate_hpo_fold(learner, params, cache))
        build = self.ctx.adapters.resolve_build_learner()
        # strip protocol-only keys that wrappers may not accept
        clean = {k: v for k, v in params.items() if k not in {"native_categorical_feature_declaration", "random_state", "random_seed"}}
        if learner == "xgboost":
            clean["random_state"] = self.ctx.fold_seed
            cfg = {"xgboost": clean}
        else:
            clean["random_seed"] = self.ctx.fold_seed
            cfg = {"catboost": clean}
        model = build(learner, cfg, cat_features=self.ctx.pre.meta.categorical_cols)
        model.fit(cache["X_train"], cache["y_train"])
        p = model.predict_proba(cache["X_val"])
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return float(log_loss(cache["y_val"], np.column_stack([1 - p, p]), labels=[0, 1]))

    # ---- P07 ----
    def p07_a0plus(self) -> None:
        for learner in GBDT_LEARNERS:
            g_max = max(self.ctx.generator_costs.get("A2", 0.0), self.ctx.generator_costs.get("A3", 0.0))
            timings = self.ctx.a0_hpo_timings.get(learner) or [1e-3]
            t_bar = float(np.mean(timings[:STANDARD_HPO_CANDIDATES]))
            if t_bar <= 0:
                t_bar = 1e-6
            k_extra = compute_k_extra(g_max=g_max, t_bar=t_bar, cap=200)
            if self.ctx.adapters.hpo_candidate_limit is not None:
                k_extra = min(k_extra, 2)  # keep toy runs tiny while preserving continuation logic
            total = STANDARD_HPO_CANDIDATES + k_extra
            if self.ctx.adapters.hpo_candidate_limit is not None:
                # toy: continue from limited prefix
                base = self.ctx.adapters.hpo_candidate_limit
                total = base + k_extra
                seq = a0_plus_candidates(
                    dataset_id=self.ctx.dataset_id,
                    repeat=self.ctx.repeat,
                    fold=self.ctx.fold,
                    learner=learner,
                    k_extra=k_extra,
                )
                # ensure A0 prefix identity: first base of A0+ equals A0's base
                seq = seq[:total]
                start_idx = base
            else:
                seq = a0_plus_candidates(
                    dataset_id=self.ctx.dataset_id,
                    repeat=self.ctx.repeat,
                    fold=self.ctx.fold,
                    learner=learner,
                    k_extra=k_extra,
                )
                start_idx = STANDARD_HPO_CANDIDATES

            # verify prefix: first 20 of A0+ match A0 sequence
            a0_seq = shared_arm_candidates(
                dataset_id=self.ctx.dataset_id,
                repeat=self.ctx.repeat,
                fold=self.ctx.fold,
                learner=learner,
                n_candidates=min(STANDARD_HPO_CANDIDATES, len(seq)),
            )
            if seq[: len(a0_seq)] != a0_seq:
                raise AssertionError("A0+ candidate sequence is not a prefix extension of A0")

            rows = []
            metrics_for_select = []
            for idx, params in enumerate(seq):
                fold_scores = []
                t0 = time.perf_counter()
                for inner_i in range(INNER_CV_FOLDS):
                    cache = self.ctx.inner_cache["A0"][inner_i]
                    score = self._eval_hpo_fold(learner, params, cache)
                    fold_scores.append(score)
                elapsed = time.perf_counter() - t0
                mean_ll = float(np.mean(fold_scores))
                std_ll = float(np.std(fold_scores, ddof=0))
                rows.append(
                    {
                        "candidate_index": idx,
                        "hyperparameters": json.dumps(params, sort_keys=True),
                        "inner_fold_scores": json.dumps(fold_scores),
                        "mean_log_loss": mean_ll,
                        "std_log_loss": std_ll,
                        "timings": elapsed,
                        "status": "ok",
                        "is_extra": idx >= start_idx,
                    }
                )
                metrics_for_select.append(
                    {
                        "candidate_index": idx,
                        "mean_inner_log_loss": mean_ll,
                        "std_inner_log_loss": std_ll,
                    }
                )
            best_idx = select_hpo_candidate(metrics_for_select)
            out_dir = self.ctx.root / "hpo" / learner / "A0plus"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(out_dir / "candidates.parquet")
            compute_match = {
                "learner": learner,
                "g_max": g_max,
                "t_bar": t_bar,
                "k_extra": k_extra,
                "total_configs": len(seq),
                "standard_prefix": start_idx,
                "generator_costs": dict(self.ctx.generator_costs),
            }
            write_json(out_dir / "compute_match.json", compute_match)
            best = {
                "learner": learner,
                "arm": "A0+",
                "best_candidate_index": best_idx,
                "best_hyperparameters": seq[best_idx],
                "n_candidates": len(seq),
            }
            write_json(out_dir / "best.json", best)
            self.ctx.hpo_best[(learner, "A0+")] = best
            self.ctx.a0plus_info[learner] = compute_match

    # ---- P08 ----
    def p08_final_learners(self) -> None:
        import warnings

        cells = expected_final_cells()
        build = self.ctx.adapters.resolve_build_learner()
        for cell in cells:
            learner, arm = cell["learner"], cell["arm"]
            if arm == "A0":
                train_X, train_y = self.ctx.X_parts["train"], self.ctx.y_parts["train"]
                aug_checksum = _sha256_frame(train_X)
            elif arm == "A0+":
                train_X, train_y = self.ctx.X_parts["train"], self.ctx.y_parts["train"]
                aug_checksum = _sha256_frame(train_X)
            else:
                aug = self.ctx.outer_aug[arm]
                train_X, train_y = aug.X, aug.y
                aug_checksum = _sha256_frame(train_X)

            ckpt_sha256 = None
            ckpt_prefix = None
            if learner in GBDT_LEARNERS:
                best = self.ctx.hpo_best[(learner, arm)]
                params = dict(best["best_hyperparameters"])
                params.pop("native_categorical_feature_declaration", None)
                if learner == "xgboost":
                    params["random_state"] = self.ctx.fold_seed
                    cfg = {"xgboost": params}
                else:
                    params["random_seed"] = self.ctx.fold_seed
                    cfg = {"catboost": params}
                hpo_idx = best["best_candidate_index"]
                hpo_count = best["n_candidates"]
            else:
                # TFM frozen identities — verify FULL-FILE checkpoint SHA before execution
                ckpt = self.ctx.repo_root / CHECKPOINTS[learner]
                ckpt_sha256, ckpt_prefix = self._verify_tfm_checkpoint(learner, ckpt)
                cfg = {
                    learner: {
                        "model_path": str(ckpt),
                        "n_estimators": 8,
                        "softmax_temperature": 1.0,
                        **(
                            {
                                "inference_precision": "float32",
                                "n_preprocessing_jobs": 1,
                                "random_state": 0,
                                "ignore_pretraining_limits": False,
                            }
                            if learner == "tabpfn"
                            else {
                                "checkpoint_version": "tabicl-classifier-v1-20250208.ckpt",
                                "use_amp": False,
                                "batch_size": 8,
                                "average_logits": True,
                                "random_state": 42,
                            }
                        ),
                    }
                }
                hpo_idx = None
                hpo_count = 0

            captured: list[str] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = build(learner, cfg, cat_features=self.ctx.pre.meta.categorical_cols)
                with timed_gpu() as fit_t:
                    model.fit(train_X, train_y)
                for w in caught:
                    msg = str(w.message)
                    if "device" in msg.lower() or "FutureWarning" in w.category.__name__ or "auto_scale" in msg:
                        captured.append(f"{w.category.__name__}: {msg}")
            self.ctx.fitted[(learner, arm)] = model

            man = {
                "learner": learner,
                "family": getattr(model, "family", "unknown"),
                "arm": arm,
                "training_artifact_checksum": aug_checksum,
                "selected_hyperparameters": cfg.get(learner, cfg),
                "checkpoint_sha256": ckpt_sha256,
                "checkpoint_prefix_1mb_sha256": ckpt_prefix,
                "learner_fit_seconds": fit_t.seconds,
                "peak_gpu_memory_mb": fit_t.peak_gpu_memory_mb,
                "seed": self.ctx.fold_seed,
                "environment": self.ctx.adapters.resource_snapshot(),
                "hpo_candidate_count": hpo_count,
                "best_hpo_candidate_index": hpo_idx,
                "runtime_warnings": captured,
                "status": "ok",
                "scientific_status": SCIENTIFIC_STATUS_CONFIRMATORY if self.ctx.adapters.scientific else self.ctx.adapters.mark,
            }
            write_json(self.ctx.root / "manifests" / "cells" / f"{cell_manifest_stem(learner, arm)}.json", man)

    def _verify_tfm_checkpoint(self, learner: str, path: Path) -> tuple[str, str]:
        if not path.exists():
            raise AssertionError(f"missing frozen checkpoint for {learner}: {path}")
        data = path.read_bytes()
        full = _sha256_bytes(data)
        expected = CHECKPOINT_SHA256[learner]
        if full != expected:
            raise AssertionError(
                f"{learner} checkpoint SHA256 mismatch: got {full}, expected {expected}"
            )
        prefix = _sha256_bytes(data[: 1024 * 1024])
        return full, prefix

    # ---- P09 ----
    def p09_predict_cal_a(self) -> None:
        X = self.ctx.X_parts["cal_a"]
        row_ids = self.ctx.split.cal_a
        for learner, arm in self.ctx.fitted:
            model = self.ctx.fitted[(learner, arm)]
            guard = self.ctx.guard.get(learner, arm)
            with timed_gpu(reset=False) as _:
                if learner in ("tabpfn", "tabicl"):
                    preds = {}
                    for label, temp in TEMPERATURES.items():
                        # if API exposes softmax_temperature, set; else reuse same mock probs
                        if hasattr(model, "params"):
                            model.params["softmax_temperature"] = temp
                        elif hasattr(model, "model") and hasattr(model.model, "softmax_temperature"):
                            model.model.softmax_temperature = temp
                        preds[label] = np.asarray(model.predict_proba(X), dtype=np.float64)
                    # document: primary re-uses fit; temperature applied at predict when API permits
                    self.ctx.cal_a_preds[(learner, arm)] = preds
                else:
                    p = np.asarray(model.predict_proba(X), dtype=np.float64)
                    self.ctx.cal_a_preds[(learner, arm)] = {"primary": p}
            guard.mark_cal_a_complete()
            out = self.ctx.root / "predictions" / learner / arm_to_fs_name(arm)
            out.mkdir(parents=True, exist_ok=True)
            payload = {"row_ids": row_ids.tolist()}
            for k, v in self.ctx.cal_a_preds[(learner, arm)].items():
                payload[f"p_{k}"] = v.tolist()
            write_json(out / "cal_a.json", payload)

    # ---- P10 ----
    def p10_calibrators(self) -> None:
        y = self.ctx.y_parts["cal_a"]
        for (learner, arm), preds in self.ctx.cal_a_preds.items():
            p_raw = preds["primary"]
            platt = PlattCalibrator().fit(p_raw, y)
            iso = IsotonicCalibrator().fit(p_raw, y)
            thr = tune_threshold_f1(y, p_raw)
            out = self.ctx.root / "postprocessing" / learner / arm_to_fs_name(arm)
            out.mkdir(parents=True, exist_ok=True)
            write_json(
                out / "platt.json",
                {
                    "coef": platt.model.coef_.ravel().tolist(),
                    "intercept": platt.model.intercept_.ravel().tolist(),
                },
            )
            with (out / "isotonic.pkl").open("wb") as f:
                pickle.dump(iso, f)
            write_json(out / "threshold.json", {"threshold": thr, "tie_rule": "smallest_threshold"})
            self.ctx.post[(learner, arm)] = {"platt": platt, "isotonic": iso, "threshold": thr}
            self.ctx.guard.get(learner, arm).mark_calibrators_complete()

    # ---- P11 ----
    def p11_predict_cal_b(self) -> None:
        X = self.ctx.X_parts["cal_b"]
        row_ids = self.ctx.split.cal_b
        for (learner, arm), model in self.ctx.fitted.items():
            # restore primary temperature for TFMs
            if hasattr(model, "params"):
                model.params["softmax_temperature"] = 1.0
            p = np.asarray(model.predict_proba(X), dtype=np.float64)
            self.ctx.cal_b_preds[(learner, arm)] = p
            out = self.ctx.root / "predictions" / learner / arm_to_fs_name(arm)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "cal_b.json", {"row_ids": row_ids.tolist(), "p_raw": p.tolist()})
            self.ctx.guard.get(learner, arm).mark_cal_b_complete()

    # ---- P12 ----
    def p12_conformal(self) -> None:
        y = self.ctx.y_parts["cal_b"]
        for (learner, arm), p in self.ctx.cal_b_preds.items():
            scores = conformity_score(p, y)
            qhat, meta = conformal_quantile_with_meta(scores, alpha=0.10)
            meta["quantile"] = qhat
            meta["score_checksum"] = _sha256_array(scores)
            out = self.ctx.root / "postprocessing" / learner / arm_to_fs_name(arm)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "conformal.json", meta)
            self.ctx.conformal[(learner, arm)] = meta
            self.ctx.guard.get(learner, arm).mark_conformal_complete()

    # ---- P13 ----
    def p13_predict_test(self) -> None:
        X = self.ctx.X_parts["test"]
        row_ids = self.ctx.split.test
        for (learner, arm), model in self.ctx.fitted.items():
            guard = self.ctx.guard.get(learner, arm)
            guard.unlock_test()
            guard.assert_test_allowed()
            if learner in ("tabpfn", "tabicl"):
                preds = {}
                for label, temp in TEMPERATURES.items():
                    if hasattr(model, "params"):
                        model.params["softmax_temperature"] = temp
                    preds[label] = np.asarray(model.predict_proba(X), dtype=np.float64)
                self.ctx.test_preds[(learner, arm)] = preds
            else:
                p = np.asarray(model.predict_proba(X), dtype=np.float64)
                self.ctx.test_preds[(learner, arm)] = {"primary": p}
            out = self.ctx.root / "predictions" / learner / arm_to_fs_name(arm)
            out.mkdir(parents=True, exist_ok=True)
            payload = {"row_ids": row_ids.tolist(), "test_unlocked_at": guard.test_unlocked_at}
            for k, v in self.ctx.test_preds[(learner, arm)].items():
                payload[f"p_{k}"] = v.tolist()
            write_json(out / "test.json", payload)
            # Do not mutate P08 cell manifests (provenance). Unlock time lives in test.json.

    # ---- P14 ----
    def p14_metrics(self) -> None:
        y_test = self.ctx.y_parts["test"]
        rows = []
        for (learner, arm), preds in self.ctx.test_preds.items():
            post = self.ctx.post[(learner, arm)]
            p_raw = preds["primary"]
            p_platt = post["platt"].transform(p_raw)
            p_iso = post["isotonic"].transform(p_raw)
            thr = post["threshold"]
            qhat = self.ctx.conformal[(learner, arm)]["quantile"]
            sets = prediction_sets(p_raw, qhat)
            cmet = set_metrics(sets, y_test)
            met = compute_metrics(
                y_test,
                p_raw,
                p_platt,
                threshold=thr,
                conformal_set_size=cmet["conformal_set_size"],
                conformal_coverage=cmet["conformal_coverage"],
            )
            # isotonic log loss
            from sklearn.metrics import log_loss as sk_ll
            from src.calibration.posthoc import clip_prob

            p_iso_c = clip_prob(p_iso)
            ll_iso = float(sk_ll(y_test, np.column_stack([1 - p_iso_c, p_iso_c]), labels=[0, 1]))

            cell_man = self.ctx.root / "manifests" / "cells" / f"{cell_manifest_stem(learner, arm)}.json"
            cell = json.loads(cell_man.read_text()) if cell_man.exists() else {}
            aug = self.ctx.outer_aug.get("A0" if arm in {"A0", "A0+"} else arm)
            repair_row = float(getattr(aug, "repair_fraction", 0.0)) if aug else 0.0
            repair_cell = float((getattr(aug, "extra", None) or {}).get("repair_cell_fraction", 0.0)) if aug else 0.0
            n_train_real = int(len(self.ctx.y_parts["train"]))
            ytr = self.ctx.y_parts["train"]
            n_min = int(ytr.sum())
            n_maj = int(len(ytr) - n_min)

            if arm in {"A0", "A0+"}:
                n_final = n_train_real
                prev_final = float(ytr.mean())
            else:
                n_final = int(len(aug.y))
                prev_final = float(np.mean(aug.y))

            family = "gbdt" if learner in GBDT_LEARNERS else "tfm"
            row = {c: None for c in CONFIRMATORY_RESULT_COLUMNS}
            row.update(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "freeze_tag": FREEZE_TAG,
                    "cohort_sha256": (self.ctx.validation or {}).get("cohort_sha256"),
                    "unit_id": self.ctx.root.name,
                    "dataset_id": self.ctx.dataset_id,
                    "dataset_version": getattr(self.ctx.bundle, "version", None),
                    "repeat": self.ctx.repeat,
                    "fold": self.ctx.fold,
                    "seed": self.ctx.fold_seed,
                    "learner": learner,
                    "family": family,
                    "arm": arm,
                    "temperature": 1.0,
                    "calibration": "raw",
                    "scientific_status": SCIENTIFIC_STATUS_CONFIRMATORY if self.ctx.adapters.scientific else self.ctx.adapters.mark,
                    "n_train_real": n_train_real,
                    "n_train_final": n_final,
                    "n_majority_real": n_maj,
                    "n_minority_real": n_min,
                    "minority_prevalence_real": float(ytr.mean()),
                    "minority_prevalence_final": prev_final,
                    "hpo_candidate_count": cell.get("hpo_candidate_count"),
                    "best_hpo_candidate_index": cell.get("best_hpo_candidate_index"),
                    "log_loss": met["log_loss_raw"],
                    "auroc": met["auroc"],
                    "calibration_slope": met["calibration_slope"],
                    "calibration_intercept": met["calibration_intercept"],
                    "conformal_set_size": met["conformal_set_size"],
                    "conformal_coverage": met["conformal_coverage"],
                    "f1_tuned": met["f1_tuned"],
                    "brier": met["brier"],
                    "auprc": met["auprc"],
                    "repair_row_fraction": repair_row,
                    "repair_cell_fraction": repair_cell,
                    "generator_fit_seconds": float(getattr(aug, "generator_fit_seconds", 0.0)) if aug else 0.0,
                    "generator_sample_seconds": float(getattr(aug, "generator_sample_seconds", 0.0)) if aug else 0.0,
                    "hpo_seconds": None,
                    "learner_fit_seconds": cell.get("learner_fit_seconds"),
                    "inference_seconds": None,
                    "peak_gpu_memory_mb": cell.get("peak_gpu_memory_mb"),
                    "manifest_path": str(cell_man),
                    "status": "ok",
                    "failure_reason": "",
                }
            )
            # store supplementary platt/isotonic as separate calibration rows? schema has one calibration field.
            # Keep primary raw row; write sidecar metrics with platt/isotonic log loss.
            write_json(
                self.ctx.root / "metrics" / f"{cell_manifest_stem(learner, arm)}.json",
                {
                    **met,
                    "log_loss_raw": met["log_loss_raw"],
                    "log_loss_platt": met["log_loss_platt"],
                    "log_loss_isotonic": ll_iso,
                    "threshold": thr,
                    "qhat": qhat,
                },
            )
            rows.append(row)
        self.ctx.result_rows = rows
        df = pd.DataFrame(rows, columns=CONFIRMATORY_RESULT_COLUMNS)
        out = self.ctx.root / "metrics" / "results.parquet"
        df.to_parquet(out)
        df.to_csv(out.with_suffix(".csv"), index=False)

    # ---- P15 ----
    def p15_validate_outputs(self) -> None:
        expected = {(c["learner"], c["arm"]) for c in expected_final_cells()}
        got = {(r["learner"], r["arm"]) for r in self.ctx.result_rows}
        if got != expected:
            raise AssertionError(f"cell set mismatch: missing={expected-got} extra={got-expected}")
        if len(self.ctx.result_rows) != 18:
            raise AssertionError(f"expected 18 cells, got {len(self.ctx.result_rows)}")
        for r in self.ctx.result_rows:
            if self.ctx.adapters.scientific:
                if r["scientific_status"] != SCIENTIFIC_STATUS_CONFIRMATORY:
                    raise AssertionError("scientific_status must be CONFIRMATORY")
            if "SMOKE_ONLY" in str(r.get("scientific_status", "")):
                raise AssertionError("smoke flag in confirmatory result")
            if r["status"] != "ok":
                raise AssertionError(f"cell not ok: {r}")
        for (learner, arm), guard in self.ctx.guard.cells.items():
            if guard.test_unlocked_at is None:
                raise AssertionError(f"TEST never unlocked for {learner}/{arm}")
        report = {
            "n_cells": len(self.ctx.result_rows),
            "expected_cells": 18,
            "duplicate_cells": False,
            "scientific_status_ok": True,
            "smoke_flags": False,
            "test_access_ok": True,
            "result_table": str(self.ctx.root / "metrics" / "results.parquet"),
            "result_checksum": _sha256_bytes((self.ctx.root / "metrics" / "results.parquet").read_bytes()),
        }
        write_json(self.ctx.root / "manifests" / "validation_report.json", report)

    # ---- P16 ----
    def p16_finalize(self) -> None:
        """Write unit_complete with P00–P15 hashes only (non-circular P16 rule).

        P16 checksum is recorded afterwards in status/finalization_record.json and
        unit_status by the orchestrator — see FINALIZATION_HASH_RULE.
        """
        report_path = self.ctx.root / "manifests" / "validation_report.json"
        if not report_path.exists():
            raise AssertionError("validation_report.json missing")
        phase_hashes = {}
        status_path = self.ctx.root / "status" / "unit_status.json"
        if status_path.exists():
            st = json.loads(status_path.read_text())
            for p, rec in st.get("phases", {}).items():
                if p == "P16_FINALIZE_UNIT":
                    continue  # avoid null/circular P16 entry inside unit_complete
                phase_hashes[p] = (rec.get("output_hashes") or {}).get("checksum")
        cell_statuses = {
            f"{r['learner']}_{r['arm']}": r["status"] for r in self.ctx.result_rows
        }
        result_path = self.ctx.root / "metrics" / "results.parquet"
        complete = {
            "unit_id": self.ctx.root.name,
            "protocol_version": PROTOCOL_VERSION,
            "freeze_tag": FREEZE_TAG,
            "cohort_sha256": (self.ctx.validation or {}).get("cohort_sha256"),
            "phase_hashes": phase_hashes,
            "phase_hashes_scope": "P00_P15",
            "finalization_hash_rule": FINALIZATION_HASH_RULE,
            "cell_statuses": cell_statuses,
            "result_table_checksum": _sha256_bytes(result_path.read_bytes()),
            "completion_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE",
            "scientific": self.ctx.adapters.scientific,
            "mark": self.ctx.adapters.mark,
        }
        write_json(self.ctx.root / "status" / "unit_complete.json", complete)
        # aggregation-safe index entry (no scientific aggregation)
        index_dir = self.ctx.repo_root / "results" / "confirmatory" / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            index_dir / f"{self.ctx.root.name}.json",
            {
                "unit_id": self.ctx.root.name,
                "results_path": str(result_path),
                "unit_complete": str(self.ctx.root / "status" / "unit_complete.json"),
                "scientific_status": SCIENTIFIC_STATUS_CONFIRMATORY if self.ctx.adapters.scientific else self.ctx.adapters.mark,
            },
        )
