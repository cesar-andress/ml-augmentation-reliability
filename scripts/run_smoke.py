#!/usr/bin/env python
"""End-to-end smoke test: 4 learners x A0/A1/A2 (+ XGBoost x A3 smoke-only)."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.arms import (
    assert_balanced_to_majority,
    augment_a0,
    augment_a1,
    augment_a2,
    augment_a3_tabddpm_via_subprocess,
)
from src.calibration.posthoc import PlattCalibrator
from src.calibration.threshold import tune_threshold_f1
from src.conformal.split_conformal import conformity_score, conformal_quantile, prediction_sets, set_metrics
from src.data.openml_loader import binarize_labels, load_openml_raw
from src.data.splitting import build_split_for_fold
from src.evaluation.metrics import compute_metrics
from src.experiment.manifest import create_manifest, save_manifest
from src.experiment.schema import RESULT_COLUMNS, append_result, rows_to_frame
from src.learners.wrappers import build_learner
from src.logging_utils import setup_logger, write_json
from src.logging_utils.timing import timed_gpu
from src.preprocessing.pipeline import fit_preprocessor


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cell(
    *,
    cfg: dict,
    bundle,
    y_bin: np.ndarray,
    split,
    X_all: pd.DataFrame,
    learner_name: str,
    arm: str,
    logger,
) -> dict:
    fold_seed = split.seed
    X_train_raw = X_all.iloc[split.train]
    X_cala_raw = X_all.iloc[split.cal_a]
    X_calb_raw = X_all.iloc[split.cal_b]
    X_test_raw = X_all.iloc[split.test]
    y_train = y_bin[split.train]
    y_cala = y_bin[split.cal_a]
    y_calb = y_bin[split.cal_b]
    y_test = y_bin[split.test]

    pre = fit_preprocessor(
        X_train_raw,
        unknown_category_sentinel=int(cfg["preprocessing"]["unknown_category_sentinel"]),
    )
    Xtr = pre.transform(X_train_raw)
    Xca = pre.transform(X_cala_raw)
    Xcb = pre.transform(X_calb_raw)
    Xte = pre.transform(X_test_raw)

    gen_fit = 0.0
    gen_sample = 0.0
    scientific_mark = ""

    if arm == "A0":
        aug = augment_a0(Xtr, y_train)
    elif arm == "A1":
        aug = augment_a1(Xtr, y_train, random_state=fold_seed)
        assert_balanced_to_majority(aug.y)
    elif arm == "A2":
        aug = augment_a2(
            Xtr,
            y_train,
            pre.meta,
            random_state=fold_seed,
            k_neighbors=int(cfg["augmentation"]["smote_k_neighbors"]),
        )
        assert_balanced_to_majority(aug.y)
    elif arm == "A3":
        job_dir = ROOT / cfg["paths"]["a3_jobs"] / f"{learner_name}_fold{split.fold}"
        aug = augment_a3_tabddpm_via_subprocess(
            Xtr,
            y_train,
            pre.meta,
            job_dir=job_dir,
            synthcity_python=str(ROOT / cfg["envs"]["synthcity_python"]),
            random_state=fold_seed,
            smoke_config=cfg["augmentation"]["a3_smoke"],
        )
        assert_balanced_to_majority(aug.y)
        gen_fit = aug.generator_fit_seconds
        gen_sample = aug.generator_sample_seconds
        scientific_mark = aug.scientific_mark
    else:
        raise ValueError(arm)

    if arm in {"A1", "A2"} and abs(aug.repair_fraction) > 1e-12 and arm == "A1":
        # ROS must be zero repair
        raise AssertionError(f"A1 repair_fraction expected 0, got {aug.repair_fraction}")

    learner = build_learner(learner_name, cfg["learners"], cat_features=pre.meta.categorical_cols)
    if learner_name in {"xgboost", "catboost"}:
        scientific_mark = scientific_mark or cfg["learners"][learner_name].get("mark", "SMOKE_ONLY_FIXED_HPARAMS")

    with timed_gpu() as fit_t:
        learner.fit(aug.X, aug.y)
    with timed_gpu(reset=False) as inf_t:
        p_cala = learner.predict_proba(Xca)
        p_calb = learner.predict_proba(Xcb)
        p_test = learner.predict_proba(Xte)

    platt = PlattCalibrator().fit(p_cala, y_cala)
    p_test_platt = platt.transform(p_test)
    thr = tune_threshold_f1(y_cala, p_cala)

    scores = conformity_score(p_calb, y_calb)
    qhat = conformal_quantile(scores, alpha=float(cfg["experiment"]["alpha_conformal"]))
    sets = prediction_sets(p_test, qhat)
    cmetrics = set_metrics(sets, y_test)

    metrics = compute_metrics(
        y_test,
        p_test,
        p_test_platt,
        threshold=thr,
        conformal_set_size=cmetrics["conformal_set_size"],
        conformal_coverage=cmetrics["conformal_coverage"],
    )

    ckpt_paths = {}
    if learner_name == "tabpfn":
        ckpt_paths["tabpfn"] = str(ROOT / cfg["learners"]["tabpfn"]["model_path"])
    if learner_name == "tabicl":
        ckpt_paths["tabicl"] = str(ROOT / cfg["learners"]["tabicl"]["model_path"])

    n_real = len(y_train)
    n_final = len(aug.y)
    prev_real = float(y_train.mean())
    prev_final = float(aug.y.mean())

    manifest = create_manifest(
        repo_root=ROOT,
        dataset_id=bundle.openml_id,
        dataset_name=bundle.name,
        dataset_version=bundle.version,
        dataset_checksum=bundle.checksum,
        fold=split.fold,
        repeat=split.repeat,
        seed=fold_seed,
        learner=learner_name,
        arm=arm,
        preprocessing_config={
            "unknown_category_sentinel": pre.meta.unknown_category_sentinel,
            "n_numeric": len(pre.meta.numeric_cols),
            "n_categorical": len(pre.meta.categorical_cols),
            "n_missing_indicators": len(pre.meta.missing_indicator_cols),
        },
        augmentation_config={"arm": arm, "method": aug.method, "repair_fraction": aug.repair_fraction},
        hyperparameters=cfg["learners"].get(learner_name, {}),
        checkpoint_paths=ckpt_paths,
        timings={
            "learner_fit_seconds": fit_t.seconds,
            "inference_seconds": inf_t.seconds,
            "generator_fit_seconds": gen_fit,
            "generator_sample_seconds": gen_sample,
        },
        peak_gpu_memory_mb=fit_t.peak_gpu_memory_mb,
        extra={"threshold": thr, "qhat": qhat, "scientific_mark": scientific_mark},
    )
    mpath = ROOT / cfg["paths"]["smoke"] / "manifests" / f"{learner_name}_{arm}_r{split.repeat}_f{split.fold}.json"
    save_manifest(manifest, mpath)

    # persist predictions
    pred_dir = ROOT / "results" / "predictions" / "smoke"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y_test": y_test,
            "p_raw": p_test,
            "p_platt": p_test_platt,
            "set_size": [len(s) for s in sets],
        }
    ).to_parquet(pred_dir / f"{learner_name}_{arm}.parquet")

    row = {c: None for c in RESULT_COLUMNS}
    row.update(
        {
            "dataset_id": bundle.openml_id,
            "dataset_name": bundle.name,
            "repeat": split.repeat,
            "fold": split.fold,
            "seed": fold_seed,
            "learner": learner_name,
            "family": learner.family,
            "arm": arm,
            "temperature": float(cfg["experiment"]["temperature"]),
            "calibration": "platt",
            "n_train_real": n_real,
            "n_train_final": n_final,
            "minority_prevalence_real": prev_real,
            "minority_prevalence_final": prev_final,
            "log_loss": metrics["log_loss"],
            "auroc": metrics["auroc"],
            "calibration_slope": metrics["calibration_slope"],
            "calibration_intercept": metrics["calibration_intercept"],
            "conformal_set_size": metrics["conformal_set_size"],
            "conformal_coverage": metrics["conformal_coverage"],
            "f1_tuned": metrics["f1_tuned"],
            "brier": metrics["brier"],
            "auprc": metrics["auprc"],
            "repair_fraction": aug.repair_fraction,
            "generator_fit_seconds": gen_fit,
            "generator_sample_seconds": gen_sample,
            "learner_fit_seconds": fit_t.seconds,
            "inference_seconds": inf_t.seconds,
            "peak_gpu_memory_mb": fit_t.peak_gpu_memory_mb,
            "status": "ok",
            "failure_reason": "",
            "manifest_path": str(mpath),
            "scientific_mark": scientific_mark,
        }
    )
    # also store raw log loss in metrics sidecar
    write_json(
        ROOT / "results" / "metrics" / "smoke" / f"{learner_name}_{arm}.json",
        {**metrics, "log_loss_raw": metrics["log_loss_raw"], "threshold": thr, "qhat": qhat},
    )
    logger.info("%s %s OK log_loss=%.4f set_size=%.3f", learner_name, arm, metrics["log_loss"], metrics["conformal_set_size"])
    return row


def failure_row(*, bundle, split, learner_name, arm, err, scientific_mark="") -> dict:
    row = {c: None for c in RESULT_COLUMNS}
    row.update(
        {
            "dataset_id": getattr(bundle, "openml_id", None),
            "dataset_name": getattr(bundle, "name", None),
            "repeat": split.repeat if split else None,
            "fold": split.fold if split else None,
            "seed": split.seed if split else None,
            "learner": learner_name,
            "family": None,
            "arm": arm,
            "temperature": 1.0,
            "calibration": "platt",
            "status": "fail",
            "failure_reason": err,
            "manifest_path": "",
            "scientific_mark": scientific_mark,
            "n_train_real": None,
            "n_train_final": None,
            "minority_prevalence_real": None,
            "minority_prevalence_final": None,
            "log_loss": None,
            "auroc": None,
            "calibration_slope": None,
            "calibration_intercept": None,
            "conformal_set_size": None,
            "conformal_coverage": None,
            "f1_tuned": None,
            "brier": None,
            "auprc": None,
            "repair_fraction": None,
            "generator_fit_seconds": None,
            "generator_sample_seconds": None,
            "learner_fit_seconds": None,
            "inference_seconds": None,
            "peak_gpu_memory_mb": None,
        }
    )
    return row


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    logger = setup_logger("smoke", ROOT / cfg["paths"]["logs"] / "smoke.log")

    # checkpoint identities
    ckpt_info = {}
    for key, rel in [
        ("tabpfn", cfg["learners"]["tabpfn"]["model_path"]),
        ("tabicl", cfg["learners"]["tabicl"]["model_path"]),
    ]:
        p = ROOT / rel
        ckpt_info[key] = {
            "path": str(p),
            "exists": p.exists(),
            "sha256": _sha256_file(p) if p.exists() else None,
            "identity": (
                "Prior-Labs/TabPFN-v2-clf/tabpfn-v2-classifier.ckpt"
                if key == "tabpfn"
                else "jingang/TabICL/tabicl-classifier-v1-20250208.ckpt"
            ),
        }
    write_json(ROOT / "artifacts" / "environment" / "checkpoint_identities.json", ckpt_info)

    bundle = load_openml_raw(int(cfg["dataset"]["openml_id"]))
    y_bin, y_meta = binarize_labels(bundle.y)
    write_json(ROOT / "artifacts" / "smoke" / "label_meta.json", y_meta)

    split = build_split_for_fold(
        y_bin,
        n_splits=int(cfg["experiment"]["n_splits"]),
        n_repeats=int(cfg["experiment"]["n_repeats"]),
        seed=int(cfg["experiment"]["seed"]),
        repeat_index=int(cfg["experiment"]["repeat_index"]),
        fold_index=int(cfg["experiment"]["outer_fold_index"]),
        train_frac=float(cfg["split"]["train_frac_of_remainder"]),
        cal_a_frac=float(cfg["split"]["cal_a_frac_of_remainder"]),
        cal_b_frac=float(cfg["split"]["cal_b_frac_of_remainder"]),
    )
    write_json(
        ROOT / "artifacts" / "smoke" / "split_indices.json",
        {
            "train": split.train.tolist(),
            "cal_a": split.cal_a.tolist(),
            "cal_b": split.cal_b.tolist(),
            "test": split.test.tolist(),
            "repeat": split.repeat,
            "fold": split.fold,
            "seed": split.seed,
            "n": {
                "train": len(split.train),
                "cal_a": len(split.cal_a),
                "cal_b": len(split.cal_b),
                "test": len(split.test),
                "total": len(y_bin),
            },
        },
    )

    rows: list[dict] = []
    cells = [(ln, arm) for ln in cfg["learners"]["smoke_cells"] for arm in ["A0", "A1", "A2"]]
    for ln in cfg["learners"]["a3_required"]:
        cells.append((ln, "A3"))

    for learner_name, arm in cells:
        try:
            row = run_cell(
                cfg=cfg,
                bundle=bundle,
                y_bin=y_bin,
                split=split,
                X_all=bundle.X,
                learner_name=learner_name,
                arm=arm,
                logger=logger,
            )
            append_result(rows, row)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.error("FAIL %s %s: %s\n%s", learner_name, arm, err, traceback.format_exc())
            mark = "SMOKE_ONLY_NOT_SCIENTIFIC" if arm == "A3" else ""
            if learner_name in {"xgboost", "catboost"} and arm != "A3":
                mark = "SMOKE_ONLY_FIXED_HPARAMS"
            row = failure_row(bundle=bundle, split=split, learner_name=learner_name, arm=arm, err=err, scientific_mark=mark)
            append_result(rows, row)

    df = rows_to_frame(rows)
    df["scientific_status"] = "SMOKE_ONLY"
    out = ROOT / "results" / "smoke" / "smoke_results.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    summary = {
        "n_rows": len(df),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_fail": int((df["status"] == "fail").sum()),
        "failures": df.loc[df["status"] == "fail", ["learner", "arm", "failure_reason"]].to_dict(orient="records"),
        "output": str(out),
    }
    write_json(ROOT / "artifacts" / "smoke" / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
