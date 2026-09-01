"""Confirmatory unit orchestrator — Protocol v1.2.1."""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.confirmatory_constants import (
    PHASES,
    SCIENTIFIC_STATUS_CONFIRMATORY,
    unit_id,
    unit_root,
)
from src.experiment.confirmatory_plan import build_dry_run_plan, expected_final_cells
from src.experiment.confirmatory_validation import validate_pre_run
from src.experiment.phase_state import PhaseStateManager, compute_config_hash
from src.logging_utils import setup_logger, write_json


@dataclass
class ConfirmatoryRunConfig:
    repo_root: Path
    dataset_id: int
    repeat: int
    fold: int
    dry_run: bool = False
    resume: bool = False
    force_rerun_failed: bool = False
    log_level: str = "INFO"


class ConfirmatoryRunner:
    def __init__(self, cfg: ConfirmatoryRunConfig):
        self.cfg = cfg
        self.repo_root = Path(cfg.repo_root).resolve()
        self.uid = unit_id(dataset_id=cfg.dataset_id, repeat=cfg.repeat, fold=cfg.fold)
        self.root = unit_root(self.repo_root, self.uid)
        self.validation: dict[str, Any] | None = None
        self.plan: dict[str, Any] | None = None
        self._config_hash = compute_config_hash(
            {
                "dataset_id": cfg.dataset_id,
                "repeat": cfg.repeat,
                "fold": cfg.fold,
                "dry_run": cfg.dry_run,
            }
        )
        self._ensure_dirs()
        self.logger = setup_logger(
            "confirmatory",
            self.root / "logs" / "confirmatory.log",
        )
        self.phase_mgr = PhaseStateManager(
            self.root / "status" / "unit_status.json",
            config_hash=self._config_hash,
        )

    def _ensure_dirs(self) -> None:
        for sub in (
            "manifests/cells",
            "generated/A1",
            "generated/A2",
            "generated/A3",
            "hpo/inner_cache/A1",
            "hpo/inner_cache/A2",
            "hpo/inner_cache/A3",
            "predictions",
            "metrics",
            "postprocessing",
            "logs",
            "status",
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        if self.cfg.dry_run:
            return self.run_dry()
        return self.run_live()

    def run_dry(self) -> dict[str, Any]:
        self.logger.info("dry-run start unit=%s", self.uid)
        self._run_phase("P00_VALIDATE_FREEZE", self._phase_validate_freeze)
        self.plan = build_dry_run_plan(repo_root=self.repo_root, validation=self.validation or {})
        plan_path = self.root / "manifests" / "dry_run_plan.json"
        write_json(plan_path, self.plan)
        self._mark_remaining_planned()
        unit_manifest = self._build_unit_manifest(dry_run=True)
        write_json(self.root / "manifests" / "unit_manifest.json", unit_manifest)
        self.logger.info("dry-run complete plan=%s", plan_path)
        return {"mode": "dry_run", "unit_id": self.uid, "plan_path": str(plan_path), "plan": self.plan}

    def run_live(self) -> dict[str, Any]:
        """Full confirmatory execution (not invoked in engineering-only task)."""
        phase_runners = {
            "P00_VALIDATE_FREEZE": self._phase_validate_freeze,
            "P01_LOAD_DATASET": self._phase_load_dataset,
            "P02_BUILD_SPLITS": self._phase_build_splits,
            "P03_FIT_PREPROCESSING": self._phase_fit_preprocessing,
            "P04_PREPARE_OUTER_AUGMENTATIONS": self._phase_outer_augmentations,
            "P05_PREPARE_INNER_AUGMENTATION_CACHE": self._phase_inner_cache,
            "P06_RUN_GBDT_HPO": self._phase_gbdt_hpo,
            "P07_RUN_A0PLUS": self._phase_a0plus,
            "P08_FIT_FINAL_LEARNERS": self._phase_final_learners,
            "P09_PREDICT_CAL_A": self._phase_predict_cal_a,
            "P10_FIT_CALIBRATORS_AND_THRESHOLD": self._phase_calibrators,
            "P11_PREDICT_CAL_B": self._phase_predict_cal_b,
            "P12_FIT_CONFORMAL": self._phase_conformal,
            "P13_PREDICT_TEST": self._phase_predict_test,
            "P14_COMPUTE_METRICS": self._phase_metrics,
            "P15_VALIDATE_OUTPUTS": self._phase_validate_outputs,
            "P16_FINALIZE_UNIT": self._phase_finalize,
        }
        for phase in PHASES:
            if not self.phase_mgr.should_run_phase(
                phase,
                resume=self.cfg.resume,
                force_rerun_failed=self.cfg.force_rerun_failed,
                input_hashes={"config_hash": self._config_hash},
            ):
                self.logger.info("skip phase %s (resume reuse)", phase)
                continue
            self._run_phase(phase, phase_runners[phase])
        unit_manifest = self._build_unit_manifest(dry_run=False)
        write_json(self.root / "manifests" / "unit_manifest.json", unit_manifest)
        return {"mode": "live", "unit_id": self.uid, "scientific_status": SCIENTIFIC_STATUS_CONFIRMATORY}

    def _run_phase(self, phase: str, fn) -> None:
        self.phase_mgr.start_phase(phase, input_hashes={"config_hash": self._config_hash})
        t0 = datetime.now(timezone.utc)
        try:
            fn()
            out_hash = hashlib.sha256(json.dumps({"phase": phase, "ok": True}).encode()).hexdigest()
            self.phase_mgr.complete_phase(
                phase,
                output_hashes={"checksum": out_hash},
                duration=(datetime.now(timezone.utc) - t0).total_seconds(),
            )
        except Exception as e:
            self.phase_mgr.fail_phase(phase, exception=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    def _phase_validate_freeze(self) -> None:
        self.validation = validate_pre_run(
            repo_root=self.repo_root,
            dataset_id=self.cfg.dataset_id,
            repeat=self.cfg.repeat,
            fold=self.cfg.fold,
        )

    def _phase_load_dataset(self) -> None:
        raise NotImplementedError("P01 live dataset load not executed in engineering task")

    def _phase_build_splits(self) -> None:
        raise NotImplementedError("P02 live splits not executed in engineering task")

    def _phase_fit_preprocessing(self) -> None:
        raise NotImplementedError("P03 live preprocessing not executed in engineering task")

    def _phase_outer_augmentations(self) -> None:
        raise NotImplementedError("P04 live outer augmentations not executed in engineering task")

    def _phase_inner_cache(self) -> None:
        raise NotImplementedError("P05 live inner cache not executed in engineering task")

    def _phase_gbdt_hpo(self) -> None:
        raise NotImplementedError("P06 live HPO not executed in engineering task")

    def _phase_a0plus(self) -> None:
        raise NotImplementedError("P07 live A0+ not executed in engineering task")

    def _phase_final_learners(self) -> None:
        raise NotImplementedError("P08 live learner fit not executed in engineering task")

    def _phase_predict_cal_a(self) -> None:
        raise NotImplementedError("P09 live CAL-A not executed in engineering task")

    def _phase_calibrators(self) -> None:
        raise NotImplementedError("P10 live calibrators not executed in engineering task")

    def _phase_predict_cal_b(self) -> None:
        raise NotImplementedError("P11 live CAL-B not executed in engineering task")

    def _phase_conformal(self) -> None:
        raise NotImplementedError("P12 live conformal not executed in engineering task")

    def _phase_predict_test(self) -> None:
        raise NotImplementedError("P13 live TEST not executed in engineering task")

    def _phase_metrics(self) -> None:
        raise NotImplementedError("P14 live metrics not executed in engineering task")

    def _phase_validate_outputs(self) -> None:
        raise NotImplementedError("P15 live validation not executed in engineering task")

    def _phase_finalize(self) -> None:
        raise NotImplementedError("P16 live finalize not executed in engineering task")

    def _mark_remaining_planned(self) -> None:
        for phase in PHASES[1:]:
            rec = self.phase_mgr.get_phase(phase)
            if rec.get("status") == "PLANNED":
                rec["status"] = "BLOCKED"
                rec["note"] = "dry_run — not executed"
        self.phase_mgr.save()

    def _build_unit_manifest(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "unit_id": self.uid,
            "protocol_version": self.validation.get("protocol_version") if self.validation else None,
            "freeze_tag": self.validation.get("freeze_tag") if self.validation else None,
            "cohort_sha256": self.validation.get("cohort_sha256") if self.validation else None,
            "hpo_config_sha256": self.validation.get("hpo_config_sha256") if self.validation else None,
            "a3_config_sha256": self.validation.get("a3_config_sha256") if self.validation else None,
            "dataset_id": self.cfg.dataset_id,
            "repeat": self.cfg.repeat,
            "fold": self.cfg.fold,
            "dry_run": dry_run,
            "expected_final_cells": expected_final_cells(),
            "phase_status_path": str(self.root / "status" / "unit_status.json"),
            "dry_run_plan_path": str(self.root / "manifests" / "dry_run_plan.json"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
