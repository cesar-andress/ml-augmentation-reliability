"""Confirmatory unit orchestrator — Protocol v1.2.1."""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass, field
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
from src.experiment.live_adapters import LiveAdapters
from src.experiment.live_engine import LiveContext, LivePhaseEngine
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
    adapters: LiveAdapters = field(default_factory=LiveAdapters)


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
                "scientific": cfg.adapters.scientific,
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
            allow_hash_mismatch_reset=bool(cfg.dry_run),
        )
        self.engine: LivePhaseEngine | None = None

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
        """Full confirmatory (or toy/mocked) live execution."""
        self.engine = LivePhaseEngine(
            LiveContext(
                repo_root=self.repo_root,
                root=self.root,
                dataset_id=self.cfg.dataset_id,
                repeat=self.cfg.repeat,
                fold=self.cfg.fold,
                adapters=self.cfg.adapters,
                validation=self.validation,
            )
        )
        phase_runners = {
            "P00_VALIDATE_FREEZE": self._phase_validate_freeze,
            "P01_LOAD_DATASET": lambda: self.engine.p01_load_dataset(),
            "P02_BUILD_SPLITS": lambda: self.engine.p02_build_splits(),
            "P03_FIT_PREPROCESSING": lambda: self.engine.p03_fit_preprocessing(),
            "P04_PREPARE_OUTER_AUGMENTATIONS": lambda: self.engine.p04_outer_augmentations(),
            "P05_PREPARE_INNER_AUGMENTATION_CACHE": lambda: self.engine.p05_inner_cache(),
            "P06_RUN_GBDT_HPO": lambda: self.engine.p06_gbdt_hpo(),
            "P07_RUN_A0PLUS": lambda: self.engine.p07_a0plus(),
            "P08_FIT_FINAL_LEARNERS": lambda: self.engine.p08_final_learners(),
            "P09_PREDICT_CAL_A": lambda: self.engine.p09_predict_cal_a(),
            "P10_FIT_CALIBRATORS_AND_THRESHOLD": lambda: self.engine.p10_calibrators(),
            "P11_PREDICT_CAL_B": lambda: self.engine.p11_predict_cal_b(),
            "P12_FIT_CONFORMAL": lambda: self.engine.p12_conformal(),
            "P13_PREDICT_TEST": lambda: self.engine.p13_predict_test(),
            "P14_COMPUTE_METRICS": lambda: self.engine.p14_metrics(),
            "P15_VALIDATE_OUTPUTS": lambda: self.engine.p15_validate_outputs(),
            "P16_FINALIZE_UNIT": lambda: self.engine.p16_finalize(),
        }
        for phase in PHASES:
            if not self.phase_mgr.should_run_phase(
                phase,
                resume=self.cfg.resume,
                force_rerun_failed=self.cfg.force_rerun_failed,
                input_hashes={"config_hash": self._config_hash},
            ):
                self.logger.info("skip phase %s (resume reuse)", phase)
                # Hydrate in-memory context from disk for downstream phases
                if self.engine is not None:
                    self.engine.hydrate_skipped_phase(phase)
                    if phase == "P00_VALIDATE_FREEZE" and self.validation is None:
                        self.validation = self.engine.ctx.validation
                    elif phase == "P00_VALIDATE_FREEZE":
                        self.engine.ctx.validation = self.validation
                continue
            self._run_phase(phase, phase_runners[phase])
            # keep validation in sync after P00
            if phase == "P00_VALIDATE_FREEZE" and self.engine is not None:
                self.engine.ctx.validation = self.validation
                self.validation = self.engine.ctx.validation or self.validation
        unit_manifest = self._build_unit_manifest(dry_run=False)
        write_json(self.root / "manifests" / "unit_manifest.json", unit_manifest)
        return {
            "mode": "live",
            "unit_id": self.uid,
            "scientific_status": SCIENTIFIC_STATUS_CONFIRMATORY
            if self.cfg.adapters.scientific
            else self.cfg.adapters.mark,
            "unit_root": str(self.root),
        }

    def _run_phase(self, phase: str, fn) -> None:
        self.phase_mgr.start_phase(phase, input_hashes={"config_hash": self._config_hash})
        t0 = datetime.now(timezone.utc)
        try:
            fn()
            out_hash = hashlib.sha256(json.dumps({"phase": phase, "ok": True}).encode()).hexdigest()
            # Prefer concrete artifact checksum when available
            artifact = self._phase_artifact_checksum(phase)
            if artifact:
                out_hash = artifact
            self.phase_mgr.complete_phase(
                phase,
                output_hashes={"checksum": out_hash},
                duration=(datetime.now(timezone.utc) - t0).total_seconds(),
            )
        except Exception as e:
            self.phase_mgr.fail_phase(phase, exception=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise

    def _phase_artifact_checksum(self, phase: str) -> str | None:
        mapping = {
            "P01_LOAD_DATASET": self.root / "manifests" / "dataset_manifest.json",
            "P02_BUILD_SPLITS": self.root / "manifests" / "split_manifest.json",
            "P03_FIT_PREPROCESSING": self.root / "manifests" / "preprocessing_manifest.json",
            "P15_VALIDATE_OUTPUTS": self.root / "manifests" / "validation_report.json",
            "P16_FINALIZE_UNIT": self.root / "status" / "unit_complete.json",
            "P14_COMPUTE_METRICS": self.root / "metrics" / "results.parquet",
        }
        path = mapping.get(phase)
        if path and path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return None

    def _phase_validate_freeze(self) -> None:
        if self.cfg.adapters.skip_freeze_validation:
            self.validation = {
                "protocol_version": "1.2.1",
                "freeze_tag": "protocol-v1.2.1-freeze",
                "freeze_commit": "toy",
                "cohort_sha256": "toy",
                "hpo_config_sha256": "toy",
                "a3_config_sha256": "toy",
                "dataset_id": self.cfg.dataset_id,
                "dataset_name": self.cfg.adapters.expected_name or "toy",
                "dataset_version": self.cfg.adapters.expected_version or 1,
                "expected_raw_checksum": self.cfg.adapters.expected_checksum,
                "repeat": self.cfg.repeat,
                "fold": self.cfg.fold,
                "scientific_mode": True,
                "smoke_settings_allowed": False,
            }
            return
        self.validation = validate_pre_run(
            repo_root=self.repo_root,
            dataset_id=self.cfg.dataset_id,
            repeat=self.cfg.repeat,
            fold=self.cfg.fold,
        )

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
            "scientific": self.cfg.adapters.scientific,
            "mark": self.cfg.adapters.mark,
        }
