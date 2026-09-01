"""Confirmatory runner tests — mocks/toy only; no model training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment.confirmatory_constants import (
    A3_CONFIG_SHA256,
    COHORT_SHA256,
    FREEZE_COMMIT,
    FREEZE_TAG,
    HPO_CONFIG_SHA256,
    PHASES,
    PROTOCOL_VERSION,
    unit_id,
)
from src.experiment.confirmatory_plan import build_dry_run_plan, expected_final_cells
from src.experiment.confirmatory_runner import ConfirmatoryRunConfig, ConfirmatoryRunner
from src.experiment.confirmatory_validation import ConfirmatoryPreRunError, validate_pre_run
from src.experiment.phase_state import PhaseConflictError, PhaseStateManager, compute_config_hash
from src.experiment.test_access_guard import CellTestAccessGuard, ConfirmatoryTestAccessError

RUNNER = ROOT / "scripts" / "run_confirmatory.py"


def test_cli_requires_dataset_repeat_fold():
    proc = subprocess.run(
        [sys.executable, str(RUNNER)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "required" in proc.stderr.lower() or "required" in proc.stdout.lower()


def test_invalid_repeat_rejected():
    with pytest.raises(ConfirmatoryPreRunError, match="repeat"):
        validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=9, fold=0)


def test_invalid_fold_rejected():
    with pytest.raises(ConfirmatoryPreRunError, match="fold"):
        validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=9)


def test_invalid_frozen_dataset_rejected():
    with pytest.raises(ConfirmatoryPreRunError, match="not in frozen cohort"):
        validate_pre_run(repo_root=ROOT, dataset_id=99999, repeat=0, fold=0)


def test_wrong_hpo_sha_rejected(tmp_path, monkeypatch):
    bad_hpo = tmp_path / "hpo.yaml"
    bad_hpo.write_text("protocol_version: '1.2.1'\n")
    monkeypatch.setattr(
        "src.experiment.confirmatory_validation.sha256_path",
        lambda p: "deadbeef" if "hpo" in str(p) else HPO_CONFIG_SHA256,
    )
    with pytest.raises(ConfirmatoryPreRunError, match="HPO config SHA"):
        validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=0)


def test_pre_run_accepts_frozen_dataset_44():
    v = validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=0)
    assert v["protocol_version"] == PROTOCOL_VERSION
    assert v["cohort_sha256"] == COHORT_SHA256
    assert v["hpo_config_sha256"] == HPO_CONFIG_SHA256
    assert v["a3_config_sha256"] == A3_CONFIG_SHA256
    assert v["freeze_tag"] == FREEZE_TAG
    assert v["freeze_commit"] == FREEZE_COMMIT


def test_dry_run_zero_model_cuda_dataset_download():
    calls = {"openml": 0, "cuda": 0, "xgb": 0}

    def block_openml(*a, **k):
        calls["openml"] += 1
        raise AssertionError("OpenML download forbidden in dry-run")

    with mock.patch("openml.datasets.get_dataset", side_effect=block_openml):
        with mock.patch("torch.cuda.is_available", side_effect=lambda: (_ for _ in ()).throw(AssertionError("CUDA forbidden"))):
            cfg = ConfirmatoryRunConfig(
                repo_root=ROOT, dataset_id=44, repeat=0, fold=0, dry_run=True
            )
            runner = ConfirmatoryRunner(cfg)
            out = runner.run_dry()
    assert out["mode"] == "dry_run"
    assert calls["openml"] == 0


def test_phase_graph_order_in_plan():
    v = validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=0)
    plan = build_dry_run_plan(repo_root=ROOT, validation=v)
    assert plan["phase_graph"] == list(PHASES)


def test_eighteen_cell_expectation():
    cells = expected_final_cells()
    assert len(cells) == 18
    a0plus = [c for c in cells if c["arm"] == "A0+"]
    assert len(a0plus) == 2
    assert {c["learner"] for c in a0plus} == {"xgboost", "catboost"}


def test_a0plus_only_gbdt():
    cells = expected_final_cells()
    for c in cells:
        if c["arm"] == "A0+":
            assert c["learner"] in {"xgboost", "catboost"}


def test_test_access_guard_blocks_early():
    g = CellTestAccessGuard(learner="xgboost", arm="A0")
    with pytest.raises(ConfirmatoryTestAccessError):
        g.assert_test_allowed()
    g.mark_cal_a_complete()
    g.mark_calibrators_complete()
    g.mark_cal_b_complete()
    g.mark_conformal_complete()
    g.unlock_test()
    g.assert_test_allowed()
    assert g.test_unlocked_at is not None


def test_inner_cache_paths_learner_independent():
    v = validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=0)
    plan = build_dry_run_plan(repo_root=ROOT, validation=v)
    keys = plan["cache_keys"]["inner_examples"]
    assert "learner" not in json.dumps(keys)


def test_outer_cache_shared_across_learners():
    v = validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=0)
    plan = build_dry_run_plan(repo_root=ROOT, validation=v)
    assert plan["paths"]["outer_generated"]["A3"]
    assert set(plan["learners"]) == {"xgboost", "catboost", "tabpfn", "tabicl"}


def test_resume_hash_validation(tmp_path):
    status = tmp_path / "status.json"
    cfg_hash = compute_config_hash({"x": 1})
    mgr = PhaseStateManager(status, config_hash=cfg_hash)
    mgr.complete_phase("P00_VALIDATE_FREEZE", output_hashes={"checksum": "abc"})
    assert mgr.can_reuse_phase(
        "P00_VALIDATE_FREEZE",
        input_hashes={"config_hash": cfg_hash},
        output_checksum="abc",
    )
    assert not mgr.can_reuse_phase(
        "P00_VALIDATE_FREEZE",
        input_hashes={"config_hash": cfg_hash},
        output_checksum="different",
    )


def test_conflicting_complete_output_aborts(tmp_path):
    status = tmp_path / "status.json"
    cfg_hash = compute_config_hash({"x": 1})
    mgr = PhaseStateManager(status, config_hash=cfg_hash)
    mgr.complete_phase("P01_LOAD_DATASET", output_hashes={"checksum": "abc"})
    with pytest.raises(PhaseConflictError):
        mgr.assert_no_complete_conflict(
            "P01_LOAD_DATASET",
            input_hashes={"config_hash": cfg_hash},
            output_checksum="other",
        )


def test_failed_only_rerun(tmp_path):
    status = tmp_path / "status.json"
    cfg_hash = compute_config_hash({"x": 1})
    mgr = PhaseStateManager(status, config_hash=cfg_hash)
    mgr.fail_phase("P02_BUILD_SPLITS", exception="boom")
    assert mgr.should_run_phase(
        "P02_BUILD_SPLITS",
        resume=True,
        force_rerun_failed=True,
        input_hashes={"config_hash": cfg_hash},
    )
    mgr.complete_phase("P01_LOAD_DATASET", output_hashes={"checksum": "x"})
    assert not mgr.should_run_phase(
        "P01_LOAD_DATASET",
        resume=True,
        force_rerun_failed=True,
        input_hashes={"config_hash": cfg_hash},
        output_checksum="x",
    )


def test_dry_run_scientific_status_planned(tmp_path):
    unit = tmp_path / "units" / unit_id(dataset_id=44, repeat=0, fold=0)
    with mock.patch.object(ConfirmatoryRunner, "_ensure_dirs", lambda self: None):
        cfg = ConfirmatoryRunConfig(
            repo_root=ROOT, dataset_id=44, repeat=0, fold=0, dry_run=True
        )
        runner = ConfirmatoryRunner(cfg)
        runner.root = unit
        runner.root.mkdir(parents=True, exist_ok=True)
        (runner.root / "logs").mkdir(exist_ok=True)
        (runner.root / "status").mkdir(exist_ok=True)
        (runner.root / "manifests").mkdir(exist_ok=True)
        runner.phase_mgr = PhaseStateManager(
            runner.root / "status" / "unit_status.json",
            config_hash=runner._config_hash,
        )
        out = runner.run_dry()
    assert out["plan"]["scientific_status"] == "CONFIRMATORY_PLANNED_NOT_EXECUTED"


def test_dry_run_plan_content():
    v = validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=0)
    plan = build_dry_run_plan(repo_root=ROOT, validation=v)
    assert plan["dataset_id"] == 44
    assert plan["repeat"] == 0
    assert plan["fold"] == 0
    assert plan["protocol_version"] == "1.2.1"
    assert plan["hpo"]["standard_candidates"] == 20
    assert plan["hpo"]["inner_cv_folds"] == 3
    assert plan["hpo_config_sha256"] == HPO_CONFIG_SHA256
    assert plan["a3_config_sha256"] == A3_CONFIG_SHA256
    assert plan["expected_final_cell_count"] == 18


def test_manifest_traceability_after_dry_run():
    cfg = ConfirmatoryRunConfig(
        repo_root=ROOT, dataset_id=44, repeat=0, fold=0, dry_run=True
    )
    runner = ConfirmatoryRunner(cfg)
    out = runner.run_dry()
    unit_manifest = json.loads(
        (runner.root / "manifests" / "unit_manifest.json").read_text()
    )
    dry_plan = json.loads((runner.root / "manifests" / "dry_run_plan.json").read_text())
    assert unit_manifest["dry_run_plan_path"].endswith("dry_run_plan.json")
    assert dry_plan["cohort_sha256"] == COHORT_SHA256
    assert unit_manifest["expected_final_cells"] is not None


def test_confirmatory_runner_rejects_v1_2_protocol():
    from src.experiment.confirmatory_protocol import ConfirmatoryProtocolError, validate_confirmatory_protocol_start

    with pytest.raises(ConfirmatoryProtocolError, match="superseded"):
        validate_confirmatory_protocol_start(repo_root=ROOT, protocol_version="1.2")


def test_dry_run_cli_integration():
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dataset-id",
            "44",
            "--repeat",
            "0",
            "--fold",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    plan_path = ROOT / "results/confirmatory/units/d44_r0_f0/manifests/dry_run_plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text())
    assert plan["expected_final_cell_count"] == 18
    assert plan["hpo"]["standard_candidates"] == 20
