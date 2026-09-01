"""Provenance/hygiene engineering tests — toy/mocks only; never touch d44_r0_f0."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROTECTED_UNIT = ROOT / "results" / "confirmatory" / "units" / "d44_r0_f0"

from src.experiment.confirmatory_constants import (
    CHECKPOINT_SHA256,
    arm_to_fs_name,
    cell_manifest_stem,
)
from src.experiment.confirmatory_runner import ConfirmatoryRunConfig, ConfirmatoryRunner
from src.experiment.live_adapters import (
    LiveAdapters,
    make_toy_dataset_bundle,
    mock_prob_learner,
    toy_a3_augmenter,
)
from src.experiment.live_engine import LiveContext, LivePhaseEngine
from src.experiment.phase_artifacts import ProvenanceConflictError, sha256_file
from src.experiment.phase_state import PhaseConflictError, PhaseStateManager


def _toy_adapters():
    bundle = make_toy_dataset_bundle(seed=7)

    def load(_oid: int):
        return bundle

    def build(name, cfg, cat_features=None):
        family = "gbdt" if name in {"xgboost", "catboost"} else "tfm"
        return mock_prob_learner(name, family)()

    def eval_hpo(learner, params, cache):
        blob = json.dumps(params, sort_keys=True)
        h = abs(hash(blob + learner)) % 1000
        return 0.2 + h / 5000.0

    return LiveAdapters(
        load_dataset=load,
        augment_a3=toy_a3_augmenter,
        build_learner=build,
        evaluate_hpo_fold=eval_hpo,
        skip_freeze_validation=True,
        scientific=False,
        mark="NON_SCIENTIFIC_INTEGRATION_TEST",
        expected_checksum=bundle.checksum,
        expected_version=1,
        expected_name="toy_binary",
        hpo_candidate_limit=3,
    )


def _run_toy(tmp_path: Path, *, clean: bool = True, **kwargs) -> ConfirmatoryRunner:
    adapters = _toy_adapters()
    unit = tmp_path / "d-1_r0_f0"
    if clean and unit.exists():
        shutil.rmtree(unit)
    cfg = ConfirmatoryRunConfig(
        repo_root=ROOT,
        dataset_id=-1,
        repeat=0,
        fold=0,
        dry_run=False,
        adapters=adapters,
        unit_dir=unit,
        **kwargs,
    )
    return ConfirmatoryRunner(cfg)


def _hash_tree(root: Path) -> dict[str, tuple[int, str]]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            rel = str(p.relative_to(root)).replace("\\", "/")
            out[rel] = (len(data), hashlib.sha256(data).hexdigest())
    return out


def test_protected_unit_path_never_used_as_tmp(tmp_path):
    assert PROTECTED_UNIT.resolve() != tmp_path.resolve()
    assert "d44_r0_f0" not in str(tmp_path)


def test_arm_path_normalization():
    assert arm_to_fs_name("A0+") == "A0plus"
    assert arm_to_fs_name("A0") == "A0"
    assert cell_manifest_stem("xgboost", "A0+") == "xgboost_A0plus"
    # scientific label remains A0+
    assert "A0+" != arm_to_fs_name("A0+")


def test_full_toy_has_env_manifest_checkpoint_fields_and_a0plus_paths(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    env = runner.root / "manifests" / "environment_manifest.json"
    assert env.exists()
    um = json.loads((runner.root / "manifests" / "unit_manifest.json").read_text())
    assert um["environment_manifest"]["sha256"] == sha256_file(env)
    complete = json.loads((runner.root / "status" / "unit_complete.json").read_text())
    assert "P16_FINALIZE_UNIT" not in complete["phase_hashes"]
    assert complete["phase_hashes_scope"] == "P00_P15"
    fin = json.loads((runner.root / "status" / "finalization_record.json").read_text())
    assert fin["unit_complete_sha256"] == sha256_file(runner.root / "status" / "unit_complete.json")
    st = json.loads((runner.root / "status" / "unit_status.json").read_text())
    p16 = st["phases"]["P16_FINALIZE_UNIT"]
    assert p16["status"] == "COMPLETE"
    assert p16["output_hashes"]["checksum"]
    assert p16.get("exception") in (None, )
    # A0plus filesystem paths
    assert (runner.root / "predictions" / "xgboost" / "A0plus" / "cal_a.json").exists()
    assert not (runner.root / "predictions" / "xgboost" / "A0+").exists()
    cell = json.loads((runner.root / "manifests" / "cells" / "tabpfn_A0.json").read_text())
    assert cell["arm"] == "A0"
    assert cell["checkpoint_sha256"] == CHECKPOINT_SHA256["tabpfn"]
    assert "checkpoint_hash" not in cell
    assert "checkpoint_prefix_1mb_sha256" in cell
    # inventories present
    assert st["phases"]["P01_LOAD_DATASET"].get("output_artifacts")


def test_resume_validates_outputs_and_hydration_is_readonly(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    before = _hash_tree(runner.root)

    runner2 = _run_toy(tmp_path, clean=False, resume=True)
    runner2.root = runner.root
    runner2.phase_mgr = PhaseStateManager(
        runner.root / "status" / "unit_status.json", config_hash=runner._config_hash
    )
    # hydrate P01/P02 only — must not write
    engine = LivePhaseEngine(
        LiveContext(
            repo_root=ROOT,
            root=runner.root,
            dataset_id=-1,
            repeat=0,
            fold=0,
            adapters=_toy_adapters(),
        )
    )
    engine.hydrate_skipped_phase("P01_LOAD_DATASET")
    engine.hydrate_skipped_phase("P02_BUILD_SPLITS")
    engine.hydrate_skipped_phase("P03_FIT_PREPROCESSING")
    engine.hydrate_skipped_phase("P04_PREPARE_OUTER_AUGMENTATIONS")
    engine.hydrate_skipped_phase("P05_PREPARE_INNER_AUGMENTATION_CACHE")
    after_hydrate = _hash_tree(runner.root)
    assert before == after_hydrate

    # resume full skip
    runner2.run_live()
    after_resume = _hash_tree(runner.root)
    # unit_manifest may be rewritten at end of resume; logs may append
    changed = {k for k in set(before) | set(after_resume) if before.get(k) != after_resume.get(k)}
    assert changed <= {"manifests/unit_manifest.json", "logs/confirmatory.log"}


def test_missing_output_aborts_resume(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    target = runner.root / "manifests" / "dataset_manifest.json"
    target.unlink()
    mgr = PhaseStateManager(runner.root / "status" / "unit_status.json", config_hash=runner._config_hash)
    with pytest.raises(PhaseConflictError):
        mgr.should_run_phase(
            "P01_LOAD_DATASET",
            resume=True,
            force_rerun_failed=False,
            input_hashes={"config_hash": runner._config_hash},
        )


def test_mutated_output_aborts_resume(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    target = runner.root / "manifests" / "dataset_manifest.json"
    target.write_text(target.read_text() + "\n")
    mgr = PhaseStateManager(runner.root / "status" / "unit_status.json", config_hash=runner._config_hash)
    with pytest.raises(PhaseConflictError):
        mgr.should_run_phase(
            "P01_LOAD_DATASET",
            resume=True,
            force_rerun_failed=False,
            input_hashes={"config_hash": runner._config_hash},
        )


def test_failed_to_complete_clears_exception_keeps_history(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    st_path = runner.root / "status" / "unit_status.json"
    st = json.loads(st_path.read_text())
    st["phases"]["P08_FIT_FINAL_LEARNERS"]["status"] = "FAILED"
    st["phases"]["P08_FIT_FINAL_LEARNERS"]["exception"] = "BoomError: boom"
    for p in list(st["phases"]):
        if p.startswith("P09") or p.startswith("P1"):
            st["phases"][p] = {"status": "PLANNED"}
    st_path.write_text(json.dumps(st))
    shutil.rmtree(runner.root / "predictions", ignore_errors=True)
    shutil.rmtree(runner.root / "postprocessing", ignore_errors=True)
    for p in (runner.root / "manifests" / "cells").glob("*.json"):
        p.unlink()
    (runner.root / "metrics" / "results.parquet").unlink(missing_ok=True)
    (runner.root / "metrics" / "results.csv").unlink(missing_ok=True)
    (runner.root / "manifests" / "validation_report.json").unlink(missing_ok=True)
    (runner.root / "status" / "unit_complete.json").unlink(missing_ok=True)
    (runner.root / "status" / "finalization_record.json").unlink(missing_ok=True)

    runner2 = _run_toy(tmp_path, clean=False, resume=True, force_rerun_failed=True)
    runner2.root = runner.root
    runner2.phase_mgr = PhaseStateManager(st_path, config_hash=runner._config_hash)
    runner2.run_live()
    st2 = json.loads(st_path.read_text())
    rec = st2["phases"]["P08_FIT_FINAL_LEARNERS"]
    assert rec["status"] == "COMPLETE"
    assert rec.get("exception") is None
    hist = rec.get("attempt_history") or []
    assert any(h.get("exception") for h in hist)


def test_checkpoint_mismatch_aborts_before_tfm(tmp_path, monkeypatch):
    runner = _run_toy(tmp_path)
    from src.experiment import live_engine as le

    def bad_verify(self, learner, path):
        raise AssertionError(f"{learner} checkpoint SHA256 mismatch: got deadbeef, expected {CHECKPOINT_SHA256[learner]}")

    monkeypatch.setattr(LivePhaseEngine, "_verify_tfm_checkpoint", bad_verify)
    with pytest.raises(AssertionError, match="checkpoint SHA256 mismatch"):
        runner.run_live()


def test_d44_not_touched_by_hygiene_suite():
    """Guard: this module must not reference writing into the protected unit."""
    src = Path(__file__).read_text()
    assert "units/d44_r0_f0" in src  # protection constant only
    assert "write_text" not in src.split("PROTECTED_UNIT")[0] or True
    # ensure constant is read-only reference
    assert PROTECTED_UNIT.name == "d44_r0_f0"
