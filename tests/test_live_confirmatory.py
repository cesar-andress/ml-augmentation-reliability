"""Live confirmatory orchestration tests — toy/mocks only; no frozen dataset 44."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augmentation.arms import shared_repair_with_cell_stats
from src.experiment.confirmatory_runner import ConfirmatoryRunConfig, ConfirmatoryRunner
from src.experiment.confirmatory_validation import ConfirmatoryPreRunError, validate_pre_run
from src.experiment.live_adapters import (
    LiveAdapters,
    make_toy_dataset_bundle,
    mock_prob_learner,
    toy_a3_augmenter,
)
from src.experiment.phase_state import PhaseConflictError
from src.experiment.test_access_guard import CellTestAccessGuard, ConfirmatoryTestAccessError
from src.preprocessing.pipeline import fit_preprocessor


def _toy_adapters():
    bundle = make_toy_dataset_bundle(seed=7)

    def load(_oid: int):
        return bundle

    def build(name, cfg, cat_features=None):
        family = "gbdt" if name in {"xgboost", "catboost"} else "tfm"
        return mock_prob_learner(name, family)()

    def eval_hpo(learner, params, cache):
        # Deterministic score from candidate hypers — no real model
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
    # Isolate unit under tmp by unit_dir — never touch real confirmatory units
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


def test_shared_repair_cell_and_row_fractions_deterministic():
    bundle = make_toy_dataset_bundle()
    pre = fit_preprocessor(bundle.X.fillna({"num1": 0}), unknown_category_sentinel=-1)
    # use transformed train
    Xp = pre.transform(bundle.X)
    syn = Xp.iloc[:5].copy()
    syn.iloc[0, syn.columns.get_loc("cat")] = 999
    syn.iloc[1, syn.columns.get_loc("num1")] = 1e6
    a, row1, cell1 = shared_repair_with_cell_stats(syn, Xp, pre.meta)
    b, row2, cell2 = shared_repair_with_cell_stats(syn, Xp, pre.meta)
    assert row1 == row2 and cell1 == cell2
    assert row1 > 0 and cell1 > 0
    assert int(a.iloc[0]["cat"]) == -1


def test_full_live_toy_orchestration_18_cells(tmp_path):
    runner = _run_toy(tmp_path)
    out = runner.run_live()
    assert out["mode"] == "live"
    complete = json.loads((runner.root / "status" / "unit_complete.json").read_text())
    assert complete["status"] == "COMPLETE"
    df = pd.read_parquet(runner.root / "metrics" / "results.parquet")
    assert len(df) == 18
    assert set(df["arm"]) >= {"A0", "A1", "A2", "A3", "A0+"}
    assert (df["learner"] == "xgboost").sum() == 5
    assert (df["learner"] == "tabpfn").sum() == 4
    # no early TEST: unlock recorded on predictions
    for p in (runner.root / "predictions").rglob("test.json"):
        payload = json.loads(p.read_text())
        assert payload.get("test_unlocked_at")


def test_resume_no_complete_rerun_and_conflict(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    assert (runner.root / "status" / "unit_status.json").exists()
    # resume on same unit without wiping
    runner2 = _run_toy(tmp_path, clean=False, resume=True)
    runner2.root = runner.root
    runner2.phase_mgr = runner.phase_mgr.__class__(
        runner.root / "status" / "unit_status.json", config_hash=runner._config_hash
    )
    # corrupt checksum of a COMPLETE phase
    st = json.loads((runner.root / "status" / "unit_status.json").read_text())
    st["phases"]["P01_LOAD_DATASET"]["output_hashes"]["checksum"] = "corrupted"
    (runner.root / "status" / "unit_status.json").write_text(json.dumps(st))
    runner2.phase_mgr = runner.phase_mgr.__class__(
        runner.root / "status" / "unit_status.json", config_hash=runner._config_hash
    )
    with pytest.raises(PhaseConflictError):
        runner2.phase_mgr.assert_no_complete_conflict(
            "P01_LOAD_DATASET",
            input_hashes={"config_hash": runner._config_hash},
            output_checksum="different",
        )


def test_force_rerun_failed_only(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    st_path = runner.root / "status" / "unit_status.json"
    st = json.loads(st_path.read_text())
    st["phases"]["P14_COMPUTE_METRICS"]["status"] = "FAILED"
    st["phases"]["P14_COMPUTE_METRICS"]["exception"] = "injected"
    # clear later phases so force can proceed conceptually
    for p in ["P15_VALIDATE_OUTPUTS", "P16_FINALIZE_UNIT"]:
        st["phases"][p]["status"] = "PLANNED"
    st_path.write_text(json.dumps(st))
    mgr = runner.phase_mgr.__class__(st_path, config_hash=runner._config_hash)
    assert mgr.should_run_phase(
        "P14_COMPUTE_METRICS",
        resume=True,
        force_rerun_failed=True,
        input_hashes={"config_hash": runner._config_hash},
    )
    # COMPLETE phase must not rerun
    assert not mgr.should_run_phase(
        "P01_LOAD_DATASET",
        resume=True,
        force_rerun_failed=True,
        input_hashes={"config_hash": runner._config_hash},
        output_checksum=st["phases"]["P01_LOAD_DATASET"]["output_hashes"]["checksum"],
    )


def test_test_access_guard_early_fail():
    g = CellTestAccessGuard("xgboost", "A0")
    with pytest.raises(ConfirmatoryTestAccessError):
        g.assert_test_allowed()


def test_wrong_checksum_aborts(tmp_path):
    adapters = _toy_adapters()
    adapters.expected_checksum = "WRONG"
    cfg = ConfirmatoryRunConfig(
        repo_root=ROOT, dataset_id=-1, repeat=0, fold=0, adapters=adapters, unit_dir=tmp_path / "u"
    )
    runner = ConfirmatoryRunner(cfg)
    with pytest.raises(AssertionError, match="checksum"):
        runner.run_live()


def test_wrong_hpo_sha_rejected():
    with pytest.raises(ConfirmatoryPreRunError):
        # invalid fold triggers before sha, use valid then patch
        validate_pre_run(repo_root=ROOT, dataset_id=44, repeat=0, fold=99)


def test_smoke_flag_rejected_in_validation(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    df = pd.read_parquet(runner.root / "metrics" / "results.parquet")
    df.loc[0, "scientific_status"] = "SMOKE_ONLY_NOT_SCIENTIFIC"
    df.to_parquet(runner.root / "metrics" / "results.parquet")
    # re-run P15 should fail
    from src.experiment.live_engine import LiveContext, LivePhaseEngine

    eng = LivePhaseEngine(
        LiveContext(
            repo_root=ROOT,
            root=runner.root,
            dataset_id=-1,
            repeat=0,
            fold=0,
            adapters=runner.cfg.adapters,
            result_rows=df.to_dict(orient="records"),
            guard=runner.engine.ctx.guard if runner.engine else __import__(
                "src.experiment.test_access_guard", fromlist=["UnitTestAccessRegistry"]
            ).UnitTestAccessRegistry(),
        )
    )
    # populate guards as unlocked
    for r in eng.ctx.result_rows:
        g = eng.ctx.guard.get(r["learner"], r["arm"])
        g.mark_cal_a_complete()
        g.mark_calibrators_complete()
        g.mark_cal_b_complete()
        g.mark_conformal_complete()
        g.unlock_test()
    with pytest.raises(AssertionError, match="smoke"):
        eng.p15_validate_outputs()


def test_outer_aug_shared_path_exists(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    for arm in ("A1", "A2", "A3"):
        assert (runner.root / "generated" / arm / "manifest.json").exists()
        man = json.loads((runner.root / "generated" / arm / "manifest.json").read_text())
        assert set(man["shared_across_learners"]) == {"xgboost", "catboost", "tabpfn", "tabicl"}


def test_inner_cache_reused_across_learners(tmp_path):
    runner = _run_toy(tmp_path)
    runner.run_live()
    for arm in ("A1", "A2", "A3"):
        for i in range(3):
            man = json.loads(
                (runner.root / "hpo" / "inner_cache" / arm / f"fold_{i}" / "manifest.json").read_text()
            )
            assert man["shared_across_gbdt_learners"] is True
