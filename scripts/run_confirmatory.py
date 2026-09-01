#!/usr/bin/env python
"""Canonical confirmatory execution entry point — Protocol v1.2.1."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment.confirmatory_runner import ConfirmatoryRunConfig, ConfirmatoryRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Augmentation Reliability confirmatory runner v1.2.1")
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--dry-run", action="store_true", help="Plan only; no model/data/CUDA access")
    p.add_argument("--resume", action="store_true", help="Reuse validated completed phases")
    p.add_argument(
        "--force-rerun-failed",
        action="store_true",
        help="Rerun FAILED phases only; never overwrite valid COMPLETE phases",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Safety: refuse accidental live confirmatory unless dry-run or explicit env override
    # (toy/integration tests import ConfirmatoryRunner directly with adapters).
    if not args.dry_run:
        import os

        if os.environ.get("CONFIRMATORY_LIVE_AUTHORIZED") != "YES":
            print(
                "REFUSING live confirmatory execution without CONFIRMATORY_LIVE_AUTHORIZED=YES.\n"
                "Use --dry-run, or authorize only when READY_FOR_CONFIRMATORY_EXECUTION == YES.",
                file=sys.stderr,
            )
            return 2

    cfg = ConfirmatoryRunConfig(
        repo_root=ROOT,
        dataset_id=args.dataset_id,
        repeat=args.repeat,
        fold=args.fold,
        dry_run=args.dry_run,
        resume=args.resume,
        force_rerun_failed=args.force_rerun_failed,
        log_level=args.log_level,
    )
    runner = ConfirmatoryRunner(cfg)
    result = runner.run()
    print(json.dumps({k: v for k, v in result.items() if k != "plan"}, indent=2))
    if args.dry_run and "plan" in result:
        plan = result["plan"]
        summary = {
            "unit_id": plan["unit_id"],
            "scientific_status": plan["scientific_status"],
            "expected_final_cell_count": plan["expected_final_cell_count"],
            "hpo_standard_candidates": plan["hpo"]["standard_candidates"],
            "inner_cv_folds": plan["hpo"]["inner_cv_folds"],
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
