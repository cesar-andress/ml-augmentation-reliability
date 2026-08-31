#!/usr/bin/env python
"""Stage A objective screening + Stage B semantic-review CSV (decisions blank).

Does NOT download the entire TabArena suite. Uses a curated candidate ID list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.openml_loader import screen_candidates


# Curated smoke-scale candidate list (OpenML IDs). Full suite screening is deferred.
# Includes TabArena-ish / classic binary sets; not exhaustive.
DEFAULT_CANDIDATES = [
    # classic binary
    37,    # diabetes
    31,    # credit-g
    50,    # tic-tac-toe (may fail continuous)
    1510,  # wdbc
    1067,  # kc1
    1068,  # pc1
    1049,  # pc4
    1050,  # pc3
    1464,  # blood-transfusion
    40701, # churn
    40945, # Titanic
    1487,  # ozone-level-8hr
    44,    # spambase
    29,    # credit-approval
    15,    # breast-w
]


def main():
    cfg_path = ROOT / "configs" / "screening_candidates.yaml"
    if cfg_path.exists():
        ids = yaml.safe_load(cfg_path.read_text())["openml_ids"]
        pool = yaml.safe_load(cfg_path.read_text()).get("source_pool", "curated_smoke_list")
    else:
        ids = DEFAULT_CANDIDATES
        pool = "curated_smoke_list_not_full_tabarena"

    cand, review = screen_candidates(ids, source_pool=pool)
    out_dir = ROOT / "artifacts" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "dataset_candidates.csv"
    review_path = out_dir / "dataset_semantic_review.csv"
    cand.to_csv(cand_path, index=False)
    review.to_csv(review_path, index=False)
    print(f"candidates: {len(cand)} -> {cand_path}")
    print(f"semantic review: {len(review)} -> {review_path}")
    if "pass_objective" in cand.columns:
        print("objective pass IDs:", cand.loc[cand["pass_objective"] == True, "openml_id"].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
