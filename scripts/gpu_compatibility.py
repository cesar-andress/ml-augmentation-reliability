#!/usr/bin/env python
"""Minimal real GPU compatibility tests. Do not infer from install alone."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "environment" / "gpu_compatibility.json"


def peak_mb():
    import torch

    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    return None


def run_case(name, fn):
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        detail = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "name": name,
            "status": "PASS",
            "wall_clock_seconds": time.perf_counter() - t0,
            "peak_vram_mb": peak_mb(),
            "exception": None,
            "detail": detail,
        }
    except Exception as e:
        return {
            "name": name,
            "status": "FAIL",
            "wall_clock_seconds": time.perf_counter() - t0,
            "peak_vram_mb": peak_mb(),
            "exception": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "detail": None,
        }


def test_torch_cuda():
    import torch

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    y = x @ x.T
    return {"result_mean": float(y.mean().item()), "cuda": True}


def test_xgboost_cuda():
    import numpy as np
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] > 0).astype(int)
    clf = XGBClassifier(device="cuda", tree_method="hist", n_estimators=10, max_depth=3)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:5, 1]
    return {"proba_head": p.tolist()}


def test_tabpfn():
    import numpy as np
    import torch
    from tabpfn import TabPFNClassifier

    ckpt = ROOT / "checkpoints" / "tabpfn" / "tabpfn-v2-classifier.ckpt"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    clf = TabPFNClassifier(
        n_estimators=8,
        model_path=str(ckpt),
        device="cuda",
        ignore_pretraining_limits=False,
        inference_precision=torch.float32,
        n_preprocessing_jobs=1,
        random_state=0,
        softmax_temperature=1.0,
        auto_scale_n_estimators=False,
    )
    clf.fit(X, y)
    p = clf.predict_proba(X[:8])[:, 1]
    return {"ckpt": str(ckpt), "proba_head": p.astype(float).tolist()}


def test_tabicl():
    import numpy as np
    from tabicl import TabICLClassifier

    ckpt = ROOT / "checkpoints" / "tabicl" / "tabicl-classifier-v1-20250208.ckpt"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    clf = TabICLClassifier(
        n_estimators=8,
        model_path=str(ckpt),
        checkpoint_version="tabicl-classifier-v1-20250208.ckpt",
        use_amp=False,
        batch_size=8,
        average_logits=True,
        random_state=42,
        softmax_temperature=1.0,
        device="cuda",
        allow_auto_download=False,
    )
    clf.fit(X, y)
    p = clf.predict_proba(X[:8])[:, 1]
    return {"ckpt": str(ckpt), "proba_head": p.astype(float).tolist()}


def test_synthcity_tabddpm():
    """Run via synthcity env subprocess for isolation."""
    import subprocess
    import tempfile

    import numpy as np
    import pandas as pd

    job = Path(tempfile.mkdtemp(prefix="gpu_a3_", dir=str(ROOT / "artifacts" / "smoke")))
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"f{i}" for i in range(4)])
    y = np.array([0] * 25 + [1] * 15)
    X.to_parquet(job / "train.parquet")
    np.save(job / "y.npy", y)
    (job / "meta.json").write_text(
        json.dumps(
            {
                "numeric_cols": list(X.columns),
                "categorical_cols": [],
                "missing_indicator_cols": [],
                "category_levels": {},
                "unknown_category_sentinel": -1,
                "output_columns": list(X.columns),
                "majority_label": 0,
                "minority_label": 1,
                "k": 5,
            }
        )
    )
    (job / "gen_config.json").write_text(
        json.dumps({"n_iter": 2, "batch_size": 32, "random_state": 0, "mark": "SMOKE_ONLY_NOT_SCIENTIFIC"})
    )
    py = ROOT / ".venv_synthcity" / "bin" / "python"
    worker = ROOT / "scripts" / "a3_tabddpm_worker.py"
    proc = subprocess.run([str(py), str(worker), str(job)], capture_output=True, text=True)
    status = json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {}
    if proc.returncode != 0 or status.get("status") != "ok":
        raise RuntimeError(f"TabDDPM fail: {status}\n{proc.stderr}")
    return {"job_dir": str(job), "status": status}


def main():
    cases = [
        ("A_torch_cuda", test_torch_cuda),
        ("B_xgboost_cuda", test_xgboost_cuda),
        ("C_tabpfn_v2", test_tabpfn),
        ("D_tabicl_v1", test_tabicl),
        ("E_synthcity_tabddpm", test_synthcity_tabddpm),
    ]
    results = [run_case(n, f) for n, f in cases]
    payload = {
        "results": results,
        "summary": {r["name"]: r["status"] for r in results},
        "all_pass": all(r["status"] == "PASS" for r in results),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print("wrote", OUT)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
