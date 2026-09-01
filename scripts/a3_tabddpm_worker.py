#!/usr/bin/env python
"""SynthCity TabDDPM worker (runs inside .venv_synthcity).

Reads job_dir/{train.parquet,y.npy,meta.json,gen_config.json}
Writes synthetic_minority.parquet and status.json

Protocol mode (scientific_mode=true):
  TabDDPM is_classification=True on full TRAIN with target column;
  generate k rows conditioned on the minority label.

Smoke mode (default / SMOKE_ONLY):
  minority-feature-only fit with tiny n_iter (NON-SCIENTIFIC).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def main(job_dir: Path) -> int:
    status = {"status": "fail"}
    try:
        X = pd.read_parquet(job_dir / "train.parquet")
        y = np.load(job_dir / "y.npy")
        meta = json.loads((job_dir / "meta.json").read_text())
        cfg = json.loads((job_dir / "gen_config.json").read_text())

        minority_label = int(meta["minority_label"])
        k = int(meta["k"])
        if k < 1:
            raise RuntimeError(f"invalid k={k}")

        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        random_state = int(cfg.get("random_state", 0))
        scientific = bool(cfg.get("scientific_mode", False)) and not str(cfg.get("mark", "")).startswith("SMOKE")

        if scientific:
            # Frozen Protocol v1.2.1 A3 protocol-mode constructor settings
            plugin_kwargs = {
                "is_classification": True,
                "n_iter": int(cfg.get("n_iter", 1000)),
                "lr": float(cfg.get("lr", 0.002)),
                "weight_decay": float(cfg.get("weight_decay", 0.0001)),
                "batch_size": int(cfg.get("batch_size", 1024)),
                "num_timesteps": int(cfg.get("num_timesteps", 1000)),
                "gaussian_loss_type": str(cfg.get("gaussian_loss_type", "mse")),
                "scheduler": str(cfg.get("scheduler", "cosine")),
                "model_type": str(cfg.get("model_type", "mlp")),
                "model_params": dict(cfg.get("model_params") or {}),
                "dim_embed": int(cfg.get("dim_embed", 128)),
                "continuous_encoder": str(cfg.get("continuous_encoder", "quantile")),
                "cont_encoder_params": dict(cfg.get("cont_encoder_params") or {}),
                "validation_size": float(cfg.get("validation_size", 0)),
                "compress_dataset": bool(cfg.get("compress_dataset", False)),
                "sampling_patience": int(cfg.get("sampling_patience", 500)),
                "random_state": random_state,
                "device": str(cfg.get("device", "cuda")),
            }
            target_col = "__target__"
            df = X.copy()
            df[target_col] = y
            loader = GenericDataLoader(df, target_column=target_col, random_state=random_state)

            t_fit0 = time.perf_counter()
            plugin = Plugins().get("ddpm", **plugin_kwargs)
            plugin.fit(loader)
            fit_s = time.perf_counter() - t_fit0

            cond = np.full(k, minority_label)
            t_s0 = time.perf_counter()
            syn = plugin.generate(count=k, cond=cond)
            sample_s = time.perf_counter() - t_s0
            mark = cfg.get("mark", "")
        else:
            # Smoke / non-scientific tiny path (historical smoke identity)
            X_min = X.loc[y == minority_label].copy()
            if len(X_min) < 2:
                raise RuntimeError(f"too few minority rows for TabDDPM: {len(X_min)}")
            plugin_kwargs = {
                "n_iter": int(cfg.get("n_iter", 2)),
                "batch_size": int(cfg.get("batch_size", 64)),
                "num_timesteps": int(cfg.get("num_timesteps", 10)),
                "random_state": random_state,
                "is_classification": False,
            }
            t_fit0 = time.perf_counter()
            plugin = Plugins().get("ddpm", **plugin_kwargs)
            loader = GenericDataLoader(X_min)
            plugin.fit(loader)
            fit_s = time.perf_counter() - t_fit0

            t_s0 = time.perf_counter()
            syn = plugin.generate(count=k)
            sample_s = time.perf_counter() - t_s0
            mark = cfg.get("mark", "SMOKE_ONLY_NOT_SCIENTIFIC")

        if hasattr(syn, "dataframe"):
            syn_df = syn.dataframe()
        elif isinstance(syn, pd.DataFrame):
            syn_df = syn
        else:
            syn_df = pd.DataFrame(np.asarray(syn), columns=X.columns)

        # Drop target column if present; align to feature columns only
        if "__target__" in syn_df.columns:
            syn_df = syn_df.drop(columns=["__target__"])
        syn_df = syn_df.reindex(columns=list(X.columns))
        out_path = job_dir / "synthetic_minority.parquet"
        syn_df.to_parquet(out_path)

        if len(syn_df) != k:
            # accept and let main env enforce exact-k policy
            pass

        status = {
            "status": "ok",
            "fit_seconds": fit_s,
            "sample_seconds": sample_s,
            "n_requested": k,
            "n_generated": int(len(syn_df)),
            "plugin": "ddpm",
            "plugin_class": "TabDDPMPlugin",
            "plugin_kwargs": plugin_kwargs,
            "random_state": random_state,
            "mark": mark,
            "scientific_mode": scientific,
            "config_sha256": cfg.get("config_sha256"),
        }
        (job_dir / "status.json").write_text(json.dumps(status, indent=2, default=str))
        return 0
    except Exception as e:
        status = {
            "status": "fail",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        (job_dir / "status.json").write_text(json.dumps(status, indent=2))
        print(status["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: a3_tabddpm_worker.py JOB_DIR", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
