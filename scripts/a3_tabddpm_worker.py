#!/usr/bin/env python
"""SynthCity TabDDPM worker (runs inside .venv_synthcity).

Reads job_dir/{train.parquet,y.npy,meta.json,gen_config.json}
Writes synthetic_minority.parquet and status.json
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
        X_min = X.loc[y == minority_label].copy()
        if len(X_min) < 2:
            raise RuntimeError(f"too few minority rows for TabDDPM: {len(X_min)}")

        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        # Tiny smoke-only configuration.
        # In synthcity==0.2.12 TabDDPM is registered as plugin name "ddpm"
        # (class TabDDPMPlugin). This is the library identity, not a substitute algorithm.
        random_state = int(cfg.get("random_state", 0))
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

        if hasattr(syn, "dataframe"):
            syn_df = syn.dataframe()
        elif isinstance(syn, pd.DataFrame):
            syn_df = syn
        else:
            syn_df = pd.DataFrame(np.asarray(syn), columns=X.columns)

        # Align columns
        syn_df = syn_df.reindex(columns=list(X.columns))
        out_path = job_dir / "synthetic_minority.parquet"
        syn_df.to_parquet(out_path)

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
            "mark": cfg.get("mark", "SMOKE_ONLY_NOT_SCIENTIFIC"),
        }
        (job_dir / "status.json").write_text(json.dumps(status, indent=2))
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
