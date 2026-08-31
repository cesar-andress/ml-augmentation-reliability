"""Reproducibility manifest creation and persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--porcelain"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in [
        "xgboost",
        "catboost",
        "tabpfn",
        "tabicl",
        "imblearn",
        "sklearn",
        "statsmodels",
        "openml",
        "pandas",
        "numpy",
        "scipy",
        "pyarrow",
        "yaml",
        "joblib",
        "psutil",
        "torch",
    ]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            out[name] = f"MISSING:{type(e).__name__}"
    return out


def create_manifest(
    *,
    repo_root: str | Path,
    dataset_id: Any,
    dataset_name: str,
    dataset_version: Any,
    dataset_checksum: str | None,
    fold: int,
    repeat: int,
    seed: int,
    learner: str,
    arm: str,
    preprocessing_config: dict[str, Any],
    augmentation_config: dict[str, Any],
    hyperparameters: dict[str, Any],
    checkpoint_paths: dict[str, str] | None = None,
    timings: dict[str, float] | None = None,
    peak_gpu_memory_mb: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    checkpoint_paths = checkpoint_paths or {}
    checkpoint_identities = {
        k: {"path": str(v), "sha256": _sha256_file(v)} for k, v in checkpoint_paths.items()
    }

    gpu = {}
    try:
        import subprocess as sp

        line = sp.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        name, driver = [x.strip() for x in line.split(",")[:2]]
        smi = sp.check_output(["nvidia-smi"], text=True)
        cuda = None
        for ln in smi.splitlines():
            if "CUDA Version:" in ln:
                cuda = ln.split("CUDA Version:")[-1].strip().split()[0]
                break
        gpu = {"name": name, "driver": driver, "cuda_runtime_nvidia_smi": cuda}
    except Exception as e:
        gpu = {"error": f"{type(e).__name__}: {e}"}

    manifest = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "git": _git_info(repo_root),
        "cpu": platform.processor() or platform.machine(),
        "cpu_model": _cpu_model(),
        "gpu": gpu,
        "python": {
            "version": platform.python_version(),
            "executable": os.sys.executable,
        },
        "package_versions": _package_versions(),
        "checkpoints": checkpoint_identities,
        "dataset": {
            "openml_id": dataset_id,
            "name": dataset_name,
            "version": dataset_version,
            "checksum": dataset_checksum,
        },
        "fold": fold,
        "repeat": repeat,
        "seed": seed,
        "learner": learner,
        "arm": arm,
        "preprocessing": preprocessing_config,
        "augmentation": augmentation_config,
        "hyperparameters": hyperparameters,
        "timings": timings or {},
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "extra": extra or {},
    }
    return manifest


def _cpu_model() -> str | None:
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if "Model name" in line or "Nombre del modelo" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def save_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
