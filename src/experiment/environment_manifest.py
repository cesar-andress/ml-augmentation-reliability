"""Environment manifest for confirmatory units — engineering provenance only."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.confirmatory_constants import (
    A3_CONFIG_SHA256,
    CHECKPOINT_SHA256,
    CHECKPOINTS,
    FREEZE_TAG,
    HPO_CONFIG_SHA256,
    PROTOCOL_VERSION,
    SYNTHCITY_PYTHON,
)


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_environment_manifest(
    *,
    repo_root: Path,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    main_py = Path(sys.executable)
    synth_py = repo_root / SYNTHCITY_PYTHON

    gpu_name = None
    gpu_vram_mb = None
    nvidia_driver = None
    cuda_runtime = None
    try:
        import torch

        cuda_runtime = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_mb = float(torch.cuda.get_device_properties(0).total_memory / (1024**2))
    except Exception:
        pass
    nvidia_smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nvidia_smi:
        nvidia_driver = nvidia_smi.split("\n")[0].strip()

    ram_bytes = None
    try:
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass

    synthcity_version = None
    synthcity_py_version = None
    synthcity_torch = None
    if synth_py.exists():
        synthcity_py_version = _run([str(synth_py), "-c", "import sys; print(sys.version.split()[0])"])
        synthcity_version = _run(
            [str(synth_py), "-c", "import importlib.metadata as m; print(m.version('synthcity'))"]
        )
        synthcity_torch = _run([str(synth_py), "-c", "import torch; print(torch.__version__)"])

    git_head = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])

    tabpfn_path = repo_root / CHECKPOINTS["tabpfn"]
    tabicl_path = repo_root / CHECKPOINTS["tabicl"]
    tabpfn_sha = _sha256_file(tabpfn_path)
    tabicl_sha = _sha256_file(tabicl_path)

    freeze_tag = (validation or {}).get("freeze_tag", FREEZE_TAG)
    protocol_version = (validation or {}).get("protocol_version", PROTOCOL_VERSION)
    hpo_sha = (validation or {}).get("hpo_config_sha256", HPO_CONFIG_SHA256)
    a3_sha = (validation or {}).get("a3_config_sha256", A3_CONFIG_SHA256)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host_os": platform.platform(),
        "kernel": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "gpu_identity": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "nvidia_driver": nvidia_driver,
        "cuda_runtime": cuda_runtime,
        "main_python_executable": str(main_py),
        "main_python_version": platform.python_version(),
        "main_package_versions": {
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "pandas": _pkg_version("pandas"),
            "scikit-learn": _pkg_version("scikit-learn"),
            "imbalanced-learn": _pkg_version("imbalanced-learn"),
            "xgboost": _pkg_version("xgboost"),
            "catboost": _pkg_version("catboost"),
            "torch": _pkg_version("torch"),
            "tabpfn": _pkg_version("tabpfn"),
            "tabicl": _pkg_version("tabicl"),
        },
        "synthcity_python_executable": str(synth_py) if synth_py.exists() else None,
        "synthcity_python_version": synthcity_py_version,
        "synthcity_version": synthcity_version,
        "synthcity_torch_version": synthcity_torch,
        "git_head": git_head,
        "freeze_tag": freeze_tag,
        "protocol_version": protocol_version,
        "hpo_config_sha256": hpo_sha,
        "a3_config_sha256": a3_sha,
        "tabpfn_checkpoint_sha256": tabpfn_sha,
        "tabicl_checkpoint_sha256": tabicl_sha,
        "tabpfn_checkpoint_sha256_expected": CHECKPOINT_SHA256["tabpfn"],
        "tabicl_checkpoint_sha256_expected": CHECKPOINT_SHA256["tabicl"],
    }


def write_environment_manifest(root: Path, *, repo_root: Path, validation: dict[str, Any] | None) -> dict[str, Any]:
    from src.logging_utils import write_json

    man = build_environment_manifest(repo_root=repo_root, validation=validation)
    path = root / "manifests" / "environment_manifest.json"
    write_json(path, man)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    man_meta = {"path": "manifests/environment_manifest.json", "sha256": digest, "size": path.stat().st_size}
    return {"manifest": man, "artifact": man_meta}
