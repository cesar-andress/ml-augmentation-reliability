"""Confirmatory protocol v1.2.1 validation — refuses incomplete or smoke configs."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import yaml

from src.experiment.hpo_v1_2_1 import HPO_CONFIG_PATH, load_hpo_config

A3_CONFIG_PATH = "configs/a3_protocol_v1_2_1.yaml"
FREEZE_V1_2_1_PATH = "artifacts/manifests/confirmatory_freeze_v1_2_1.yaml"
COHORT_SHA256 = "209bc80826843940da92799ca48b406a34df7e2dbafd2ff26590d092187ecb34"

ACCEPTED_PROTOCOL_VERSION = "1.2.1"
REJECTED_PROTOCOL_VERSIONS = ("1.2", "1.1")


class ConfirmatoryProtocolError(RuntimeError):
    """Raised when confirmatory execution preconditions are not met."""


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_a3_config(path: str = A3_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("protocol_version") != ACCEPTED_PROTOCOL_VERSION:
        raise ConfirmatoryProtocolError(
            f"A3 config protocol_version must be {ACCEPTED_PROTOCOL_VERSION!r}, got {cfg.get('protocol_version')!r}"
        )
    return cfg


def load_freeze_v1_2_1(path: str = FREEZE_V1_2_1_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_a3_protocol_mode(a3_cfg: dict[str, Any] | None = None) -> None:
    a3_cfg = a3_cfg or load_a3_config()
    if a3_cfg.get("scientific_mode") is not True:
        raise ConfirmatoryProtocolError("A3 scientific_mode must be true")
    if a3_cfg.get("smoke_settings_allowed") is not False:
        raise ConfirmatoryProtocolError("A3 smoke_settings_allowed must be false")

    ctor = a3_cfg["constructor"]
    for key, expected in [
        ("n_iter", 1000),
        ("num_timesteps", 1000),
        ("batch_size", 1024),
        ("is_classification", True),
    ]:
        if ctor.get(key) != expected:
            raise ConfirmatoryProtocolError(f"A3 constructor {key} must be {expected!r}, got {ctor.get(key)!r}")


def validate_hpo_config(hpo_cfg: dict[str, Any] | None = None) -> None:
    hpo_cfg = hpo_cfg or load_hpo_config()
    budget = hpo_cfg["standard_budget"]
    if int(budget["n_candidates"]) != 20:
        raise ConfirmatoryProtocolError("standard_hpo_candidates must be 20")
    if int(budget["inner_cv"]["n_folds"]) != 3:
        raise ConfirmatoryProtocolError("inner_cv_folds must be 3")
    forbidden = set(budget.get("forbidden_partitions_for_selection", []))
    if not {"CAL-A", "CAL-B", "TEST"}.issubset(forbidden):
        raise ConfirmatoryProtocolError("HPO must forbid CAL-A, CAL-B, and TEST for selection")


def validate_confirmatory_protocol_start(
    *,
    repo_root: str | Path,
    protocol_version: str,
    hpo_config_path: str | Path | None = None,
    a3_config_path: str | Path | None = None,
    freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    """Refuse confirmatory start unless v1.2.1 freeze artifacts match."""
    repo_root = Path(repo_root)
    if protocol_version in REJECTED_PROTOCOL_VERSIONS:
        raise ConfirmatoryProtocolError(
            f"confirmatory run refused: protocol_version {protocol_version!r} is superseded; "
            f"use {ACCEPTED_PROTOCOL_VERSION!r}"
        )
    if protocol_version != ACCEPTED_PROTOCOL_VERSION:
        raise ConfirmatoryProtocolError(
            f"confirmatory run refused: unknown protocol_version {protocol_version!r}"
        )

    hpo_path = repo_root / (hpo_config_path or HPO_CONFIG_PATH)
    a3_path = repo_root / (a3_config_path or A3_CONFIG_PATH)
    freeze_v121 = repo_root / (freeze_path or FREEZE_V1_2_1_PATH)

    if not freeze_v121.exists():
        raise ConfirmatoryProtocolError(f"missing confirmatory freeze: {freeze_v121}")

    freeze = load_freeze_v1_2_1(freeze_v121)
    if freeze.get("protocol_version") != ACCEPTED_PROTOCOL_VERSION:
        raise ConfirmatoryProtocolError("confirmatory_freeze_v1_2_1 protocol_version mismatch")

    cohort_sha = freeze.get("dataset_cohort_sha256")
    if cohort_sha != COHORT_SHA256:
        raise ConfirmatoryProtocolError(f"cohort SHA256 mismatch: {cohort_sha!r}")

    hpo_cfg = load_hpo_config(str(hpo_path))
    a3_cfg = load_a3_config(str(a3_path))
    validate_hpo_config(hpo_cfg)
    validate_a3_protocol_mode(a3_cfg)

    expected_hpo_sha = freeze["artifact_sha256"]["hpo_v1_2_1_yaml"]
    expected_a3_sha = freeze["artifact_sha256"]["a3_protocol_v1_2_1_yaml"]
    actual_hpo_sha = sha256_path(hpo_path)
    actual_a3_sha = sha256_path(a3_path)

    if actual_hpo_sha != expected_hpo_sha:
        raise ConfirmatoryProtocolError("hpo config hash does not match frozen hash")
    if actual_a3_sha != expected_a3_sha:
        raise ConfirmatoryProtocolError("A3 config hash does not match frozen hash")

    return {
        "protocol_version": ACCEPTED_PROTOCOL_VERSION,
        "hpo_config_sha256": actual_hpo_sha,
        "a3_config_sha256": actual_a3_sha,
        "cohort_sha256": cohort_sha,
        "scientific_mode": a3_cfg["scientific_mode"],
        "smoke_settings_allowed": a3_cfg["smoke_settings_allowed"],
        "standard_hpo_candidates": hpo_cfg["standard_budget"]["n_candidates"],
        "inner_cv_folds": hpo_cfg["standard_budget"]["inner_cv"]["n_folds"],
    }


def verify_synthcity_tabddpm_signature(
    *,
    synthcity_python: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect installed SynthCity 0.2.12 TabDDPMPlugin constructor defaults."""
    try:
        from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin

        return _tabddpm_signature_from_plugin(TabDDPMPlugin)
    except ModuleNotFoundError:
        repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        py = Path(synthcity_python or repo_root / ".venv_synthcity/bin/python")
        if not py.exists():
            return {
                "package": "synthcity",
                "version": "unknown",
                "verified": False,
                "error": f"synthcity not importable and {py} missing",
            }
        import json
        import subprocess

        script = """
import inspect, json
import importlib.metadata as im
from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin
sig = inspect.signature(TabDDPMPlugin.__init__)
defaults = {}
for name, param in sig.parameters.items():
    if name == "self":
        continue
    if param.default is not inspect.Parameter.empty:
        val = param.default
        if isinstance(val, (bool, int, float, str)):
            defaults[name] = val
        elif isinstance(val, dict):
            defaults[name] = val
        else:
            defaults[name] = repr(val)
print(json.dumps({"version": im.version("synthcity"), "defaults": defaults, "signature": str(sig)}))
"""
        proc = subprocess.run([str(py), "-c", script], capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout.strip())
        return _tabddpm_signature_from_defaults(
            version=payload["version"],
            signature=payload["signature"],
            defaults=payload["defaults"],
        )


def _tabddpm_signature_from_plugin(TabDDPMPlugin: Any) -> dict[str, Any]:
    sig = inspect.signature(TabDDPMPlugin.__init__)
    defaults: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    try:
        import importlib.metadata as im

        version = im.version("synthcity")
    except Exception:
        version = "unknown"
    return _tabddpm_signature_from_defaults(version=version, signature=str(sig), defaults=defaults)


def _tabddpm_signature_from_defaults(
    *,
    version: str,
    signature: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "is_classification": False,
        "n_iter": 1000,
        "lr": 0.002,
        "weight_decay": 0.0001,
        "batch_size": 1024,
        "num_timesteps": 1000,
        "gaussian_loss_type": "mse",
        "scheduler": "cosine",
        "model_type": "mlp",
        "log_interval": 100,
        "dim_embed": 128,
        "continuous_encoder": "quantile",
        "validation_size": 0,
        "random_state": 0,
        "compress_dataset": False,
        "sampling_patience": 500,
        "model_params": {},
        "cont_encoder_params": {},
    }

    mismatches = {}
    for key, exp in expected.items():
        if key not in defaults:
            mismatches[key] = {"expected": exp, "actual": "MISSING"}
            continue
        act = defaults[key]
        if act != exp:
            mismatches[key] = {"expected": exp, "actual": act}

    return {
        "package": "synthcity",
        "version": version,
        "plugin_class": "TabDDPMPlugin",
        "signature": signature,
        "defaults": {k: repr(v) for k, v in sorted(defaults.items())},
        "expected_defaults": expected,
        "mismatches": mismatches,
        "verified": version == "0.2.12" and not mismatches,
    }
