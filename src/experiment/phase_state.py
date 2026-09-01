"""Phase status persistence, resume, and idempotency for confirmatory units."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.confirmatory_constants import PHASE_STATUS_VALUES, PHASES


class PhaseConflictError(RuntimeError):
    """Raised when a completed phase conflicts with current configuration."""


class PhaseStateManager:
    def __init__(self, status_path: Path, *, config_hash: str):
        self.status_path = status_path
        self.config_hash = config_hash
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        if self.status_path.exists():
            state = json.loads(self.status_path.read_text(encoding="utf-8"))
            if state.get("config_hash") and state["config_hash"] != self.config_hash:
                raise PhaseConflictError(
                    f"unit config_hash mismatch: stored={state['config_hash']!r} current={self.config_hash!r}"
                )
            return state
        return {
            "config_hash": self.config_hash,
            "phases": {p: {"status": "PLANNED"} for p in PHASES},
        }

    def save(self) -> None:
        self.status_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def get_phase(self, phase: str) -> dict[str, Any]:
        return self._state["phases"][phase]

    def start_phase(self, phase: str, *, input_hashes: dict[str, str] | None = None) -> None:
        rec = self.get_phase(phase)
        rec.update(
            {
                "status": "RUNNING",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "input_hashes": input_hashes or {},
                "retry_count": rec.get("retry_count", 0),
            }
        )
        self.save()

    def complete_phase(
        self,
        phase: str,
        *,
        output_hashes: dict[str, str] | None = None,
        duration: float | None = None,
    ) -> None:
        rec = self.get_phase(phase)
        end = datetime.now(timezone.utc).isoformat()
        rec.update(
            {
                "status": "COMPLETE",
                "end_time": end,
                "output_hashes": output_hashes or {},
            }
        )
        if duration is not None:
            rec["duration"] = duration
        elif rec.get("start_time"):
            start = datetime.fromisoformat(rec["start_time"])
            end_dt = datetime.fromisoformat(end)
            rec["duration"] = (end_dt - start).total_seconds()
        self.save()

    def fail_phase(self, phase: str, *, exception: str, retry_count: int | None = None) -> None:
        rec = self.get_phase(phase)
        rec.update(
            {
                "status": "FAILED",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "exception": exception,
            }
        )
        if retry_count is not None:
            rec["retry_count"] = retry_count
        self.save()

    def can_reuse_phase(self, phase: str, *, input_hashes: dict[str, str], output_checksum: str) -> bool:
        rec = self.get_phase(phase)
        if rec.get("status") != "COMPLETE":
            return False
        stored_in = rec.get("input_hashes") or {}
        if stored_in and stored_in != input_hashes:
            return False
        stored_out = (rec.get("output_hashes") or {}).get("checksum")
        return stored_out == output_checksum

    def assert_no_complete_conflict(
        self,
        phase: str,
        *,
        input_hashes: dict[str, str],
        output_checksum: str,
    ) -> None:
        rec = self.get_phase(phase)
        if rec.get("status") != "COMPLETE":
            return
        if not self.can_reuse_phase(phase, input_hashes=input_hashes, output_checksum=output_checksum):
            raise PhaseConflictError(f"conflicting COMPLETE phase {phase}; aborting")

    def should_run_phase(
        self,
        phase: str,
        *,
        resume: bool,
        force_rerun_failed: bool,
        input_hashes: dict[str, str],
        output_checksum: str | None = None,
    ) -> bool:
        rec = self.get_phase(phase)
        status = rec.get("status", "PLANNED")
        if status == "COMPLETE":
            if output_checksum and self.can_reuse_phase(
                phase, input_hashes=input_hashes, output_checksum=output_checksum
            ):
                return False
            self.assert_no_complete_conflict(
                phase, input_hashes=input_hashes, output_checksum=output_checksum or ""
            )
            return False
        if status == "FAILED":
            return force_rerun_failed
        if resume and status == "COMPLETE":
            return False
        return True


def compute_config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
