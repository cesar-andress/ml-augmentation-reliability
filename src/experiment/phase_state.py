"""Phase status persistence, resume, and idempotency for confirmatory units."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.confirmatory_constants import PHASE_STATUS_VALUES, PHASES
from src.experiment.phase_artifacts import ProvenanceConflictError, validate_stored_inventory


class PhaseConflictError(RuntimeError):
    """Raised when a completed phase conflicts with current configuration."""


class PhaseStateManager:
    def __init__(
        self,
        status_path: Path,
        *,
        config_hash: str,
        allow_hash_mismatch_reset: bool = False,
        unit_root: Path | None = None,
    ):
        self.status_path = status_path
        self.config_hash = config_hash
        self.allow_hash_mismatch_reset = allow_hash_mismatch_reset
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if unit_root is not None:
            self._unit_root = Path(unit_root)
        elif status_path.name == "unit_status.json" and status_path.parent.name == "status":
            self._unit_root = status_path.parent.parent
        else:
            self._unit_root = status_path.parent
        self._state = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        if self.status_path.exists():
            state = json.loads(self.status_path.read_text(encoding="utf-8"))
            if state.get("config_hash") and state["config_hash"] != self.config_hash:
                # Never silently wipe a finalized confirmatory unit
                complete_marker = self._unit_root / "status" / "unit_complete.json"
                if complete_marker.exists():
                    raise PhaseConflictError(
                        f"refusing config_hash reset: finalized unit exists at {complete_marker}; "
                        f"stored={state['config_hash']!r} current={self.config_hash!r}"
                    )
                if self.allow_hash_mismatch_reset:
                    return {
                        "config_hash": self.config_hash,
                        "phases": {p: {"status": "PLANNED"} for p in PHASES},
                    }
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

    def _append_history(self, phase: str, entry: dict[str, Any]) -> None:
        rec = self.get_phase(phase)
        hist = list(rec.get("attempt_history") or [])
        hist.append(entry)
        rec["attempt_history"] = hist

    def start_phase(self, phase: str, *, input_hashes: dict[str, str] | None = None) -> None:
        rec = self.get_phase(phase)
        prior_status = rec.get("status", "PLANNED")
        attempt_no = int(rec.get("retry_count", 0)) + (1 if prior_status == "FAILED" else 0)
        if prior_status == "FAILED":
            # Preserve failure evidence before clearing active exception for the new attempt.
            self._append_history(
                phase,
                {
                    "attempt": attempt_no,
                    "prior_status": "FAILED",
                    "start_time": rec.get("start_time"),
                    "end_time": rec.get("end_time"),
                    "exception": rec.get("exception"),
                    "output_checksum": (rec.get("output_hashes") or {}).get("checksum"),
                    "output_artifacts": rec.get("output_artifacts"),
                },
            )
            rec["retry_count"] = attempt_no
        rec.update(
            {
                "status": "RUNNING",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "input_hashes": input_hashes or {},
                "retry_count": rec.get("retry_count", 0),
                "exception": None,
            }
        )
        # Drop stale failure fields from active record
        rec.pop("failure_reason", None)
        self.save()

    def complete_phase(
        self,
        phase: str,
        *,
        output_hashes: dict[str, str] | None = None,
        output_artifacts: list[dict[str, Any]] | None = None,
        duration: float | None = None,
    ) -> None:
        rec = self.get_phase(phase)
        end = datetime.now(timezone.utc).isoformat()
        update: dict[str, Any] = {
            "status": "COMPLETE",
            "end_time": end,
            "output_hashes": output_hashes or {},
            "exception": None,
        }
        if output_artifacts is not None:
            update["output_artifacts"] = output_artifacts
        rec.update(update)
        if duration is not None:
            rec["duration"] = duration
        elif rec.get("start_time"):
            start = datetime.fromisoformat(rec["start_time"])
            end_dt = datetime.fromisoformat(end)
            rec["duration"] = (end_dt - start).total_seconds()
        self._append_history(
            phase,
            {
                "attempt": int(rec.get("retry_count", 0)),
                "prior_status": "RUNNING",
                "status": "COMPLETE",
                "start_time": rec.get("start_time"),
                "end_time": end,
                "exception": None,
                "output_checksum": (output_hashes or {}).get("checksum"),
                "output_artifacts": output_artifacts,
            },
        )
        self.save()

    def fail_phase(self, phase: str, *, exception: str, retry_count: int | None = None) -> None:
        rec = self.get_phase(phase)
        end = datetime.now(timezone.utc).isoformat()
        rec.update(
            {
                "status": "FAILED",
                "end_time": end,
                "exception": exception,
            }
        )
        if retry_count is not None:
            rec["retry_count"] = retry_count
        self._append_history(
            phase,
            {
                "attempt": int(rec.get("retry_count", 0)),
                "prior_status": "RUNNING",
                "status": "FAILED",
                "start_time": rec.get("start_time"),
                "end_time": end,
                "exception": exception,
                "output_checksum": None,
                "output_artifacts": None,
            },
        )
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

    def validate_complete_for_resume(self, phase: str, *, input_hashes: dict[str, str]) -> None:
        """Require persisted COMPLETE outputs to exist and match stored inventories."""
        rec = self.get_phase(phase)
        if rec.get("status") != "COMPLETE":
            return
        try:
            validate_stored_inventory(
                self._unit_root,
                phase,
                stored_artifacts=rec.get("output_artifacts"),
                stored_checksum=(rec.get("output_hashes") or {}).get("checksum"),
                input_hashes=rec.get("input_hashes") or {},
                current_input_hashes=input_hashes,
            )
        except ProvenanceConflictError as e:
            raise PhaseConflictError(str(e)) from e

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
            if not resume:
                # Fresh run colliding with prior COMPLETE requires explicit checksum match
                if output_checksum is None:
                    raise PhaseConflictError(
                        f"phase {phase} already COMPLETE; use --resume or clear unit status"
                    )
                if self.can_reuse_phase(phase, input_hashes=input_hashes, output_checksum=output_checksum):
                    self.validate_complete_for_resume(phase, input_hashes=input_hashes)
                    return False
                self.assert_no_complete_conflict(
                    phase, input_hashes=input_hashes, output_checksum=output_checksum
                )
                return False
            # --resume: reuse COMPLETE only after validating persisted outputs
            self.validate_complete_for_resume(phase, input_hashes=input_hashes)
            if output_checksum is not None:
                if self.can_reuse_phase(phase, input_hashes=input_hashes, output_checksum=output_checksum):
                    return False
                self.assert_no_complete_conflict(
                    phase, input_hashes=input_hashes, output_checksum=output_checksum
                )
            return False
        if status == "FAILED":
            return force_rerun_failed
        return True


def compute_config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
