"""Explicit TEST partition access guard for confirmatory cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class ConfirmatoryTestAccessError(RuntimeError):
    """Raised when TEST is accessed before calibration/conformal phases complete."""


@dataclass
class CellTestAccessGuard:
    """Per-cell guard: TEST predictions forbidden until post-calibration phases complete."""

    learner: str
    arm: str
    cal_a_complete: bool = False
    calibrators_complete: bool = False
    cal_b_complete: bool = False
    conformal_complete: bool = False
    test_unlocked_at: str | None = None

    def mark_cal_a_complete(self) -> None:
        self.cal_a_complete = True

    def mark_calibrators_complete(self) -> None:
        self.calibrators_complete = True

    def mark_cal_b_complete(self) -> None:
        self.cal_b_complete = True

    def mark_conformal_complete(self) -> None:
        self.conformal_complete = True

    def unlock_test(self) -> None:
        if not self._prerequisites_met():
            raise ConfirmatoryTestAccessError(
                f"TEST unlock refused for {self.learner}/{self.arm}: "
                "P09-P12 prerequisites not complete"
            )
        self.test_unlocked_at = datetime.now(timezone.utc).isoformat()

    def assert_test_allowed(self) -> None:
        if self.test_unlocked_at is None:
            raise ConfirmatoryTestAccessError(
                f"TEST access blocked for {self.learner}/{self.arm}: "
                "calibration/conformal phases incomplete"
            )

    def _prerequisites_met(self) -> bool:
        return (
            self.cal_a_complete
            and self.calibrators_complete
            and self.cal_b_complete
            and self.conformal_complete
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner": self.learner,
            "arm": self.arm,
            "cal_a_complete": self.cal_a_complete,
            "calibrators_complete": self.calibrators_complete,
            "cal_b_complete": self.cal_b_complete,
            "conformal_complete": self.conformal_complete,
            "test_unlocked_at": self.test_unlocked_at,
        }


@dataclass
class UnitTestAccessRegistry:
    cells: dict[tuple[str, str], CellTestAccessGuard] = field(default_factory=dict)

    def get(self, learner: str, arm: str) -> CellTestAccessGuard:
        key = (learner, arm)
        if key not in self.cells:
            self.cells[key] = CellTestAccessGuard(learner=learner, arm=arm)
        return self.cells[key]

    def predict_test(self, learner: str, arm: str) -> None:
        self.get(learner, arm).assert_test_allowed()
