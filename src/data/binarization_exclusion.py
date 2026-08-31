"""Artificial binarization exclusion (Protocol v1.2) — deterministic, not ID-special-cased."""

from __future__ import annotations

import re

BINARIZED_DESCRIPTION_RE = re.compile(
    r"binarized version of the original data set",
    re.IGNORECASE,
)
BINARY_CLASS_TARGET = "binaryClass"


def is_artificial_binarization(
    *,
    description: str | None,
    default_target_attribute: str | None,
) -> tuple[bool, str]:
    """Return (excluded, reason)."""
    target = (default_target_attribute or "").strip()
    if target == BINARY_CLASS_TARGET:
        return True, "default_target_attribute_is_binaryClass"
    desc = description or ""
    if BINARIZED_DESCRIPTION_RE.search(desc):
        return True, "description_matches_binarized_version_of_the_original_data_set"
    return False, ""
