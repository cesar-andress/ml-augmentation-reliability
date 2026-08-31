"""Semantic exclusion flagging from metadata only (no LLM judgment)."""

from __future__ import annotations

import re
from typing import Any


# Keyword evidence only — proposes decisions solely when metadata is explicit.
ISSUE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "C_simulated_or_synthetic": [
        re.compile(r"\bsynthetic\b", re.I),
        re.compile(r"\bsimulated\b", re.I),
        re.compile(r"artificially generated", re.I),
        re.compile(r"\btoy dataset\b", re.I),
    ],
    "D_fictional": [
        re.compile(r"\bfictional\b", re.I),
        re.compile(r"made[- ]up", re.I),
        re.compile(r"fake patients", re.I),
    ],
    "B_artificial_ovr": [
        re.compile(r"one[- ]vs[- ]rest", re.I),
        re.compile(r"one vs\.? all", re.I),
        re.compile(r"\bOvR\b"),
        re.compile(r"rest class", re.I),
        re.compile(r"binarized from multi", re.I),
    ],
    "E_non_iid": [
        re.compile(r"non[- ]i\.?i\.?d", re.I),
    ],
    "F_repeated_measurements": [
        re.compile(r"repeated measure", re.I),
        re.compile(r"multiple observations per", re.I),
    ],
    "G_grouped_observations": [
        re.compile(r"grouped by (patient|subject|user|customer)", re.I),
        re.compile(r"hierarchical data", re.I),
    ],
    "H_time_series_temporal": [
        re.compile(r"time[- ]series", re.I),
        re.compile(r"\btemporal\b", re.I),
        re.compile(r"longitudinal", re.I),
        re.compile(r"over time", re.I),
    ],
    "I_subject_leakage_risk": [
        re.compile(r"same (patient|subject).*(train|test)", re.I),
        re.compile(r"patient[- ]level leakage", re.I),
    ],
    "A_duplicate_derivative": [
        re.compile(r"\bderivative of\b", re.I),
        re.compile(r"re[- ]?upload of", re.I),
        re.compile(r"variant of dataset", re.I),
    ],
    "J_unclear_provenance": [],
}


def scan_description(text: str) -> list[tuple[str, str]]:
    """Return list of (issue_code, matched_snippet)."""
    text = text or ""
    hits: list[tuple[str, str]] = []
    for code, patterns in ISSUE_PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 40)
                hits.append((code, text[start:end].replace("\n", " ")))
                break
    return hits


def propose_semantic_row(
    *,
    resolved_openml_id: int,
    dataset_name: str,
    source_pool: str,
    objective_eligible: bool,
    description: str,
    tabarena_meta: dict[str, Any] | None = None,
    climb_meta: dict[str, Any] | None = None,
    dtype_disagreement: bool = False,
) -> dict[str, Any]:
    """Metadata-only proposals. Ambiguous -> HUMAN_REVIEW_REQUIRED."""
    blob = description or ""
    if tabarena_meta:
        blob += "\n" + " ".join(str(v) for v in tabarena_meta.values() if v is not None)
    if climb_meta:
        blob += "\n" + " ".join(str(v) for v in climb_meta.values() if v is not None)

    hits = scan_description(blob)
    if dtype_disagreement:
        hits.append(("J_unclear_provenance", "OpenML categorical_indicator vs observed dtype disagreement"))

    suspected = ";".join(sorted({h[0] for h in hits})) if hits else ""
    evidence = " | ".join(f"{c}:{s}" for c, s in hits) if hits else ""

    # Only propose KEEP/EXCLUDE when evidence is explicit and strong.
    proposed_decision = ""
    decision = ""
    decision_reason = ""
    review_status = "CLEAR"

    strong_exclude_codes = {
        "C_simulated_or_synthetic",
        "D_fictional",
        "B_artificial_ovr",
    }
    hit_codes = {h[0] for h in hits}

    if not objective_eligible:
        review_status = "N/A_OBJECTIVE_FAIL"
        proposed_decision = "EXCLUDE"
        decision = "EXCLUDE"
        decision_reason = "failed_objective_filters"
    elif hit_codes & strong_exclude_codes:
        # Explicit metadata keywords — propose EXCLUDE but still require human confirm
        # unless phrasing is unambiguous boilerplate.
        review_status = "HUMAN_REVIEW_REQUIRED"
        proposed_decision = "EXCLUDE"
        decision = ""
        decision_reason = ""
    elif hit_codes & {
        "E_non_iid",
        "F_repeated_measurements",
        "G_grouped_observations",
        "H_time_series_temporal",
        "I_subject_leakage_risk",
        "A_duplicate_derivative",
        "J_unclear_provenance",
    }:
        review_status = "HUMAN_REVIEW_REQUIRED"
        proposed_decision = "EXCLUDE"
        decision = ""
        decision_reason = ""
    else:
        # No keyword hits: still may need human skim for objectively eligible sets
        # from CLIMB/TabArena that lack description — flag thin documentation.
        desc = (description or "").strip()
        if len(desc) < 40:
            review_status = "HUMAN_REVIEW_REQUIRED"
            suspected = suspected or "J_unclear_provenance"
            evidence = evidence or "description_too_short_or_missing"
            proposed_decision = "KEEP"
            decision = ""
            decision_reason = ""
        else:
            # TabArena curated IID pool with adequate description and no flags:
            # propose KEEP with HIGH confidence metadata support (still recordable).
            if source_pool == "tabarena_v0.1":
                review_status = "PROPOSED_KEEP_METADATA_CLEAR"
                proposed_decision = "KEEP"
                decision = "KEEP"
                decision_reason = "tabarena_v0.1_curated_iid_no_semantic_keyword_hits"
            else:
                # CLIMB: paper asserts natural imbalance / real-world, but protocol
                # requires human judgment for semantic exclusions — flag for review
                # only if keywords hit. Otherwise propose KEEP from CLIMB selection
                # criteria documentation (non-LLM, paper-stated).
                review_status = "PROPOSED_KEEP_METADATA_CLEAR"
                proposed_decision = "KEEP"
                decision = "KEEP"
                decision_reason = "climb_official_list_no_semantic_keyword_hits"

    return {
        "resolved_openml_id": resolved_openml_id,
        "dataset_name": dataset_name,
        "source_pool": source_pool,
        "objective_eligible": objective_eligible,
        "suspected_issue": suspected,
        "evidence": evidence[:2000],
        "proposed_decision": proposed_decision,
        "decision": decision,
        "decision_reason": decision_reason,
        "review_status": review_status,
    }
