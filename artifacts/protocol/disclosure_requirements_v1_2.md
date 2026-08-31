# Disclosure requirements — Protocol v1.2

Issued: 2026-08-31T21:37:00Z (UTC)

The final manuscript **must** disclose all of the following (Methods, Limitations, and repository protocol history):

1. Original **D ≥ 15** rule under Protocol v1.1.
2. Screening defect affecting OpenML IDs **735, 833, 976, 1021**.
3. Why those four were removed: artificial binarization (description match and/or `binaryClass` target) under the corrected deterministic exclusion rule.
4. Exclusion of OpenML **46930** because **k = 0** (structural zero treatment at prevalence 0.5); retained only for deferred pipeline-determinism assertion.
5. Reversal of the temporary **0.50** prevalence amendment; active bound restored to **p ≤ 0.40** (with **r ≥ 0.5**).
6. One-time expansion of the candidate universe to **OpenML-CC18**.
7. Revised **D ≥ 10** rule and its derivation (sign-flip inferential-resolution margin; not a formal power calculation).
8. Outcome-blind timing: **no confirmatory model result** had been computed when v1.2 was issued.
9. Final cohort size **D**.
10. **No further candidate-source expansion** is permitted; if D &lt; 10 after v1.2, outcome is permanent `DATASET_COHORT_FINAL_NO_GO`.
