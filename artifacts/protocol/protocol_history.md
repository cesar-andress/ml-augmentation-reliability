# Protocol history — Augmentation Reliability Study

All timestamps UTC.

## 2026-08-31T20:00:00Z — Protocol v1.1 (original confirmatory design)

- Minimum cohort size **D ≥ 15**
- Candidate universe: **TabArena v0.1** ∪ **CLIMB**
- Minority prevalence range: **[0.02, 0.40]**
- Split: stratified 5-fold × 2 repeats; TRAIN/CAL-A/CAL-B/TEST as frozen
- Learners: XGBoost, CatBoost, TabPFN v2, TabICL v1
- Arms: A0, A1, A2, A3 (+ A0+ for GBDTs)
- No confirmatory model had been fitted at issuance of later amendments

Canonical snapshot: `artifacts/protocol/protocol_v1_1.yaml`

## 2026-08-31T21:08:00Z — Human semantic decision (still v1.1 screening)

- OpenML **44232** (`UCI_churn`) **EXCLUDE**
- Reason: probable subset/variant of retained classic telecom churn (**46915**, TabArena)
- Recorded to preserve dataset-level independence

## 2026-08-31T21:14:00Z — Pre-authorized prevalence amendment attempted (v1.1)

- Upper bound temporarily **0.40 → 0.50** (sole authorized rule change)
- Qualifying count after human EXCLUDE of 44232: **13** under 0.40
- After amendment: **14** (&lt; 15) → **DATASET_COHORT_NO_GO** under v1.1 stop rule
- OpenML **46930** (`hazelnut-spread-contaminant-detection`) became objectively eligible only under 0.50 (exactly balanced, p=0.5)

## 2026-08-31T21:30:00Z — Screening defect audit (outcome-blind; no confirmatory results)

- IDs **735, 833, 976, 1021** violated v1.1 semantic/artificial-binarization exclusion intent
  (OpenML descriptions contain “binarized version of the original data set” and/or `binaryClass` target)
- OpenML **46930** is exactly balanced → augmentation target **k = N_majority − N_minority = 0** (structural zero treatment); must not enter inferential cohort
- **No confirmatory model result had been computed** when Protocol v1.2 was issued

## 2026-08-31T21:37:00Z — Protocol v1.2 issued (outcome-blind)

- Expand candidate universe **exactly once** to include **OpenML-CC18** (suite 99)
- Universe = TabArena v0.1 ∪ CLIMB ∪ OpenML-CC18
- **No further expansion** permitted
- Restore hard prevalence upper bound **p ≤ 0.40**; reverse active use of 0.50 amendment (retained in history only)
- Require treatment intensity **r = (1 − 2p)/p ≥ 0.5** (equivalent to p ≤ 0.40)
- Corrected artificial-binarization exclusion (description regex + `binaryClass` target)
- Minimum cohort size **D ≥ 10** (inferential-resolution / sign-flip floor margin; not a formal power calculation)
- Confirmatory inference: dataset-level exact sign-flip tests (see `statistical_analysis_v1_2.yaml`)

Canonical snapshot: `artifacts/protocol/protocol_v1_2.yaml`
