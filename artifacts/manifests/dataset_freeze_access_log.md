# Dataset freeze access log

**Created:** 2026-08-31  
**Phase:** confirmatory dataset cohort freeze  
**Project root:** `~/papers/ml/ml`

## Allowed inputs for dataset selection

Dataset selection for this freeze uses **only**:

- OpenML metadata (IDs, versions, names, descriptions, qualities, lineage)
- TabArena v0.1 metadata (`artifacts/manifests/sources/tabarena_dataset_metadata.csv`, suite 457)
- CLIMB official dataset metadata (`artifacts/manifests/sources/climb_openml_datainfo.csv`)
- Raw OpenML features and targets (no CLIMB `.npz` arrays)
- Missingness patterns and feature-type observations from raw tables
- Duplication / provenance information from metadata and data fingerprints

## Forbidden inputs (never used during selection)

- Model predictions
- Model losses / log loss
- AUROC, AUPRC, Brier
- Calibration slope/intercept
- Conformal metrics
- HPO results
- Previous smoke performance or any file under `results/smoke/` metric columns
- Any confirmatory learner outputs

## Smoke isolation

Smoke results live under `results/smoke/`. Confirmatory outputs must live under `results/confirmatory/`. Confirmatory readers reject rows unless `scientific_status == "CONFIRMATORY"`.

## Operator attestation

This freeze run does not open or parse smoke metric columns for selection decisions.
