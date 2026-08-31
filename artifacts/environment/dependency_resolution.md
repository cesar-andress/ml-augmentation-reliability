# Dependency resolution (2026-08-31)

## Conflict

| Constraint | Source |
|---|---|
| `torch>=2.5` | `tabpfn==8.5.0` |
| `xgboost==3.4.1` (`requires_python>=3.12`) | protocol |
| `torch>=2.1,<2.3` | `synthcity==0.2.12` |
| `numpy<2.0` | `synthcity==0.2.12` |
| `xgboost<3.0.0` | `synthcity==0.2.12` |

These cannot coexist in one resolver graph.

## Decision

Minimum two project-local environments:

1. **`.venv_main`** (Python 3.12): learners, preprocessing, augmentation A0–A2, metrics, smoke orchestration.
2. **`.venv_synthcity`** (Python 3.11): TabDDPM via SynthCity only.

## File-based interface (A3)

Main env writes TRAIN features/labels + config JSON to `artifacts/smoke/a3_jobs/<job_id>/`.
SynthCity env reads that payload, fits TabDDPM, writes synthetic minority rows as parquet/npy.
Main env loads synthetic rows, applies shared repair, continues the pipeline.

## Additional pins (synthcity env)

- `opacus==1.4.0`: transitive `opacus>=1.5` requires `torch.nn.RMSNorm` (torch≥2.4), incompatible with synthcity's `torch<2.3`. Scientific package `synthcity==0.2.12` unchanged.

## TabDDPM plugin name

In `synthcity==0.2.12`, TabDDPM is registered as plugin `"ddpm"` (`TabDDPMPlugin`). Worker uses this library identity; algorithm is not substituted.
