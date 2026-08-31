# ml — Augmentation reliability infrastructure

Research prototype for a frozen empirical protocol on tabular classification with
oversampling / generative augmentation, calibration, and conformal prediction.

**Prototype root:** `~/papers/ml/ml`  
**Paper directory:** `~/papers/ml/paper` (not used by this infrastructure task)

## Environments (two required)

`synthcity==0.2.12` requires `torch<2.3`, `numpy<2`, `xgboost<3`, which conflict with
`tabpfn==8.5.0` (`torch>=2.5`) and `xgboost==3.4.1`. Resolution:

| Env | Python | Role |
|---|---|---|
| `.venv_main` | 3.12 | Learners, A0–A2, metrics, orchestration |
| `.venv_synthcity` | 3.11 | TabDDPM (A3) only; file-based IPC |

See `artifacts/environment/dependency_resolution.md`.

### Create environments

```bash
cd ~/papers/ml/ml

# Main scientific stack
uv venv .venv_main --python 3.12
uv pip install --python .venv_main/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv_main/bin/python -r requirements-main.txt

# Isolated SynthCity / TabDDPM
uv venv .venv_synthcity --python 3.11
uv pip install --python .venv_synthcity/bin/python "torch>=2.1,<2.3" --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv_synthcity/bin/python -r requirements-synthcity.txt
```

Checkpoints (exact identities):

```bash
cd ~/papers/ml/ml
.venv_main/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt", local_dir="checkpoints/tabpfn")
hf_hub_download("jingang/TabICL", "tabicl-classifier-v1-20250208.ckpt", local_dir="checkpoints/tabicl")
PY
```

## Hardware audit

```bash
cd ~/papers/ml/ml
.venv_main/bin/python scripts/audit_hardware.py
# writes artifacts/environment/hardware.json
```

## GPU compatibility

```bash
cd ~/papers/ml/ml
.venv_main/bin/python scripts/gpu_compatibility.py
# writes artifacts/environment/gpu_compatibility.json
```

## Dataset screening (Stage A + Stage B manifest)

Curated candidate IDs only — does **not** download the full TabArena suite.

```bash
cd ~/papers/ml/ml
.venv_main/bin/python scripts/screen_datasets.py
# writes:
#   artifacts/manifests/dataset_candidates.csv
#   artifacts/manifests/dataset_semantic_review.csv
```

`decision` / `decision_reason` stay blank when human review is required.

## Unit tests

```bash
cd ~/papers/ml/ml
.venv_main/bin/python -m pytest tests/ -q
```

## End-to-end smoke test

One OpenML dataset, one outer fold, 4 learners × A0/A1/A2, plus XGBoost×A3
(tiny TabDDPM config marked `SMOKE_ONLY_NOT_SCIENTIFIC`).

```bash
cd ~/papers/ml/ml
.venv_main/bin/python scripts/run_smoke.py
# writes results/raw/smoke_results.parquet
```

## Scientific marks

- `SMOKE_ONLY_FIXED_HPARAMS` — XGBoost/CatBoost fixed configs (no 20-config HPO)
- `SMOKE_ONLY_NOT_SCIENTIFIC` — A3 TabDDPM tiny generator settings

## Stop conditions (intentionally not done)

- Full TabArena / CLIMB download campaign
- All 10 folds, HPO, full TabDDPM campaign
- Mixed-effects analysis
- Paper writing under `~/papers/ml/paper`
# ml-augmentation-reliability
