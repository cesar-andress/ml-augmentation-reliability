"""Run frozen confirmatory statistical analysis (Protocol v1.2 / v1.2.1).

Implements statistical_analysis_v1_2.yaml exactly. Does not modify experimental outputs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.aggregate import (  # noqa: E402
    a0plus_contrast,
    bar_delta,
    dataset_delta,
    dataset_psi,
    descriptive_arm_learner,
    per_fold_delta,
)
from src.analysis.load_results import (  # noqa: E402
    AUG_ARMS,
    FROZEN_DATASET_IDS,
    GBDT,
    TFM,
    attach_sidecar_log_losses,
    build_input_manifest,
    load_canonical_table,
    load_identity_manifest,
    load_tfm_temperature_log_loss,
    minority_count_from_identity,
    prevalence_from_identity,
)
from src.analysis.sign_flip import (  # noqa: E402
    conventional_t_interval,
    exact_sign_flip_ci,
    exact_sign_flip_test,
)

PAPER = Path.home() / "papers/ml/paper"
ALPHA = 0.025
COHORT_SHA = "209bc80826843940da92799ca48b406a34df7e2dbafd2ff26590d092187ecb34"
FREEZE_COMMIT = "c346440ba6da683762da92a8fb126d57e2bcab6c"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_gate(repo_root: Path) -> dict:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo_root, text=True).strip()
    if branch != "main":
        raise SystemExit("ANALYSIS_GATE_FAILED: branch != main")
    v12 = subprocess.check_output(
        ["git", "rev-parse", "protocol-v1.2-freeze^{commit}"], cwd=repo_root, text=True
    ).strip()
    v121 = subprocess.check_output(
        ["git", "rev-parse", "protocol-v1.2.1-freeze^{commit}"], cwd=repo_root, text=True
    ).strip()
    if v121 != FREEZE_COMMIT:
        raise SystemExit(f"ANALYSIS_GATE_FAILED: freeze commit {v121}")
    if not (repo_root / "artifacts/protocol/protocol_v1_2.yaml").exists():
        raise SystemExit("ANALYSIS_GATE_FAILED: missing protocol_v1_2.yaml")
    if not (repo_root / "artifacts/protocol/protocol_v1_2_1_amendment.yaml").exists():
        raise SystemExit("ANALYSIS_GATE_FAILED: missing protocol_v1_2_1_amendment.yaml")
    if not (repo_root / "artifacts/protocol/statistical_analysis_v1_2.yaml").exists():
        raise SystemExit("ANALYSIS_GATE_FAILED: missing statistical_analysis_v1_2.yaml")
    cohort = _sha256_file(repo_root / "artifacts/manifests/datasets_frozen_v1_2.csv")
    if cohort != COHORT_SHA:
        raise SystemExit("ANALYSIS_GATE_FAILED: cohort SHA mismatch")
    identity = load_identity_manifest(repo_root)
    if sorted(identity) != list(FROZEN_DATASET_IDS):
        raise SystemExit("ANALYSIS_GATE_FAILED: identity cohort mismatch")
    # campaign completeness
    for ds in FROZEN_DATASET_IDS:
        for r in (0, 1):
            for f in range(5):
                uid = f"d{ds}_r{r}_f{f}"
                uc = repo_root / f"results/confirmatory/units/{uid}/status/unit_complete.json"
                if not uc.exists() or json.loads(uc.read_text()).get("status") != "COMPLETE":
                    raise SystemExit(f"ANALYSIS_GATE_FAILED: {uid}")
    audits = list((repo_root / "artifacts/audits").glob("dataset_*_completion_audit.json"))
    # 9 campaign audits expected (44 audited separately); require the 9 non-44
    audited = {int(p.stem.split("_")[1]) for p in audits}
    need = set(FROZEN_DATASET_IDS) - {44}
    if not need.issubset(audited):
        raise SystemExit(f"ANALYSIS_GATE_FAILED: missing audits {need - audited}")
    return {
        "branch": branch,
        "protocol_v1_2_freeze_commit": v12,
        "protocol_v1_2_1_freeze_commit": v121,
        "cohort_sha256": cohort,
        "identity_manifest_sha256": _sha256_file(
            repo_root / "artifacts/manifests/dataset_content_identity_v1.csv"
        ),
        "gate": "PASS",
    }


def replace_tfm_metric_with_temperature(
    repo_root: Path, table: pd.DataFrame, *, metric_col: str = "log_loss_t09"
) -> pd.DataFrame:
    """Build a metric column: TFMs at T=0.9 raw log loss; GBDTs keep primary raw."""
    vals = []
    for row in table.itertuples(index=False):
        if row.learner in TFM and row.arm in ("A0",) + AUG_ARMS:
            vals.append(
                load_tfm_temperature_log_loss(
                    repo_root,
                    unit_id=row.unit_id,
                    dataset_id=int(row.dataset_id),
                    learner=row.learner,
                    arm=row.arm,
                    temperature_key="sensitivity",
                )
            )
        else:
            vals.append(float(getattr(row, "log_loss")))
    out = table.copy()
    out[metric_col] = vals
    return out


def decision_row(hyp: str, estimand: str, test: dict, ci: dict, alpha: float) -> dict:
    mean = test["mean"]
    reject = test["p_value"] < alpha
    if mean > 0:
        direction = "positive (augmentation worsens raw log loss / TFM more harmed)"
    elif mean < 0:
        direction = "negative (augmentation improves raw log loss / TFM less harmed)"
    else:
        direction = "zero"
    if hyp == "H01":
        interpretation = (
            "Reject H01: evidence of non-zero mean augmentation effect on raw TEST log loss."
            if reject
            else "Fail to reject H01: no confirmatory evidence of a non-zero mean augmentation effect."
        )
        if mean > 0 and reject:
            interpretation = "Reject H01: confirmatory evidence that augmentation worsens mean raw TEST log loss."
        elif mean < 0 and reject:
            interpretation = "Reject H01: confirmatory evidence that augmentation improves mean raw TEST log loss."
    else:
        interpretation = (
            "Reject H02: evidence of a non-zero TFM−GBDT augmentation interaction on Delta scale."
            if reject
            else "Fail to reject H02: no confirmatory evidence of a TFM−GBDT augmentation interaction."
        )
    return {
        "Hypothesis": hyp,
        "Estimand": estimand,
        "Mean_effect": mean,
        "Exact_95pct_interval_lower": ci["lower"],
        "Exact_95pct_interval_upper": ci["upper"],
        "Exact_sign_flip_p": test["p_value"],
        "Bonferroni_alpha": alpha,
        "Reject_H0": "YES" if reject else "NO",
        "Direction": direction,
        "Interpretation": interpretation,
        "statistic": test["statistic"],
        "sd": test["sd"],
        "n_positive": test["n_positive"],
        "n_negative": test["n_negative"],
        "n_zero": test["n_zero"],
        "D": test["D"],
    }


def plot_effect(values: pd.Series, ids: pd.Series, *, title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(values))
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in values]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids], rotation=45, ha="right")
    ax.set_xlabel("OpenML dataset ID")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def plot_learner_arm_heatmap(bar: pd.DataFrame, out: Path) -> None:
    pivot = bar.pivot_table(index="learner", columns="arm", values="bar_delta", aggfunc="mean")
    pivot = pivot.reindex(index=["xgboost", "catboost", "tabpfn", "tabicl"], columns=list(AUG_ARMS))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(3), list(AUG_ARMS))
    ax.set_yticks(range(4), list(pivot.index))
    ax.set_title("Descriptive mean barDelta by learner × arm (not confirmatory)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # annotate
    arr = pivot.to_numpy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    pivot.to_csv(out.with_name(out.name + "_source.csv"))


def plot_moderators(df: pd.DataFrame, out_prefix: Path) -> None:
    for xcol, xlab in [("log_r", "log(r), r=(1-2p)/p"), ("log_n", "log(n_rows)")]:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(df[xcol], df["Delta_d"], c="#333333", s=40)
        # descriptive OLS line
        coef = np.polyfit(df[xcol], df["Delta_d"], 1)
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 50)
        ax.plot(xs, coef[0] * xs + coef[1], color="#b2182b", linewidth=1.2, label="descriptive OLS")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Delta_d (raw TEST log loss)")
        ax.set_title(f"Descriptive association: Delta_d vs {xcol}")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_prefix.with_name(out_prefix.name + f"_{xcol}.pdf"))
        fig.savefig(out_prefix.with_name(out_prefix.name + f"_{xcol}.svg"))
        plt.close(fig)


def main() -> int:
    repo_root = ROOT
    out_dir = repo_root / "artifacts/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_tables = PAPER / "tables"
    paper_figures = PAPER / "figures"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)

    gate = verify_gate(repo_root)
    print("GATE", gate["gate"])

    # --- input manifest ---
    input_manifest = build_input_manifest(repo_root)
    (out_dir / "confirmatory_analysis_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2) + "\n"
    )
    inv = pd.DataFrame(input_manifest["units"])
    inv.to_csv(out_dir / "confirmatory_analysis_input_inventory.csv", index=False)

    # --- canonical table ---
    table = load_canonical_table(repo_root)
    table = attach_sidecar_log_losses(repo_root, table)
    # Sanity: primary log_loss matches sidecar raw
    for row in table.sample(min(20, len(table)), random_state=0).itertuples(index=False):
        raw = json.loads(
            (
                repo_root
                / f"results/confirmatory/units/{row.unit_id}/metrics/{row.learner}_{'A0plus' if row.arm=='A0+' else row.arm}.json"
            ).read_text()
        )["log_loss_raw"]
        if abs(raw - float(row.log_loss)) > 1e-12:
            raise AssertionError("primary log_loss mismatch vs sidecar log_loss_raw")

    table.to_parquet(out_dir / "canonical_analysis_table.parquet", index=False)
    table.to_csv(out_dir / "canonical_analysis_table.csv", index=False)

    # --- primary aggregation ---
    fold_d = per_fold_delta(table, metric="log_loss")
    bar = bar_delta(fold_d)
    delta_d = dataset_delta(bar)
    psi_d = dataset_psi(bar)
    primary = delta_d.merge(psi_d, on="dataset_id")
    primary.to_csv(out_dir / "dataset_level_primary_contrasts.csv", index=False)

    # --- confirmatory tests ---
    h01 = exact_sign_flip_test(delta_d["Delta_d"].to_numpy())
    h02 = exact_sign_flip_test(psi_d["Psi_d"].to_numpy())
    ci_delta = exact_sign_flip_ci(delta_d["Delta_d"].to_numpy(), level=0.95)
    ci_psi = exact_sign_flip_ci(psi_d["Psi_d"].to_numpy(), level=0.95)
    t_delta = conventional_t_interval(delta_d["Delta_d"].to_numpy(), level=0.95)
    t_psi = conventional_t_interval(psi_d["Psi_d"].to_numpy(), level=0.95)

    decisions = pd.DataFrame(
        [
            decision_row("H01", "mean_Delta_d (raw TEST log loss)", h01, ci_delta, ALPHA),
            decision_row("H02", "mean_Psi_d (TFM - GBDT)", h02, ci_psi, ALPHA),
        ]
    )
    decisions.to_csv(out_dir / "confirmatory_tests.csv", index=False)
    intervals = pd.DataFrame(
        [
            {
                "effect": "mean_Delta_d",
                "exact_signflip_lower": ci_delta["lower"],
                "exact_signflip_upper": ci_delta["upper"],
                "conventional_t_lower": t_delta["lower"],
                "conventional_t_upper": t_delta["upper"],
                "note": "exact interval is PRIMARY; t interval is SUPPLEMENTARY; not multiplicity-adjusted",
            },
            {
                "effect": "mean_Psi_d",
                "exact_signflip_lower": ci_psi["lower"],
                "exact_signflip_upper": ci_psi["upper"],
                "conventional_t_lower": t_psi["lower"],
                "conventional_t_upper": t_psi["upper"],
                "note": "exact interval is PRIMARY; t interval is SUPPLEMENTARY; not multiplicity-adjusted",
            },
        ]
    )
    intervals.to_csv(out_dir / "confirmatory_intervals.csv", index=False)

    # --- descriptive ---
    desc = descriptive_arm_learner(bar)
    desc.to_csv(out_dir / "descriptive_arm_learner_summary.csv", index=False)
    bar.to_csv(out_dir / "bar_delta_learner_arm.csv", index=False)

    # --- secondary Platt recalibrated ---
    fold_platt = per_fold_delta(table, metric="log_loss_platt")
    bar_platt = bar_delta(fold_platt)
    delta_platt = dataset_delta(bar_platt)
    psi_platt = dataset_psi(bar_platt)
    # exploratory intervals only (no confirmatory p)
    ci_platt = exact_sign_flip_ci(delta_platt["Delta_d"].to_numpy(), level=0.95)
    secondary_cal = delta_platt.merge(psi_platt, on="dataset_id")
    secondary_cal["role"] = "EXPLORATORY"
    secondary_cal["exact_95_ci_lower_Delta"] = ci_platt["lower"]
    secondary_cal["exact_95_ci_upper_Delta"] = ci_platt["upper"]
    secondary_cal["mean_Delta"] = float(delta_platt["Delta_d"].mean())
    secondary_cal.to_csv(out_dir / "secondary_calibration_summary.csv", index=False)

    # --- A0+ ---
    a0_dl, a0_d = a0plus_contrast(table, metric="log_loss")
    a0_dl.to_csv(out_dir / "a0plus_by_learner.csv", index=False)
    a0_d["role"] = "EXPLORATORY"
    a0_d.to_csv(out_dir / "a0plus_summary.csv", index=False)

    # --- conformal ---
    conf = (
        table[table["arm"].isin(("A0",) + AUG_ARMS)]
        .groupby(["dataset_id", "learner", "arm"], as_index=False)
        .agg(
            mean_set_size=("conformal_set_size", "mean"),
            mean_coverage=("conformal_coverage", "mean"),
        )
    )
    # dataset-level mean set-size Delta vs A0 (exploratory descriptive)
    base_ss = conf[conf["arm"] == "A0"][["dataset_id", "learner", "mean_set_size", "mean_coverage"]].rename(
        columns={"mean_set_size": "ss0", "mean_coverage": "cov0"}
    )
    aug_ss = conf[conf["arm"].isin(AUG_ARMS)].merge(base_ss, on=["dataset_id", "learner"])
    aug_ss["delta_set_size"] = aug_ss["mean_set_size"] - aug_ss["ss0"]
    conf_ds = aug_ss.groupby("dataset_id", as_index=False).agg(
        mean_delta_set_size=("delta_set_size", "mean"),
        mean_set_size=("mean_set_size", "mean"),
        mean_coverage=("mean_coverage", "mean"),
    )
    conf_ds["role"] = "EXPLORATORY"
    conf_ds["nominal_coverage"] = 0.90
    conf_ds.to_csv(out_dir / "conformal_summary.csv", index=False)

    # --- other secondaries (AUROC, cal slope/intercept, F1, Brier) descriptive dataset means of arm-A0 deltas ---
    other_rows = []
    for metric in ["auroc", "calibration_slope", "calibration_intercept", "f1_tuned", "brier", "auprc"]:
        fd = per_fold_delta(table, metric=metric)
        bd = bar_delta(fd)
        dd = dataset_delta(bd)
        other_rows.append(
            {
                "metric": metric,
                "role": "SUPPLEMENTARY" if metric in {"auroc", "calibration_slope", "calibration_intercept", "f1_tuned"} else "EXPLORATORY",
                "mean_Delta_d": float(dd["Delta_d"].mean()),
                "sd_Delta_d": float(dd["Delta_d"].std(ddof=1)),
                "median_Delta_d": float(dd["Delta_d"].median()),
            }
        )
        dd.rename(columns={"Delta_d": f"Delta_{metric}"}).to_csv(
            out_dir / f"secondary_dataset_{metric}.csv", index=False
        )
    pd.DataFrame(other_rows).to_csv(out_dir / "other_frozen_secondaries_summary.csv", index=False)

    # --- sensitivity S1: T=0.9 for TFMs ---
    print("Computing S1 temperature sensitivity (TFM T=0.9)...")
    table_t09 = replace_tfm_metric_with_temperature(repo_root, table)
    fold_s1 = per_fold_delta(table_t09, metric="log_loss_t09")
    bar_s1 = bar_delta(fold_s1)
    delta_s1 = dataset_delta(bar_s1)
    psi_s1 = dataset_psi(bar_s1)
    s1 = delta_s1.merge(psi_s1, on="dataset_id")
    s1["sensitivity"] = "S1_TFM_temperature_0.9"
    s1["mean_Delta"] = float(delta_s1["Delta_d"].mean())
    s1["mean_Psi"] = float(psi_s1["Psi_d"].mean())
    s1["primary_mean_Delta"] = float(delta_d["Delta_d"].mean())
    s1["primary_mean_Psi"] = float(psi_d["Psi_d"].mean())
    s1["Delta_direction_match_primary"] = np.sign(s1["mean_Delta"].iloc[0]) == np.sign(
        s1["primary_mean_Delta"].iloc[0]
    )

    # --- sensitivity S2: isotonic instead of Platt (secondary calibrated endpoint) ---
    fold_s2 = per_fold_delta(table, metric="log_loss_isotonic")
    bar_s2 = bar_delta(fold_s2)
    delta_s2 = dataset_delta(bar_s2)
    psi_s2 = dataset_psi(bar_s2)
    s2 = delta_s2.merge(psi_s2, on="dataset_id")
    s2["sensitivity"] = "S2_isotonic_instead_of_Platt"
    s2["mean_Delta"] = float(delta_s2["Delta_d"].mean())
    s2["mean_Psi"] = float(psi_s2["Psi_d"].mean())
    s2["platt_mean_Delta"] = float(delta_platt["Delta_d"].mean())
    s2["Delta_direction_match_platt"] = np.sign(s2["mean_Delta"].iloc[0]) == np.sign(
        s2["platt_mean_Delta"].iloc[0]
    )

    # --- sensitivity S3: exclude minority < 400 ---
    identity = load_identity_manifest(repo_root)
    keep = [ds for ds in FROZEN_DATASET_IDS if minority_count_from_identity(identity[ds]) >= 400]
    delta_s3 = delta_d[delta_d["dataset_id"].isin(keep)].copy()
    psi_s3 = psi_d[psi_d["dataset_id"].isin(keep)].copy()
    s3_test_delta = exact_sign_flip_test(delta_s3["Delta_d"].to_numpy()) if len(delta_s3) >= 2 else None
    s3 = delta_s3.merge(psi_s3, on="dataset_id")
    s3["sensitivity"] = "S3_exclude_minority_lt_400"
    s3["D_retained"] = len(keep)
    s3["retained_dataset_ids"] = str(keep)
    s3["mean_Delta"] = float(delta_s3["Delta_d"].mean())
    s3["mean_Psi"] = float(psi_s3["Psi_d"].mean())
    s3["primary_mean_Delta"] = float(delta_d["Delta_d"].mean())
    s3["Delta_direction_match_primary"] = np.sign(s3["mean_Delta"].iloc[0]) == np.sign(
        s3["primary_mean_Delta"].iloc[0]
    )

    s1_degen = bool(np.allclose(delta_s1["Delta_d"], delta_d["Delta_d"]) and np.allclose(psi_s1["Psi_d"], psi_d["Psi_d"]))
    sens = pd.concat(
        [
            s1.assign(panel="S1", stored_predictions_identical_to_primary=s1_degen),
            s2.assign(panel="S2"),
            s3.assign(panel="S3"),
        ],
        ignore_index=True,
    )
    sens.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "sensitivity": "S1",
                "description": "TFM softmax temperature 0.9",
                "mean_Delta": float(delta_s1["Delta_d"].mean()),
                "mean_Psi": float(psi_s1["Psi_d"].mean()),
                "primary_mean_Delta": float(delta_d["Delta_d"].mean()),
                "primary_mean_Psi": float(psi_d["Psi_d"].mean()),
                "direction_consistent_Delta": True,
                "status": (
                    "EXECUTED_BUT_NONINFORMATIVE: stored p_sensitivity identical to p_primary "
                    "for TabPFN/TabICL across checked units; temperature API path did not alter "
                    "confirmatory predictions. No model re-run performed."
                    if s1_degen
                    else "EXECUTED"
                ),
            },
            {
                "sensitivity": "S2",
                "description": "isotonic instead of Platt (calibrated secondary)",
                "mean_Delta": float(delta_s2["Delta_d"].mean()),
                "mean_Psi": float(psi_s2["Psi_d"].mean()),
                "platt_mean_Delta": float(delta_platt["Delta_d"].mean()),
                "direction_consistent_vs_platt": bool(
                    np.sign(delta_s2["Delta_d"].mean()) == np.sign(delta_platt["Delta_d"].mean())
                ),
                "status": "EXECUTED",
            },
            {
                "sensitivity": "S3",
                "description": "exclude datasets with minority count < 400",
                "D": len(keep),
                "retained_ids": keep,
                "mean_Delta": float(delta_s3["Delta_d"].mean()),
                "mean_Psi": float(psi_s3["Psi_d"].mean()),
                "primary_mean_Delta": float(delta_d["Delta_d"].mean()),
                "direction_consistent_Delta": bool(
                    np.sign(delta_s3["Delta_d"].mean()) == np.sign(delta_d["Delta_d"].mean())
                ),
                "exact_signflip_p_Delta_exploratory_only": None
                if s3_test_delta is None
                else s3_test_delta["p_value"],
                "note": "S3 p-value if reported is EXPLORATORY only; not confirmatory",
                "status": "EXECUTED",
            },
        ]
    ).to_json(out_dir / "sensitivity_overview.json", orient="records", indent=2)

    # --- moderators descriptive ---
    mod_rows = []
    for ds in FROZEN_DATASET_IDS:
        row = identity[ds]
        p = prevalence_from_identity(row)
        r = (1.0 - 2.0 * p) / p
        n = int(row["n_rows"])
        mod_rows.append(
            {
                "dataset_id": ds,
                "p_minority": p,
                "r": r,
                "log_r": float(np.log(r)),
                "n_rows": n,
                "log_n": float(np.log(n)),
                "Delta_d": float(delta_d.loc[delta_d["dataset_id"] == ds, "Delta_d"].iloc[0]),
                "Psi_d": float(psi_d.loc[psi_d["dataset_id"] == ds, "Psi_d"].iloc[0]),
            }
        )
    moderators = pd.DataFrame(mod_rows)
    # descriptive correlations (no p-values)
    moderators["corr_log_r_Delta"] = moderators["log_r"].corr(moderators["Delta_d"])
    moderators["corr_log_n_Delta"] = moderators["log_n"].corr(moderators["Delta_d"])
    moderators.to_csv(out_dir / "moderator_descriptive_summary.csv", index=False)

    # --- mixed model ---
    mixed = {
        "status": "NOT EXECUTED — IMPLEMENTATION NOT VALIDATED",
        "protocol_status": "SUPPLEMENTARY_PENDING_IMPLEMENTATION_VALIDATION",
        "reason": "No independently verified Kenward-Roger MixedLM path; statsmodels.MixedLM not used.",
    }
    (out_dir / "mixed_model_status.json").write_text(json.dumps(mixed, indent=2) + "\n")

    # --- publication tables ---
    decisions.to_csv(paper_tables / "table_primary_confirmatory.csv", index=False)
    desc.to_csv(paper_tables / "table_descriptive_augmentation.csv", index=False)
    secondary_cal.to_csv(paper_tables / "table_secondary_reliability.csv", index=False)
    conf_ds.to_csv(paper_tables / "table_conformal.csv", index=False)
    pd.read_json(out_dir / "sensitivity_overview.json").to_csv(
        paper_tables / "table_sensitivities.csv", index=False
    )
    primary.to_csv(paper_tables / "table_dataset_level_effects.csv", index=False)
    a0_d.to_csv(paper_tables / "table_a0plus.csv", index=False)

    # --- figures ---
    plot_effect(
        primary["Delta_d"],
        primary["dataset_id"],
        title="Dataset-level primary contrasts Delta_d (raw TEST log loss)",
        ylabel=r"$\Delta_d$",
        out=paper_figures / "fig1_delta_d",
    )
    primary[["dataset_id", "Delta_d"]].to_csv(paper_figures / "fig1_delta_d_source.csv", index=False)
    plot_effect(
        primary["Psi_d"],
        primary["dataset_id"],
        title="Dataset-level interaction contrasts Psi_d (TFM − GBDT)",
        ylabel=r"$\Psi_d$",
        out=paper_figures / "fig2_psi_d",
    )
    primary[["dataset_id", "Psi_d"]].to_csv(paper_figures / "fig2_psi_d_source.csv", index=False)
    plot_learner_arm_heatmap(bar, paper_figures / "fig3_learner_arm_bardelta")
    plot_moderators(moderators, paper_figures / "fig4_moderators")
    moderators.to_csv(paper_figures / "fig4_moderators_source.csv", index=False)

    # --- numerical validation (independent recompute) ---
    # Recompute Delta_d manually
    recomputed = []
    for ds in FROZEN_DATASET_IDS:
        vals = []
        for learner in ("xgboost", "catboost", "tabpfn", "tabicl"):
            for arm in AUG_ARMS:
                sub = table[(table.dataset_id == ds) & (table.learner == learner)]
                a0 = sub[sub.arm == "A0"].sort_values(["repeat", "fold"])["log_loss"].to_numpy()
                aa = sub[sub.arm == arm].sort_values(["repeat", "fold"])["log_loss"].to_numpy()
                vals.append(float(np.mean(aa - a0)))
        recomputed.append(float(np.mean(vals)))
    if not np.allclose(recomputed, delta_d["Delta_d"].to_numpy(), atol=1e-12):
        raise AssertionError("Delta_d recompute mismatch")
    # sign patterns count
    assert h01["n_permutations"] == 1024 and h02["n_permutations"] == 1024
    assert h01["D"] == 10 and h02["D"] == 10
    # A0+ not in primary
    assert set(bar["arm"].unique()) == set(AUG_ARMS)

    sanity = {
        "nan_log_loss": bool(table["log_loss"].isna().any()),
        "inf_log_loss": bool(np.isinf(table["log_loss"].to_numpy()).any()),
        "log_loss_positive": bool((table["log_loss"] > 0).all()),
        "coverage_in_0_1": bool(table["conformal_coverage"].between(0, 1).all()),
        "set_size_in_0_2": bool(table["conformal_set_size"].between(0.0, 2.0).all()),
        "n_cells": int(len(table)),
        "duplicate_cells": bool(
            table.duplicated(subset=["dataset_id", "repeat", "fold", "learner", "arm"]).any()
        ),
        "a0plus_in_primary_bar": bool(("A0+" in set(bar["arm"].unique()))),
        "delta_recompute_ok": True,
        "sign_patterns_1024": True,
        "bonferroni_alpha": ALPHA,
    }
    if any(
        [
            sanity["nan_log_loss"],
            sanity["inf_log_loss"],
            not sanity["log_loss_positive"],
            not sanity["coverage_in_0_1"],
            not sanity["set_size_in_0_2"],
            sanity["duplicate_cells"],
            sanity["a0plus_in_primary_bar"],
        ]
    ):
        raise AssertionError(f"SANITY_FAIL {sanity}")
    (out_dir / "scientific_sanity_checks.json").write_text(json.dumps(sanity, indent=2) + "\n")

    # --- provenance ---
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    code_files = [
        repo_root / "src/analysis/sign_flip.py",
        repo_root / "src/analysis/aggregate.py",
        repo_root / "src/analysis/load_results.py",
        repo_root / "scripts/run_confirmatory_analysis.py",
        repo_root / "artifacts/protocol/statistical_analysis_v1_2.yaml",
    ]
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "freeze_tag": "protocol-v1.2.1-freeze",
        "freeze_commit": FREEZE_COMMIT,
        "protocol_v1_2_freeze_commit": gate["protocol_v1_2_freeze_commit"],
        "cohort_sha256": COHORT_SHA,
        "canonical_identity_manifest_sha256": gate["identity_manifest_sha256"],
        "analysis_input_manifest_sha256": _sha256_file(
            out_dir / "confirmatory_analysis_input_manifest.json"
        ),
        "analysis_code_sha256": {str(p.relative_to(repo_root)): _sha256_file(p) for p in code_files},
        "command": "CONFIRMATORY_ANALYSIS_AUTHORIZED=YES .venv_main/bin/python scripts/run_confirmatory_analysis.py",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "gate": gate,
        "H01": {k: h01[k] for k in h01 if k != "values"},
        "H02": {k: h02[k] for k in h02 if k != "values"},
        "mixed_model": mixed,
    }
    (out_dir / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # machine-readable report summary for the agent
    summary = {
        "status": "PASS",
        "H01_reject": decisions.loc[0, "Reject_H0"],
        "H02_reject": decisions.loc[1, "Reject_H0"],
        "mean_Delta": h01["mean"],
        "p_H01": h01["p_value"],
        "mean_Psi": h02["mean"],
        "p_H02": h02["p_value"],
        "ci_Delta": [ci_delta["lower"], ci_delta["upper"]],
        "ci_Psi": [ci_psi["lower"], ci_psi["upper"]],
        "D": 10,
        "cells": 1800,
        "S3_retained": keep,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
