"""Condition-clustered diagnostics for the full LOGO MC-dropout analysis.

The pass-level predictions are clustered in 16 held-out grinding conditions.
This script reports both descriptive pass-level metrics and condition-level,
cluster-bootstrap, and leave-one-condition-out sensitivities without treating
the 319 passes as independent experimental units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "reports/evidence/uncertainty/mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv"
TABLES = ROOT / "reports/evidence/tables"
TABLES.mkdir(parents=True, exist_ok=True)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["abs_error"].to_numpy()
    uncertainty = frame["y_std"].to_numpy()
    threshold = np.quantile(error, 0.8)
    high_error = error >= threshold
    return {
        "pearson_r": float(pearsonr(uncertainty, error).statistic),
        "spearman_rho": float(spearmanr(uncertainty, error).statistic),
        "auroc_top20_error": float(roc_auc_score(high_error, uncertainty)),
    }


def main() -> None:
    df = pd.read_csv(INPUT)
    if "abs_error" not in df:
        df["abs_error"] = (df["y_true"] - df["y_mean"]).abs()

    pass_metrics = metrics(df)
    condition_means = df.groupby("condition_id", as_index=False)[["y_std", "abs_error"]].mean()
    condition_metrics = {
        "pearson_r": float(pearsonr(condition_means["y_std"], condition_means["abs_error"]).statistic),
        "spearman_rho": float(spearmanr(condition_means["y_std"], condition_means["abs_error"]).statistic),
    }

    rng = np.random.default_rng(42)
    conditions = df["condition_id"].unique()
    boot_rows: list[dict[str, float]] = []
    for _ in range(5_000):
        sampled = rng.choice(conditions, size=len(conditions), replace=True)
        sampled_frame = pd.concat([df[df["condition_id"] == cid] for cid in sampled], ignore_index=True)
        boot_rows.append(metrics(sampled_frame))
    boot = pd.DataFrame(boot_rows)

    loco_rows: list[dict[str, float]] = []
    for condition in conditions:
        row = {"held_out_condition": int(condition), **metrics(df[df["condition_id"] != condition])}
        loco_rows.append(row)
    loco = pd.DataFrame(loco_rows)

    summary_rows = []
    for name, value in pass_metrics.items():
        summary_rows.append({
            "metric": name,
            "estimate_type": "pass_level_descriptive",
            "estimate": value,
            "ci_low": np.nan,
            "ci_high": np.nan,
        })
        summary_rows.append({
            "metric": name,
            "estimate_type": "condition_cluster_bootstrap_95ci",
            "estimate": value,
            "ci_low": float(boot[name].quantile(0.025)),
            "ci_high": float(boot[name].quantile(0.975)),
        })
    for name, value in condition_metrics.items():
        summary_rows.append({
            "metric": name,
            "estimate_type": "condition_mean_n16",
            "estimate": value,
            "ci_low": np.nan,
            "ci_high": np.nan,
        })

    pd.DataFrame(summary_rows).to_csv(TABLES / "clustered_uncertainty_diagnostics.csv", index=False)
    loco.to_csv(TABLES / "clustered_uncertainty_loco.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nLOCO ranges")
    print(loco.drop(columns="held_out_condition").agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
