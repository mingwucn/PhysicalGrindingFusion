#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""Generate supplementary analyses requested by the scientific review.

Outputs:
  reports/evidence/tables/top_models_summary.csv
  reports/evidence/tables/top_models_per_condition_mae.csv
  reports/evidence/tables/condition7_sensitivity_top_pairs.csv
  reports/evidence/plots/results/top_models_fold_mae.png
  reports/evidence/plots/results/top_models_per_condition_mae.png
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import (
    FigureProfiles,
    MutableFigure,
    PublicationPlotter,
)

PublicationPlotter.set_style()

TABLES_DIR = ROOT / "reports" / "evidence" / "tables"
PLOTS_DIR = ROOT / "reports" / "evidence" / "plots" / "results"
OVERLEAF_DIR = ROOT / "overleaf" / "images"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_folds(s: str) -> list[dict]:
    return ast.literal_eval(s)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    boot = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def make_figure(name: str, profile):
    figure = MutableFigure(
        name,
        profile=profile,
        out_dir=PLOTS_DIR,
        overleaf_dir=OVERLEAF_DIR,
        metadata={"generator": "scripts/review_required_analyses.py"},
    )
    fig, axes = figure.create()
    return figure, fig, axes


def main() -> int:
    df = pd.read_csv(TABLES_DIR / "full_results_logo_only.csv")
    wst_path = TABLES_DIR / "wst_results.csv"
    if wst_path.exists():
        wst = pd.read_csv(wst_path)
        # Keep the same columns used below; folds are unavailable in wst_results
        # and are not needed for the controlled representation table/figure.
        keep = ["model", "config", "mae_mean", "mae_std"]
        df = pd.concat([df, wst[keep]], ignore_index=True, sort=False)
    has_folds = df["folds"].notna()
    df["folds_list"] = None
    df.loc[has_folds, "folds_list"] = df.loc[has_folds, "folds"].apply(parse_folds)
    df["mae_per_fold"] = None
    df.loc[has_folds, "mae_per_fold"] = df.loc[has_folds, "folds_list"].apply(
        lambda folds: np.array([f["mae"] for f in folds])
    )

    # ------------------------------------------------------------------
    # 1. Top 5 model summary statistics
    # ------------------------------------------------------------------
    df_folded = df[has_folds].copy()
    top5 = df_folded.nsmallest(5, "mae_mean").copy()
    summary_rows = []
    per_condition_rows = []

    for _, row in top5.iterrows():
        mae = row["mae_per_fold"]
        q25, q75 = np.quantile(mae, [0.25, 0.75])
        ci_lo, ci_hi = bootstrap_ci(mae)
        mae_excl_c7 = np.delete(mae, 6)  # fold index 6 corresponds to condition 7
        summary_rows.append({
            "model": row["model"],
            "config": row["config"],
            "mean_mae": row["mae_mean"],
            "std_mae": row["mae_std"],
            "median_mae": float(np.median(mae)),
            "iqr_mae": float(q75 - q25),
            "trim10_mae": float(stats.trim_mean(mae, 0.1)),
            "min_fold_mae": float(np.min(mae)),
            "max_fold_mae": float(np.max(mae)),
            "bootstrap_ci_lo": ci_lo,
            "bootstrap_ci_hi": ci_hi,
            "mean_excl_condition7": float(mae_excl_c7.mean()),
            "std_excl_condition7": float(mae_excl_c7.std()),
        })

        for fold_idx, mae_val in enumerate(mae):
            per_condition_rows.append({
                "model": row["model"],
                "config": row["config"],
                "condition": fold_idx + 1,
                "fold_mae": mae_val,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "top_models_summary.csv", index=False)
    pd.DataFrame(per_condition_rows).to_csv(TABLES_DIR / "top_models_per_condition_mae.csv", index=False)
    print("Saved top_models_summary.csv and top_models_per_condition_mae.csv")

    # ------------------------------------------------------------------
    # 2. Condition-7 sensitivity for key pairwise comparisons
    # ------------------------------------------------------------------
    pairs = [
        ("RandomForestModel", "ae_logspec+vib_logspec", "RandomForestModel", "ae_spec+vib_spec+pp"),
        ("RandomForestModel", "ae_logspec+vib_logspec", "ResNetVibCNN", "vib_spec"),
        ("RandomForestModel", "ae_logspec+vib_logspec", "LightGBMModel", "vib_logspec"),
    ]
    sens_rows = []
    for m1, c1, m2, c2 in pairs:
        r1 = df[(df["model"] == m1) & (df["config"] == c1)].iloc[0]
        r2 = df[(df["model"] == m2) & (df["config"] == c2)].iloc[0]
        maes1, maes2 = r1["mae_per_fold"], r2["mae_per_fold"]
        diffs = maes1 - maes2
        diffs_excl = np.delete(diffs, 6)
        sens_rows.append({
            "model_a": m1, "config_a": c1,
            "model_b": m2, "config_b": c2,
            "mean_diff_all": float(diffs.mean()),
            "mean_diff_excl_c7": float(diffs_excl.mean()),
            "median_diff_all": float(np.median(diffs)),
            "median_diff_excl_c7": float(np.median(diffs_excl)),
        })
    pd.DataFrame(sens_rows).to_csv(TABLES_DIR / "condition7_sensitivity_top_pairs.csv", index=False)
    print("Saved condition7_sensitivity_top_pairs.csv")

    # ------------------------------------------------------------------
    # 3. Figures
    # ------------------------------------------------------------------
    # Per-fold MAE for top 5
    publication_configs = {
        "ae_logspec+vib_logspec": "AE-dB-z + Vib-dB-z",
        "ae_mel+vib_mel": "AE-log-mel + Vib-log-mel",
        "ae_logspec+vib_logspec+pp": "AE-dB-z + Vib-dB-z + PP",
        "ae_mel+vib_mel+pp": "AE-log-mel + Vib-log-mel + PP",
        "ae_spec+vib_spec+pp": "AE-dB + Vib-dB + PP",
    }
    publication_models = {"RandomForestModel": "Random forest"}
    managed, fig, ax = make_figure("top_models_fold_mae.png", FigureProfiles.DOUBLE)
    x = np.arange(16) + 1
    for _, row in top5.iterrows():
        label = f"{publication_models.get(row['model'], row['model'])} / {publication_configs.get(row['config'], row['config'])}"
        ax.plot(x, row["mae_per_fold"], marker="o", ms=3, label=label)
    ax.axvline(7, color="gray", linestyle="--", lw=0.8, label="Condition 7")
    ax.set_xlabel("Condition (fold)")
    ax.set_ylabel("Fold MAE (µm)")
    ax.set_title("Per-fold MAE for the top five model-configuration pairs")
    ax.legend(fontsize=PublicationPlotter.LEGEND_SIZE, loc="upper left", frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    managed.save()

    # Per-condition MAE heatmap for top 5
    managed, fig, ax = make_figure("top_models_per_condition_mae.png", FigureProfiles.DOUBLE)
    heat = pd.DataFrame({
        f"{publication_models.get(r['model'], r['model'])}\n{publication_configs.get(r['config'], r['config'])}": r["mae_per_fold"]
        for _, r in top5.iterrows()
    }, index=np.arange(1, 17))
    im = ax.imshow(heat.T, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels(np.arange(1, 17))
    ax.set_yticks(np.arange(len(heat.columns)))
    ax.set_yticklabels(heat.columns, fontsize=5)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Model / Configuration")
    ax.set_title("Per-condition fold MAE (top five pairs)")
    fig.colorbar(im, ax=ax, label="MAE (µm)")
    fig.tight_layout()
    managed.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
