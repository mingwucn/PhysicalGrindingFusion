#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
Generate a methods figure illustrating MC-dropout uncertainty on real data.

Outputs:
    reports/evidence/plots/methods/methods_uncertainty_overview.png

The figure shows:
  (a) 95% prediction intervals for ResNetVibCNN on vib_spec
  (b) Reliability diagram: predicted std vs. observed absolute error
  (c) Coverage by uncertainty bin
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

UNC_DIR = ROOT / "reports" / "evidence" / "uncertainty"
OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()


def main() -> int:
    csv_path = UNC_DIR / "mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv"
    subset_path = UNC_DIR / "mc_dropout_ResNetVibCNN_vib_spec.csv"
    rel_path = UNC_DIR / "mc_dropout_reliability.csv"
    if not csv_path.exists():
        print("Full LOGO MC-dropout CSV not found. Run scripts/mc_dropout_full_logo.py first.")
        return 1
    if not rel_path.exists():
        print("Reliability CSV not found. Run scripts/reliability_diagram.py first.")
        return 1

    df = pd.read_csv(csv_path)
    rel = pd.read_csv(rel_path)

    # Normalise column names across old/new CSV layouts
    if "y_pred" in df.columns and "y_mean" not in df.columns:
        df["y_mean"] = df["y_pred"]
    for old, new in (("lower", "pi_lower_95"), ("upper", "pi_upper_95")):
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "abs_error" not in df.columns:
        df["abs_error"] = np.abs(df["y_mean"] - df["y_true"])
    if "covered" not in df.columns:
        df["covered"] = (df["y_true"] >= df["pi_lower_95"]) & (df["y_true"] <= df["pi_upper_95"])

    managed = MutableFigure(
        "methods_uncertainty_overview.png",
        profile=FigureProfiles.TOP_SPAN_TWO_BOTTOM,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/generate_methods_uncertainty_figure.py"},
    )
    fig, initial_axes = managed.create()
    for axis in np.asarray(initial_axes).flat:
        axis.remove()
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.52, wspace=0.30)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    # (a) Prediction intervals sorted by predicted mean
    order = np.argsort(df["y_mean"].to_numpy())
    y_true = df["y_true"].to_numpy()[order]
    y_mean = df["y_mean"].to_numpy()[order]
    lower = df["pi_lower_95"].to_numpy()[order]
    upper = df["pi_upper_95"].to_numpy()[order]
    x = np.arange(len(order))

    ax = axes[0]
    ax.fill_between(x, lower, upper, alpha=0.3, color=PublicationPalette.UNCERTAINTY, label="Nominal 95% MC-dropout epistemic band")
    ax.plot(x, y_mean, color=PublicationPalette.PREDICTED, lw=1.5, label="Predicted mean")
    ax.scatter(x, y_true, c=PublicationPalette.OBSERVED, s=15, zorder=3, label="Observed $R_a$")
    ax.set_xlabel("Sample index (sorted by prediction)")
    ax.set_ylabel("Surface roughness $R_a$ (µm)")
    ax.legend(loc="upper left")

    # (b) Reliability: predicted std vs observed error
    ax = axes[1]
    max_val = max(rel["bin_mean_std"].max(), rel["bin_mean_abs_error"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", label="MAE = predicted SD reference")
    ax.scatter(rel["bin_mean_std"], rel["bin_mean_abs_error"], s=120, color=PublicationPalette.UNCERTAINTY, edgecolor="black", zorder=3)
    ax.set_xlabel("Mean predicted std in bin (µm)")
    ax.set_ylabel("Mean observed absolute error (µm)")
    ax.legend(loc="upper left")

    # (c) Coverage by uncertainty bin
    ax = axes[2]
    ax.axhline(0.95, color="black", linestyle="--", label="Nominal 95%")
    ax.bar(range(len(rel)), rel["bin_coverage"], color=PublicationPalette.UNCERTAINTY, edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Uncertainty bin (low → high)")
    ax.set_ylabel("Empirical coverage")
    ax.legend(loc="lower right")

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.90)
    managed.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
