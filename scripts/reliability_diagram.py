# Output mapping: see docs/figure_script_toc.md
"""Reliability diagram for MC-dropout / ensemble uncertainties."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

PublicationPlotter.set_style()

OUT_DIR = Path("reports/evidence/uncertainty")
SUBSET_CSV = OUT_DIR / "mc_dropout_ResNetVibCNN_vib_spec.csv"
FULL_CSV = OUT_DIR / "mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv"
N_BINS = 5


def load_df():
    """Prefer full 16-fold LOGO file; fall back to the original 59-sample subset."""
    if FULL_CSV.exists():
        df = pd.read_csv(FULL_CSV)
    else:
        df = pd.read_csv(SUBSET_CSV)
    # Normalise column names across old/new CSV layouts
    if "y_pred" in df.columns and "y_mean" not in df.columns:
        df["y_mean"] = df["y_pred"]
    for old, new in (("lower", "pi_lower_95"), ("upper", "pi_upper_95")):
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "abs_error" not in df.columns:
        df["abs_error"] = np.abs(df["y_true"] - df["y_mean"])
    if "covered" not in df.columns:
        df["covered"] = (df["y_true"] >= df["pi_lower_95"]) & (df["y_true"] <= df["pi_upper_95"])
    return df


def main():
    df = load_df()

    # Expected absolute error quantified as z*std for a 95% Gaussian interval
    df["expected_abs"] = df["y_std"]  # use std directly
    df.sort_values("expected_abs", inplace=True)
    # Create N_BINS nearly equal-sized bins by predicted uncertainty quantiles
    df["bin"] = pd.qcut(df["expected_abs"], q=N_BINS, labels=False, duplicates="drop")
    rows = []
    for b in sorted(df["bin"].unique()):
        sub = df[df["bin"] == b]
        rows.append({
            "bin_mean_std": sub["y_std"].mean(),
            "bin_mean_abs_error": sub["abs_error"].mean(),
            "bin_coverage": sub["covered"].mean(),
            "n": len(sub),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "mc_dropout_reliability.csv", index=False)

    # Coverage compares absolute error with the 95% interval half-width.
    table["bin_mean_half_width"] = table["bin_mean_std"] * 1.96

    managed = MutableFigure(
        "mc_dropout_reliability.png",
        profile=FigureProfiles.TWO_PANEL_ROW,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/reliability_diagram.py"},
    )
    fig, axes = managed.create()

    # Left: coverage per uncertainty bin
    ax = axes[0]
    ax.axhline(0.95, color="k", linestyle="--", label="nominal 95%")
    ax.bar(
        range(len(table)),
        table["bin_coverage"],
        color=PublicationPalette.UNCERTAINTY,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Uncertainty bin (low → high)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage by uncertainty bin")
    ax.legend(loc="lower left", frameon=False)

    # Right: mean absolute error and comparable interval half-width per bin
    ax = axes[1]
    x = np.arange(len(table))
    ax.plot(x, table["bin_mean_abs_error"], "o-", ms=5, color=PublicationPalette.OBSERVED, label="Mean absolute error")
    ax.plot(x, table["bin_mean_half_width"], "s--", ms=5, color=PublicationPalette.UNCERTAINTY, label="Nominal 95% half-width")
    ax.set_xlabel("Uncertainty bin (low → high)")
    ax.set_ylabel(r"Value ($\mu$m)")
    ax.set_title("Error and interval half-width by bin")
    ax.legend(loc="upper left", frameon=False)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.88, wspace=0.32)
    managed.save()

if __name__ == "__main__":
    main()
