#!/usr/bin/env python3
"""
Generate a bar chart comparing the best LOGO MAE per signal representation family.

Inputs:
    reports/evidence/tables/signal_representation_comparison.csv

Outputs:
    reports/evidence/plots/methods/methods_signal_representation_comparison.png
    overleaf/images/methods_signal_representation_comparison.png
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

TABLE_PATH = ROOT / "reports" / "evidence" / "tables" / "signal_representation_comparison.csv"
OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

# Preferred order and display labels
ORDER = [
    "dB-z spectrogram",
    "Log-mel spectrogram",
    "Spectrogram",
    "WST",
    "Trajectory",
    "Time-domain features",
    "Process params",
]

LABEL_MAP = {
    "dB-z spectrogram": "dB-z\nspectrogram",
    "Log-mel spectrogram": "Log-mel\nspectrogram",
    "Spectrogram": "dB\nspectrogram",
    "WST": "Wavelet\nscattering",
    "Trajectory": "Trajectory",
    "Time-domain features": "Time-domain\nfeatures",
    "Process params": "Process\nparameters",
}


def parse_mean_std(s: str) -> tuple[float, float]:
    parts = s.split("±")
    return float(parts[0].strip()), float(parts[1].strip())


def main() -> int:
    if not TABLE_PATH.exists():
        print(f"{TABLE_PATH} not found. Run scripts/build_signal_representation_comparison.py first.")
        return 1

    df = pd.read_csv(TABLE_PATH)
    # Rename "Other" group to "Trajectory" if it contains vib_trajectory
    df.loc[df["config"] == "vib_trajectory", "group"] = "Trajectory"

    records = []
    for g in ORDER:
        row = df[df["group"] == g]
        if row.empty:
            continue
        row = row.iloc[0]
        mae_mean, mae_std = parse_mean_std(row["MAE (µm)"])
        records.append({
            "group": g,
            "label": LABEL_MAP[g],
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "model": row["model"],
            "config": row["config"],
        })

    plot_df = pd.DataFrame(records)

    managed = MutableFigure(
        "methods_signal_representation_comparison.png",
        profile=FigureProfiles.DOUBLE,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/generate_signal_representation_bar_chart.py"},
    )
    fig, ax = managed.create()
    x = np.arange(len(plot_df))
    colors = [PublicationPalette.model(model, i) for i, model in enumerate(plot_df["model"])]
    bars = ax.bar(
        x,
        plot_df["mae_mean"],
        # MAE cannot be negative. Retain the upper fold SD and cap only the
        # lower display whisker at zero.
        yerr=np.vstack(
            [np.minimum(plot_df["mae_std"], plot_df["mae_mean"]), plot_df["mae_std"]]
        ),
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylabel("Mean LOGO MAE (µm)")
    ax.set_title("Best mean MAE by signal representation family")
    ax.set_ylim(bottom=0)

    # Annotate bars
    for bar, mean, std in zip(bars, plot_df["mae_mean"], plot_df["mae_std"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.001,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    managed.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
