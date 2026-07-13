#!/usr/bin/env python3
"""Render the supplementary fold-wise RF attribution stability figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette

TABLE = ROOT / "reports" / "evidence" / "tables" / "supp_shap_fold_stability.csv"
OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "results"


def main() -> int:
    data = pd.read_csv(TABLE).sort_values("held_out_condition")
    figure = MutableFigure(
        "shap_fold_stability.png",
        profile=FigureProfiles.VERTICAL_DUO,
        out_dir=OUT_DIR,
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/generate_shap_fold_stability_figure.py"},
    )
    fig, axes = figure.create()
    condition = data["held_out_condition"]

    axes[0].plot(condition, data["top_vib_freq_kHz"], marker="o", color=PublicationPalette.OBSERVED)
    axes[0].set_ylabel("Dominant vibration bin (kHz)")

    axes[1].plot(condition, data["top_ae_freq_kHz"], marker="o", color=PublicationPalette.MODEL_FAMILY["LightGBMModel"])
    axes[1].set_xlabel("Held-out condition")
    axes[1].set_ylabel("Dominant AE bin (kHz)")
    axes[1].set_xticks(range(1, 17))

    for axis in axes:
        axis.axvline(7, color=PublicationPalette.CONDITION_7, linestyle="--", linewidth=0.8)
        axis.set_xlim(0.5, 16.5)
    fig.tight_layout()
    figure.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
