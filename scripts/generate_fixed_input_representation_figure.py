#!/usr/bin/env python3
"""Render the fixed-input representation comparison as a readable heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import (  # noqa: E402
    FigureProfiles,
    PublicationFigure,
    PublicationPlotter,
)


TABLE = ROOT / "reports" / "evidence" / "tables" / "fixed_input_representation_comparison.csv"
PLOTS = ROOT / "reports" / "evidence" / "plots" / "results"
OVERLEAF = ROOT / "overleaf" / "images"

FAMILIES = ["dB", "dB-z", "log-mel", "WST", "time-domain", "process parameters"]
FAMILY_LABELS = {
    "dB": "AE-dB + Vib-dB",
    "dB-z": "AE-dB-z + Vib-dB-z",
    "log-mel": "AE-log-mel + Vib-log-mel",
    "WST": "AE-WST + Vib-WST",
    "time-domain": "AE/Vib time-domain features",
    "process parameters": "Process parameters",
}
LEARNERS = ["RandomForestModel", "RidgeRegressionModel", "LightGBMModel"]
LEARNER_LABELS = {
    "RandomForestModel": "Random forest",
    "RidgeRegressionModel": "Ridge",
    "LightGBMModel": "LightGBM",
}


class FixedInputRepresentationHeatmap(PublicationFigure):
    """Concrete OOP figure that mutates the shared double-column profile."""

    profile = FigureProfiles.DOUBLE_TALL

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(
            "representation_controlled_comparison.png",
            profile=self.profile,
            out_dir=PLOTS,
            overleaf_dir=OVERLEAF,
            metadata={
                "generator": "scripts/generate_fixed_input_representation_figure.py",
                "plot_type": "fixed-input representation heatmap",
                "input": "fixed_input_representation_comparison.csv",
            },
        )
        self.data = data

    def draw(self) -> None:
        assert self.fig is not None and self.axes is not None
        ax = self.axes
        matrix = np.array([
            [
                self.data.loc[
                    (self.data["family"] == family) & (self.data["learner"] == learner),
                    "mean_mae",
                ].iloc[0]
                for learner in LEARNERS
            ]
            for family in FAMILIES
        ])
        std = np.array([
            [
                self.data.loc[
                    (self.data["family"] == family) & (self.data["learner"] == learner),
                    "std_mae",
                ].iloc[0]
                for learner in LEARNERS
            ]
            for family in FAMILIES
        ])

        image = ax.imshow(
            matrix,
            cmap="YlGnBu_r",
            norm=Normalize(vmin=float(matrix.min()), vmax=float(matrix.max())),
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(LEARNERS)), [LEARNER_LABELS[key] for key in LEARNERS])
        ax.set_yticks(np.arange(len(FAMILIES)), [FAMILY_LABELS[key] for key in FAMILIES])
        ax.set_xlabel("Fixed learner")
        ax.set_ylabel("Identical input configuration")
        ax.set_title("Fixed-input LOGO representation comparison")

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                rgba = image.cmap(image.norm(matrix[row, col]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                colour = "black" if luminance > 0.52 else "white"
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.4f}\n$\\pm${std[row, col]:.4f}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=8,
                )
        cbar = self.fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label("Mean LOGO MAE (µm)")
        self.fig.subplots_adjust(left=0.30, right=0.92, bottom=0.16, top=0.90)


def main() -> int:
    if not TABLE.exists():
        raise FileNotFoundError(f"Missing {TABLE}; run fixed_input_representation_comparison.py first.")
    PublicationPlotter.set_style()
    data = pd.read_csv(TABLE)
    data = data[data["learner"].isin(LEARNERS)].copy()
    expected = len(FAMILIES) * len(LEARNERS)
    if len(data) != expected:
        raise ValueError(f"Expected {expected} fixed-input rows, found {len(data)}")
    figure = FixedInputRepresentationHeatmap(data)
    figure.render()
    figure.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
