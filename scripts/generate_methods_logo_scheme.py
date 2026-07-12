#!/usr/bin/env python3
"""
Generate the methods LOGO cross-validation schematic.

Outputs:
    reports/evidence/plots/methods/methods_logo_scheme.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

N_CONDITIONS = 16
TEST_COND = 0      # condition i
VAL_COND = 1       # condition (i+1) mod 16
COLORS = {"train": PublicationPalette.TRAIN, "val": PublicationPalette.VALIDATION, "test": PublicationPalette.TEST}


def main() -> int:
    managed = MutableFigure("methods_logo_scheme.png", profile=FigureProfiles.DOUBLE, out_dir=OUT_DIR, overleaf_dir=ROOT / "overleaf" / "images", metadata={"generator": "scripts/generate_methods_logo_scheme.py"})
    fig, ax = managed.create()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        6,
        7.5,
        "Leave-one-condition-out cross-validation",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    # Grid of conditions
    n_cols = 8
    n_rows = 2
    box_w = 1.0
    box_h = 0.9
    x0 = 1.2
    y0 = 4.5
    dx = 1.15
    dy = 1.2

    for c in range(N_CONDITIONS):
        row = c // n_cols
        col = c % n_cols
        x = x0 + col * dx
        y = y0 - row * dy

        # Use 1-based condition labels to match the manuscript.
        c1 = c + 1
        if c == TEST_COND:
            color = COLORS["test"]
            label = f"C{c1}\nTEST"
        elif c == VAL_COND:
            color = COLORS["val"]
            label = f"C{c1}\nVAL"
        else:
            color = COLORS["train"]
            label = f"C{c1}"

        rect = plt.Rectangle(
            (x, y),
            box_w,
            box_h,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )

    # Annotation arrows for one fold
    ax.annotate(
        "One condition held out as test",
        xy=(x0 + (TEST_COND % n_cols) * dx + box_w / 2, y0 - (TEST_COND // n_cols) * dy + box_h / 2),
        xytext=(x0 + 0.5, 2.2),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        fontsize=7,
        color="darkred",
        fontweight="bold",
        ha="center",
    )

    ax.annotate(
        "One condition reserved for validation",
        xy=(x0 + (VAL_COND % n_cols) * dx + box_w / 2, y0 - (VAL_COND // n_cols) * dy + box_h / 2),
        xytext=(x0 + 4.0, 2.2),
        arrowprops=dict(arrowstyle="->", color="darkgoldenrod", lw=1.5),
        fontsize=7,
        color="darkgoldenrod",
        fontweight="bold",
        ha="center",
    )

    ax.text(
        6,
        1.2,
        "Remaining 14 conditions are used for training.\n"
        "The procedure is repeated 16 times so every condition serves as test once.",
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["train"], edgecolor="black", label="Train (14 conditions)"),
        mpatches.Patch(facecolor=COLORS["val"], edgecolor="black", label="Validation (1 condition)"),
        mpatches.Patch(facecolor=COLORS["test"], edgecolor="black", label="Test (1 condition)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7,
        frameon=True,
        fancybox=False,
        shadow=False,
    )

    plt.tight_layout()
    managed.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
