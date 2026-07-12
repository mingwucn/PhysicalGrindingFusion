#!/usr/bin/env python3
"""
Generate the methods model-taxonomy figure as two cleaner side-by-side panels.

Outputs:
    reports/evidence/plots/methods/methods_model_taxonomy.png
    overleaf/images/methods_model_taxonomy.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()


def draw_box(ax, x, y, w, h, text, color, fontsize=7, text_color="black", alpha=1.0, bold=True):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color,
        edgecolor="black",
        linewidth=1.0,
        alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight="bold" if bold else "normal",
        wrap=True,
    )
    return box


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2),
    )


def draw_panel(ax, title, items, explanation_label, header_color, text_color):
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5.5,
        9.5,
        title,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="black",
    )

    # Input representations box
    input_y = 8.0
    draw_box(ax, 2.75, input_y, 5.5, 0.7, "Input representations", PublicationPalette.TRAIN, fontsize=7)

    # Branch header
    header_y = 6.6
    header_w = 9.0
    draw_box(ax, 1.0, header_y, header_w, 0.7, explanation_label, header_color, fontsize=7)
    draw_arrow(ax, 5.5, input_y, 5.5, header_y + 0.7)

    # Model boxes
    n_items = len(items)
    box_w = 4.2
    box_h = 0.55
    gap = 0.18
    total_h = n_items * box_h + (n_items - 1) * gap
    start_y = header_y - 0.6 - total_h
    header_center_x = 1.0 + header_w / 2
    for i, (model, explain) in enumerate(items):
        y = start_y + (n_items - 1 - i) * (box_h + gap)
        box_center_x = 1.0 + box_w / 2
        draw_box(ax, 1.0, y, box_w, box_h, model, header_color, fontsize=7)
        ax.text(
            1.0 + box_w + 0.25,
            y + box_h / 2,
            explain,
            ha="left",
            va="center",
            fontsize=6,
            color=text_color,
            style="italic",
        )
        draw_arrow(ax, header_center_x, header_y, box_center_x, y + box_h)

    # Output box
    output_y = 0.5
    output_w = 5.5
    output_x = 1.0 + (header_w - output_w) / 2
    draw_box(ax, output_x, output_y, output_w, 0.7, "Predict $R_a$ (µm)", PublicationPalette.CONDITION_7, fontsize=7)
    draw_arrow(ax, 1.0 + box_w / 2, start_y, output_x + output_w / 2, output_y + 0.7)


def main() -> int:
    managed = MutableFigure("methods_model_taxonomy.png", profile=FigureProfiles.TWO_PANEL_ROW, out_dir=OUT_DIR, overleaf_dir=ROOT / "overleaf" / "images", metadata={"generator": "scripts/generate_methods_model_taxonomy.py"})
    fig, axes = managed.create()

    transparent = [
        ("Ridge regression", "signed coefficients"),
        ("Random forest", "feature importance"),
        ("LightGBM / XGBoost", "TreeSHAP"),
        ("Shallow MLP", "two hidden layers"),
    ]
    draw_panel(
        axes[0],
        "Transparent baselines",
        transparent,
        "interpretable outputs",
        PublicationPalette.MODEL_FAMILY["LightGBMModel"],
        "black",
    )

    deep = [
        ("ResNetAECNN /\nResNetVibCNN", "spectrogram encoders"),
        ("ResNetFusion", "multimodal fusion"),
        ("BilinearFusion\nNetwork", "attention-based fusion"),
        ("Multiscale\nSpectrogramCNN", "multi-resolution"),
        ("TrajectoryCNN", "sequence model"),
        ("GNN /\nTabTransformer", "structure/attention"),
    ]
    draw_panel(
        axes[1],
        "Deep-learning baselines",
        deep,
        "post-hoc explanation",
        PublicationPalette.MODEL_FAMILY["ResNetVibCNN"],
        "black",
    )

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=PublicationPalette.MODEL_FAMILY["LightGBMModel"], edgecolor="black", label="Transparent"),
        mpatches.Patch(facecolor=PublicationPalette.MODEL_FAMILY["ResNetVibCNN"], edgecolor="black", label="Deep learning"),
        mpatches.Patch(facecolor=PublicationPalette.CONDITION_7, edgecolor="black", label="Target"),
        mpatches.Patch(facecolor=PublicationPalette.TRAIN, edgecolor="black", label="Input"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=6,
        frameon=True,
        fancybox=False,
        shadow=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    managed.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
