#!/usr/bin/env python3
"""Plot out-of-fold RF/dB-z TreeSHAP profiles from archived CSVs."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

XAI = ROOT / "reports" / "evidence" / "xai"
OUT = ROOT / "reports" / "evidence" / "plots" / "results"
PublicationPlotter.set_style()


def main() -> int:
    profiles = {}
    for modality in ("ae", "vib"):
        pooled = pd.read_csv(XAI / f"rf_dbz_oof_treeshap_{modality}_profile.csv")
        pooled["importance_fraction"] = pooled["mean_abs_shap"] / pooled["mean_abs_shap"].sum()
        profiles[modality] = {
            "Sample-pooled": pooled,
            "Condition-balanced": pd.read_csv(XAI / f"rf_dbz_oof_treeshap_{modality}_condition_balanced.csv"),
            "Condition-balanced, no C7": pd.read_csv(XAI / f"rf_dbz_oof_treeshap_{modality}_condition_balanced_without_condition7.csv"),
        }
    managed = MutableFigure(
        "rf_dbz_oof_treeshap.png",
        profile=FigureProfiles.TWO_PANEL_ROW,
        out_dir=OUT,
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/generate_rf_oof_shap_figure.py"},
    )
    fig, axes = managed.create()
    colors = (PublicationPalette.OBSERVED, PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"], PublicationPalette.CONDITION_7)
    for ax, modality, title in (
        (axes[0], "ae", "(a) AE out-of-fold TreeSHAP"),
        (axes[1], "vib", "(b) Vibration out-of-fold TreeSHAP"),
    ):
        for color, (label, frame) in zip(colors, profiles[modality].items()):
            ax.plot(frame["frequency_khz"], frame["importance_fraction"], linewidth=1.3, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Normalized attribution per bin")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=True)
    fig.subplots_adjust(bottom=0.22)
    managed.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
