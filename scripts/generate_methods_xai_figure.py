#!/usr/bin/env python3
"""
Generate a methods figure comparing explanation techniques on real data.

Outputs:
    reports/evidence/plots/methods/methods_xai_comparison.png

The figure compares three explanation signals for the vibration spectrogram:
  (a) Ridge coefficient magnitudes
  (b) TreeSHAP mean |SHAP| (LightGBM)
  (c) Grad-CAM frequency importance (ResNetVibCNN)
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

XAI_DIR = ROOT / "reports" / "evidence" / "xai"
OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

VIB_SR = 51_200.0
VIB_NFFT = 512


def bin_to_hz(bin_idx: int) -> float:
    return bin_idx * VIB_SR / VIB_NFFT


def load_top_bins(csv_path: Path, modality: str, top_n: int = 50) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["modality"] == modality].copy()
    # Aggregate over channels and time per freq_bin
    agg = df.groupby("freq_bin")["importance"].sum().reset_index()
    agg["hz"] = agg["freq_bin"].apply(bin_to_hz)
    agg = agg.sort_values("importance", ascending=False)
    return agg.head(top_n).sort_values("freq_bin")


def load_gradcam() -> pd.DataFrame:
    df = pd.read_csv(XAI_DIR / "gradcam_resnetvib_freq_importance.csv")
    df["hz"] = df["freq_bin"].apply(bin_to_hz)
    return df


def plot_freq_importance(ax, df, title, color):
    ax.bar(df["hz"] / 1e3, df["importance"], width=0.08, color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Importance")
    ax.set_xlim(0, 25.6)
    # Mark physical bands
    ax.axvspan(0.5, 2.0, alpha=0.08, color=PublicationPalette.MODEL_FAMILY["LightGBMModel"], label="Low-frequency chatter / structural")
    ax.axvspan(2.0, 15.0, alpha=0.08, color=PublicationPalette.OBSERVED, label="Higher-frequency vibration / chatter")


def main() -> int:
    if not (XAI_DIR / "shap_importance_RidgeRegressionModel_ae_spec+vib_spec.csv").exists():
        print("Ridge SHAP CSV not found. Run scripts/shap_spectrogram_baselines.py first.")
        return 1
    if not (XAI_DIR / "shap_importance_LightGBMModel_ae_spec+vib_spec.csv").exists():
        print("LightGBM SHAP CSV not found. Run scripts/shap_spectrogram_baselines.py first.")
        return 1
    if not (XAI_DIR / "gradcam_resnetvib_freq_importance.csv").exists():
        print("Grad-CAM CSV not found. Run scripts/gradcam_resnet_vib.py first.")
        return 1

    ridge = load_top_bins(
        XAI_DIR / "shap_importance_RidgeRegressionModel_ae_spec+vib_spec.csv",
        modality="vib_spec",
    )
    shap = load_top_bins(
        XAI_DIR / "shap_importance_LightGBMModel_ae_spec+vib_spec.csv",
        modality="vib_spec",
    )
    gradcam = load_gradcam()

    managed = MutableFigure(
        "methods_xai_comparison.png",
        profile=FigureProfiles.VERTICAL_TRIPTYCH,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/generate_methods_xai_figure.py"},
    )
    fig, axes = managed.create()

    plot_freq_importance(
        axes[0],
        ridge,
        "(a) Ridge regression coefficient magnitudes (Vib-dB)",
        PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"],
    )
    plot_freq_importance(
        axes[1],
        shap,
        "(b) TreeSHAP mean |SHAP| (LightGBM, Vib-dB)",
        PublicationPalette.MODEL_FAMILY["LightGBMModel"],
    )
    axes[2].plot(gradcam["hz"] / 1e3, gradcam["mean_importance"], color=PublicationPalette.MODEL_FAMILY["ResNetVibCNN"], lw=2)
    axes[2].fill_between(gradcam["hz"] / 1e3, gradcam["mean_importance"], alpha=0.4, color=PublicationPalette.MODEL_FAMILY["ResNetVibCNN"])
    axes[2].set_xlabel("Frequency (kHz)")
    axes[2].set_ylabel("Mean |Grad-CAM|")
    axes[2].set_xlim(0, 25.6)
    axes[2].axvspan(0.5, 2.0, alpha=0.08, color=PublicationPalette.MODEL_FAMILY["LightGBMModel"])
    axes[2].axvspan(2.0, 15.0, alpha=0.08, color=PublicationPalette.OBSERVED)

    # Legend for physical bands (only on bottom plot)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PublicationPalette.MODEL_FAMILY["LightGBMModel"], alpha=0.15, label="Low-frequency chatter / structural"),
        Patch(facecolor=PublicationPalette.OBSERVED, alpha=0.15, label="Higher-frequency vibration / chatter"),
    ]
    axes[2].legend(handles=legend_elements, loc="upper right", fontsize=7)

    plt.tight_layout()

    managed.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
