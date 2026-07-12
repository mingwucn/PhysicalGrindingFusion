#!/usr/bin/env python3
"""Generate the manuscript XAI composite figure used as Figure 19."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

XAI_DIR = ROOT / "reports" / "evidence" / "xai"
OUT_NAME = "shap_importance_LightGBMModel_ae_spec+vib_spec.png"
AE_SR = 4_000_000.0
AE_NFFT = 598
VIB_SR = 51_200.0
VIB_NFFT = 512


def ae_hz(freq_bin: int) -> float:
    return freq_bin * AE_SR / AE_NFFT


def vib_hz(freq_bin: int) -> float:
    return freq_bin * VIB_SR / VIB_NFFT


def load_top_ae_shap(top_n: int = 12) -> pd.DataFrame:
    df = pd.read_csv(XAI_DIR / "shap_importance_LightGBMModel_ae_spec+vib_spec.csv")
    ae = df[df["modality"] == "ae_spec"].copy()
    ae = ae.groupby(["channel", "freq_bin"], as_index=False)["importance"].sum()
    ae = ae.sort_values("importance", ascending=False).head(top_n).copy()
    ae["frequency_mhz"] = ae["freq_bin"].map(ae_hz) / 1e6
    ae["label"] = ae.apply(
        lambda r: f"AE channel {int(r.channel) + 1}, {r.frequency_mhz:.2f} MHz",
        axis=1,
    )
    return ae.sort_values("importance")


def load_gradcam() -> pd.DataFrame:
    df = pd.read_csv(XAI_DIR / "gradcam_resnetvib_freq_importance.csv")
    df["frequency_khz"] = df["freq_bin"].map(vib_hz) / 1e3
    return df


def main() -> int:
    PublicationPlotter.set_style()
    ae = load_top_ae_shap()
    gradcam = load_gradcam()

    managed = MutableFigure(
        OUT_NAME,
        profile=FigureProfiles.TWO_PANEL_ROW,
        out_dir=XAI_DIR,
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/generate_xai_composite_figure.py"},
    )
    fig, axes = managed.create(gridspec_kw={"width_ratios": [0.95, 1.2]})

    axes[0].barh(ae["label"], ae["importance"], color=PublicationPalette.MODEL_FAMILY["LightGBMModel"])
    axes[0].set_xlabel("TreeSHAP mean |SHAP|")
    axes[0].set_title("(a) LightGBM TreeSHAP on AE-dB")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].tick_params(axis="x", labelsize=7)

    axes[1].plot(gradcam["frequency_khz"], gradcam["mean_importance"], color=PublicationPalette.MODEL_FAMILY["ResNetVibCNN"], lw=1.8)
    axes[1].fill_between(
        gradcam["frequency_khz"],
        gradcam["mean_importance"],
        color=PublicationPalette.MODEL_FAMILY["ResNetVibCNN"],
        alpha=0.2,
    )
    axes[1].axvspan(0.5, 2.0, color=PublicationPalette.MODEL_FAMILY["LightGBMModel"], alpha=0.10, label="0.5-2 kHz")
    axes[1].axvspan(2.0, 15.0, color=PublicationPalette.OBSERVED, alpha=0.10, label="2-15 kHz")
    axes[1].set_xlim(0, 25.6)
    axes[1].set_xlabel("Vibration frequency (kHz)")
    axes[1].set_ylabel("Mean |Grad-CAM|")
    axes[1].set_title("(b) ResNetVibCNN Grad-CAM on Vib-dB")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    fig.tight_layout()
    managed.save()
    print(f"Saved {XAI_DIR / OUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
