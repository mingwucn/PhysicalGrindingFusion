#!/usr/bin/env python3
"""
Generate the methods signal-representations figure.

Outputs:
    reports/evidence/plots/methods/methods_signal_representations.png

The figure shows a 2x2 panel:
  (a) AE dB spectrogram      (b) AE log-mel spectrogram
  (c) Vibration dB spectrogram    (d) Vibration log-mel spectrogram
for condition 10, sample 1 (Ra ≈ 0.098 µm).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

CACHE_DIR = ROOT / "data" / "intermediate" / "cached_specs"
OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

# Which sample to illustrate (matches the manuscript caption)
CONDITION_ID = 10
SAMPLE_ID = 1

# Channel indices
AE_CHANNEL = 1  # archived AE plane 2; physical/filter provenance is unresolved
VIB_CHANNEL = 2  # Z-axis vibration

# Sampling rates (Hz)
AE_SR = 4_000_000.0
VIB_SR = 51_200.0

# STFT parameters used during preprocessing (matches Methods.tex)
AE_N_FFT = 598
AE_HOP = 426
VIB_N_FFT = 512
VIB_HOP = 426

# Mel bins
N_MELS = 64


def load_sample():
    mean_data = np.load(CACHE_DIR / "mean_specs.npz", allow_pickle=True)
    alt_data = np.load(CACHE_DIR / "alternative_reps.npz", allow_pickle=True)

    mask = (mean_data["condition_ids"] == CONDITION_ID) & (
        mean_data["sample_ids"] == SAMPLE_ID
    )
    idx = int(np.where(mask)[0][0])

    out = {
        "ae_spec": mean_data["ae_spec"][idx, AE_CHANNEL],  # (F, T)
        "vib_spec": mean_data["vib_spec"][idx, VIB_CHANNEL],  # (F, T)
        "ae_mel": alt_data["ae_mel"][idx, AE_CHANNEL],  # (M, T)
        "vib_mel": alt_data["vib_mel"][idx, VIB_CHANNEL],  # (M, T)
        "target": float(mean_data["targets"][idx]),
    }
    mean_data.close()
    alt_data.close()
    return out


def time_axis(n_time: int, sr: float, n_fft: int, hop: int) -> np.ndarray:
    """Return the centre time (seconds) of each STFT frame."""
    frame_time = (np.arange(n_time) * hop + n_fft // 2) / sr
    return frame_time


def freq_axis_lin(n_freq: int, sr: float, n_fft: int) -> np.ndarray:
    """Return linear frequency axis for an STFT magnitude spectrogram."""
    return np.linspace(0, sr / 2, n_freq) / 1e3  # kHz


def freq_axis_mel(n_mels: int, sr: float) -> np.ndarray:
    """Return mel-spaced frequencies in kHz."""
    # librosa-style mel frequencies
    mel_min = 2595.0 * np.log10(1.0 + (sr / 2) / 700.0)
    mels = np.linspace(0.0, mel_min, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    return freqs[1:-1] / 1e3  # kHz


def plot_panel(ax, spec, extent, title, ylabel, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    if ylabel:
        ax.set_ylabel(ylabel)
    return im


def plot_mel_panel(ax, spec, title, ylabel, cmap="viridis", vmin=None, vmax=None):
    """Plot mel data in evenly spaced mel-bin coordinates, not linear Hz."""
    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("STFT frame")
    if ylabel:
        ax.set_ylabel(ylabel)
    return im


def main() -> int:
    if not (CACHE_DIR / "mean_specs.npz").exists():
        print("Run scripts/cache_mean_spectrograms.py first.")
        return 1
    if not (CACHE_DIR / "alternative_reps.npz").exists():
        print("Run scripts/cache_alternative_representations.py first.")
        return 1

    data = load_sample()

    managed = MutableFigure(
        "methods_signal_representations.png",
        profile=FigureProfiles.TWO_BY_TWO,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/generate_methods_signal_representations.py"},
    )
    fig, axes = managed.create(gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]})

    ae_t = time_axis(data["ae_spec"].shape[1], AE_SR, AE_N_FFT, AE_HOP)
    ae_f = freq_axis_lin(data["ae_spec"].shape[0], AE_SR, AE_N_FFT)
    ae_extent = [ae_t[0], ae_t[-1], ae_f[0], ae_f[-1]]

    vib_t = time_axis(data["vib_spec"].shape[1], VIB_SR, VIB_N_FFT, VIB_HOP)
    vib_f = freq_axis_lin(data["vib_spec"].shape[0], VIB_SR, VIB_N_FFT)
    vib_extent = [vib_t[0], vib_t[-1], vib_f[0], vib_f[-1]]

    # AE dB spectrogram
    im0 = plot_panel(
        axes[0, 0],
        data["ae_spec"],
        ae_extent,
        "(a) AE dB spectrogram",
        "Frequency (kHz)",
        cmap=PublicationPalette.CONTINUOUS,
    )

    # AE log-mel spectrogram
    im1 = plot_mel_panel(
        axes[0, 1],
        data["ae_mel"],
        "(b) AE log-mel spectrogram",
        "Mel-filter index",
        cmap="cividis",
    )

    # Vibration dB spectrogram (dB)
    im2 = plot_panel(
        axes[1, 0],
        data["vib_spec"],
        vib_extent,
        "(c) Vibration dB spectrogram",
        "Frequency (kHz)",
        cmap=PublicationPalette.CONTINUOUS,
    )

    # Vibration log-mel spectrogram
    im3 = plot_mel_panel(
        axes[1, 1],
        data["vib_mel"],
        "(d) Vibration log-mel spectrogram",
        "Mel-filter index",
        cmap="cividis",
    )

    # Colorbars
    for ax, im, label in zip(
        axes.flat,
        [im0, im1, im2, im3],
        ["dB", "dB", "dB", "dB"],
    ):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)

    fig.suptitle(
        f"Example signal representations for condition {CONDITION_ID}, sample {SAMPLE_ID} "
        f"($R_a = {data['target']:.3f}$ µm)",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    managed.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
