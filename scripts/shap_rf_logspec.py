#!/usr/bin/env python3
"""
TreeSHAP explanation for the recommended random forest on fused AE and
vibration logspec descriptors.

This script reproduces the RF-specific SHAP summary figure used in the
Supplementary Materials. It trains a single random forest on the cached
AE/vibration logspec descriptors, computes TreeSHAP mean absolute importance,
and plots per-frequency importance for AE and vibration.

Outputs:
    reports/evidence/plots/results/shap_rf_ae_logspec_vib_logspec.png
    reports/evidence/xai/shap_importance_RandomForestModel_ae_logspec+vib_logspec.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPlotter

CACHE_DIR = ROOT / "data" / "intermediate" / "cached_specs"
OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "results"
CSV_DIR = ROOT / "reports" / "evidence" / "xai"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

AE_NFFT = 598
AE_SR = 4_000_000.0
VIB_NFFT = 512
VIB_SR = 51_200.0

N_BACKGROUND = 50
N_TEST = 100
SEED = 42

PublicationPlotter.set_style()


def flatten_samples(ae: np.ndarray, vib: np.ndarray) -> np.ndarray:
    """Flatten per-sample AE and Vib descriptors into row vectors."""
    n = ae.shape[0]
    ae_flat = ae.reshape(n, -1)
    vib_flat = vib.reshape(n, -1)
    return np.concatenate([ae_flat, vib_flat], axis=1)


def split_modality_importance(importance: np.ndarray, ae_shape: tuple, vib_shape: tuple):
    """Return (ae_importance, vib_importance) as flattened feature vectors."""
    ae_n = int(np.prod(ae_shape[1:]))
    vib_n = int(np.prod(vib_shape[1:]))
    return importance[:ae_n], importance[ae_n:ae_n + vib_n]


def average_time(importance: np.ndarray, shape: tuple) -> np.ndarray:
    """Average flat importance over channels and time to get frequency."""
    # shape: (n_samples, n_channels, n_freq, n_time)
    reshaped = importance.reshape(shape[1:])
    return np.abs(reshaped).mean(axis=(0, 2))  # (n_channels, n_freq)


def plot_panel(ax, freqs: np.ndarray, importance: np.ndarray, title: str, ylabel: str):
    ax.fill_between(freqs, importance, alpha=0.3)
    ax.plot(freqs, importance, lw=1.2)
    ax.set_xlabel("Frequency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def main() -> int:
    mean_path = CACHE_DIR / "mean_specs.npz"
    alt_path = CACHE_DIR / "alternative_reps.npz"
    if not mean_path.exists() or not alt_path.exists():
        print(f"Missing cached data: {mean_path} or {alt_path}")
        return 1

    mean_cache = np.load(mean_path, allow_pickle=True)
    alt_cache = np.load(alt_path, allow_pickle=True)

    ae = np.asarray(alt_cache["ae_logspec"])
    vib = np.asarray(alt_cache["vib_logspec"])
    y = np.asarray(mean_cache["targets"])

    X = flatten_samples(ae, vib)

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    bg_idx = idx[:N_BACKGROUND]
    test_idx = idx[N_BACKGROUND:N_BACKGROUND + N_TEST]

    print("Training random forest on flattened AE+Vib logspec descriptors ...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X, y)

    print("Computing TreeSHAP values ...")
    explainer = shap.TreeExplainer(model, X[bg_idx])
    shap_values = explainer.shap_values(X[test_idx], approximate=True)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ae_imp_flat, vib_imp_flat = split_modality_importance(
        mean_abs_shap, ae.shape, vib.shape
    )
    ae_imp_freq = average_time(ae_imp_flat, ae.shape)  # (n_freq,)
    vib_imp_freq = average_time(vib_imp_flat, vib.shape)  # (n_freq,)

    ae_freqs = np.arange(len(ae_imp_freq)) * AE_SR / AE_NFFT
    vib_freqs = np.arange(len(vib_imp_freq)) * VIB_SR / VIB_NFFT

    # Save CSV
    ae_df = pd.DataFrame({"freq_khz": ae_freqs / 1e3, "importance": ae_imp_freq})
    vib_df = pd.DataFrame({"freq_khz": vib_freqs / 1e3, "importance": vib_imp_freq})
    ae_df.to_csv(CSV_DIR / "shap_importance_RandomForestModel_ae_logspec.csv", index=False)
    vib_df.to_csv(CSV_DIR / "shap_importance_RandomForestModel_vib_logspec.csv", index=False)

    managed = MutableFigure(
        "shap_rf_ae_logspec_vib_logspec.png",
        profile=FigureProfiles.TWO_PANEL_ROW,
        out_dir=OUT_DIR,
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/shap_rf_logspec.py"},
    )
    fig, axes = managed.create()
    plot_panel(
        axes[0],
        ae_freqs / 1e3,
        ae_imp_freq,
        "AE TreeSHAP mean $|\\text{SHAP}|$",
        "Mean absolute SHAP",
    )
    axes[0].set_xlabel("Frequency (kHz)")
    plot_panel(
        axes[1],
        vib_freqs / 1e3,
        vib_imp_freq,
        "Vibration TreeSHAP mean $|\\text{SHAP}|$",
        "Mean absolute SHAP",
    )
    axes[1].set_xlabel("Frequency (kHz)")

    plt.tight_layout()
    managed.save()
    print(f"Saved CSVs to {CSV_DIR}")
    print(f"Saved figure to {OUT_DIR / 'shap_rf_ae_logspec_vib_logspec.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
