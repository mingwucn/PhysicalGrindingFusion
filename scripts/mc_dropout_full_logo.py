#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
Full 16-fold LOGO MC-dropout uncertainty for ResNetVibCNN on vib_spec.

For each LOGO fold we load the pre-trained checkpoint, run 50 stochastic
forward passes on the held-out test condition, and record prediction
intervals. Aggregating across all folds gives one MC-dropout estimate per
sample and therefore includes every condition, including Condition 7.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from scripts.train_and_evaluate import smart_load_data, CVSplitter, scale_data_dict
from grinding_physic_fusion.models.architectures import model_factory

PublicationPlotter.set_style()

CONFIG = "vib_spec"
MODEL_NAME = "ResNetVibCNN"
DEVICE = torch.device("cpu")
N_MC = 50
Z = 1.96
N_FOLDS = 16
REPEAT = 0
OUT_DIR = Path("reports/evidence/uncertainty")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_fold(full_data, fold: int):
    groups = full_data["condition_ids"]
    splitter = CVSplitter(n_folds=N_FOLDS, grouped=False, logo=True, seed=42)
    train_idx = test_idx = None
    for r, f, tr, val, te in splitter.split(groups):
        if f == fold and r == REPEAT:
            train_idx, test_idx = tr, te
            break
    if train_idx is None:
        raise ValueError(f"Fold {fold} not found")

    # Match training defaults: per-sample spec normalisation only, target
    # standardised per fold. Predictions are inverse-transformed to µm.
    scaled, target_scaler = scale_data_dict(
        full_data, train_idx, scale_specs=False, scale_target=True
    )
    X_test = torch.from_numpy(scaled["vib_spec"][test_idx]).float().to(DEVICE)
    y_test = scaled["targets"][test_idx]
    cond_ids = scaled["condition_ids"][test_idx]

    ckpt_path = Path(f"checkpoints/{MODEL_NAME}_{CONFIG}_fold{fold}_repeat{REPEAT}.pt")
    model = model_factory(MODEL_NAME)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    # Keep dropout active for MC sampling, but freeze batch-normalisation
    # statistics so small test batches do not shift predictions.
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    def to_um(arr: np.ndarray) -> np.ndarray:
        if target_scaler is None:
            return arr
        return target_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)

    preds = []
    with torch.no_grad():
        for _ in range(N_MC):
            out = model(X_test).squeeze(-1).cpu().numpy()
            preds.append(to_um(out))
    preds = np.stack(preds, axis=0)  # (N_MC, n_test)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    y_test_um = to_um(y_test)
    df = pd.DataFrame({
        "condition_id": cond_ids,
        "y_true": y_test_um,
        "y_pred": mean_pred,
        "y_std": std_pred,
        "lower": mean_pred - Z * std_pred,
        "upper": mean_pred + Z * std_pred,
    })
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["interval_width"] = df["upper"] - df["lower"]
    df["fold"] = fold
    return df


def render_figures(full: pd.DataFrame) -> None:
    """Render canonical full-LOGO predictions without recomputing MC draws."""
    sort_df = full.sort_values("y_pred").reset_index(drop=True)
    intervals_figure = MutableFigure(
        "mc_dropout_intervals_ResNetVibCNN_vib_spec.png",
        profile=FigureProfiles.DOUBLE,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/mc_dropout_full_logo.py", "mode": "render_only"},
    )
    fig, ax = intervals_figure.create()
    x = np.arange(len(sort_df))
    ax.fill_between(x, sort_df["lower"], sort_df["upper"], color=PublicationPalette.UNCERTAINTY, alpha=0.3, label="Nominal 95% MC-dropout epistemic band")
    ax.scatter(x, sort_df["y_true"], color=PublicationPalette.OBSERVED, s=15, zorder=3, label="Observed $R_a$")
    ax.set_xlabel("Sample index (sorted by predicted mean)")
    ax.set_ylabel(r"$R_a$ (µm)")
    ax.set_title(f"MC-dropout prediction intervals — full LOGO ({MODEL_NAME} / {CONFIG})")
    ax.legend()
    fig.tight_layout()
    intervals_figure.save()

    calibration_figure = MutableFigure(
        "mc_dropout_calibration_full_logo.png",
        profile=FigureProfiles.SINGLE,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/mc_dropout_full_logo.py", "mode": "render_only"},
    )
    fig, ax = calibration_figure.create()
    colors = np.where(full["covered"], PublicationPalette.MODEL_FAMILY["LightGBMModel"], PublicationPalette.CONDITION_7)
    ax.scatter(full["y_std"], full["abs_error"], c=colors, alpha=0.6, s=30)
    ax.set_xlabel("MC-dropout standard deviation (µm)")
    ax.set_ylabel("Absolute error (µm)")
    ax.set_title(f"MC-dropout uncertainty calibration\n{MODEL_NAME} / {CONFIG} / full LOGO")
    max_std = full["y_std"].max()
    ax.plot([0, max_std], [0, max_std], color=PublicationPalette.PREDICTED, linestyle="--", lw=1, label="error = std")
    ax.plot([0, max_std], [0, 1.96 * max_std], "k:", lw=1, label="error = 1.96 std")
    ax.legend()
    fig.tight_layout()
    calibration_figure.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true", help="Render figures from the frozen canonical CSV.")
    args = parser.parse_args()
    canonical_path = OUT_DIR / f"mc_dropout_{MODEL_NAME}_{CONFIG}_logo_all.csv"
    if args.render_only:
        if not canonical_path.exists():
            raise FileNotFoundError(f"Missing canonical MC-dropout CSV: {canonical_path}")
        render_figures(pd.read_csv(canonical_path))
        return
    print("Loading data ...", flush=True)
    full_data = smart_load_data([MODEL_NAME], [CONFIG])

    dfs = []
    for fold in range(N_FOLDS):
        print(f"Fold {fold} ...", flush=True)
        df = run_fold(full_data, fold)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    csv_path = OUT_DIR / f"mc_dropout_{MODEL_NAME}_{CONFIG}_logo_all.csv"
    full.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

    overall = {
        "model": MODEL_NAME,
        "config": CONFIG,
        "n_samples": len(full),
        "coverage_95": float(full["covered"].mean()),
        "mean_interval_width": float(full["interval_width"].mean()),
        "median_interval_width": float(full["interval_width"].median()),
        "mean_abs_error": float(full["abs_error"].mean()),
        "correlation_std_mae": float(full["y_std"].corr(full["abs_error"])),
    }
    pd.DataFrame([overall]).to_csv(
        OUT_DIR / f"mc_dropout_summary_{MODEL_NAME}_{CONFIG}_logo_all.csv", index=False
    )
    print(overall, flush=True)

    per_condition = full.groupby("condition_id").agg(
        n=("y_true", "count"),
        coverage=("covered", "mean"),
        mean_width=("interval_width", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()
    per_condition.to_csv(
        OUT_DIR / f"mc_dropout_per_condition_{MODEL_NAME}_{CONFIG}_logo_all.csv", index=False
    )
    print(per_condition, flush=True)

    render_figures(full)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
