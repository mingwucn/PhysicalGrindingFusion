#!/usr/bin/env python3
"""
Full 16-fold LOGO MC-dropout uncertainty for an arbitrary PyTorch CNN.

Usage:
    python scripts/mc_dropout_generic.py --model ResNetAECNN --config ae_spec
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.data.dataset import parse_config
from grinding_physic_fusion.models.architectures import model_factory
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from scripts.train_and_evaluate import CVSplitter, scale_data_dict, smart_load_data

PublicationPlotter.set_style()

DEVICE = torch.device("cpu")
N_MC = 50
Z = 1.96
N_FOLDS = 16
REPEAT = 0
OUT_DIR = Path("reports/evidence/uncertainty")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_spec_key(config: str) -> str:
    parts = parse_config(config)
    for p in parts:
        if "spec" in p or "wst" in p:
            return p
    raise ValueError(f"No spectrogram/WST key found in config {config}")


def run_fold(full_data, model_name: str, config: str, spec_key: str, fold: int):
    groups = full_data["condition_ids"]
    splitter = CVSplitter(n_folds=N_FOLDS, grouped=False, logo=True, seed=42)
    train_idx = test_idx = None
    for r, f, tr, val, te in splitter.split(groups):
        if f == fold and r == REPEAT:
            train_idx, test_idx = tr, te
            break
    if train_idx is None:
        raise ValueError(f"Fold {fold} not found")

    scaled, target_scaler = scale_data_dict(
        full_data, train_idx, scale_specs=False, scale_target=True
    )
    X_test = torch.from_numpy(scaled[spec_key][test_idx]).float().to(DEVICE)
    y_test = scaled["targets"][test_idx]
    cond_ids = scaled["condition_ids"][test_idx]

    safe_cfg = config.replace("+", "_")
    ckpt_path = Path(f"checkpoints/{model_name}_{safe_cfg}_fold{fold}_repeat{REPEAT}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Required deterministic checkpoint is missing: {ckpt_path}")
    model = model_factory(model_name)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    def to_um(arr: np.ndarray) -> np.ndarray:
        if target_scaler is None:
            return arr
        return target_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)

    with torch.no_grad():
        deterministic_pred = to_um(model(X_test).squeeze(-1).cpu().numpy())

    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()

    torch.manual_seed(42 + fold)
    preds = []
    with torch.no_grad():
        for _ in range(N_MC):
            out = model(X_test).squeeze(-1).cpu().numpy()
            preds.append(to_um(out))
    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    y_test_um = to_um(y_test)
    df = pd.DataFrame({
        "condition_id": cond_ids,
        "y_true": y_test_um,
        "y_pred_deterministic": deterministic_pred,
        "y_pred": mean_pred,
        "y_std": std_pred,
        "lower": mean_pred - Z * std_pred,
        "upper": mean_pred + Z * std_pred,
    })
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["interval_width"] = df["upper"] - df["lower"]
    df["checkpoint"] = str(ckpt_path)
    df["checkpoint_sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    df["fold"] = fold
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="MC-dropout uncertainty for a CNN")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model
    config = args.config
    config_display = {"vib_spec": "Vib-dB", "ae_spec": "AE-dB"}.get(config, config)
    spec_key = get_spec_key(config)
    safe_cfg = config.replace("+", "_")

    print(f"Loading data for {model_name} / {config} ...", flush=True)
    full_data = smart_load_data([model_name], [config])

    dfs = []
    for fold in range(N_FOLDS):
        print(f"Fold {fold} ...", flush=True)
        df = run_fold(full_data, model_name, config, spec_key, fold)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    csv_path = OUT_DIR / f"mc_dropout_{model_name}_{safe_cfg}_logo_all.csv"
    full.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

    overall = {
        "model": model_name,
        "config": config,
        "n_samples": len(full),
        "coverage_95": float(full["covered"].mean()),
        "mean_interval_width": float(full["interval_width"].mean()),
        "median_interval_width": float(full["interval_width"].median()),
        "mean_abs_error": float(full["abs_error"].mean()),
        "correlation_std_mae": float(full["y_std"].corr(full["abs_error"])),
    }
    pd.DataFrame([overall]).to_csv(
        OUT_DIR / f"mc_dropout_summary_{model_name}_{safe_cfg}_logo_all.csv", index=False
    )
    print(overall, flush=True)

    per_condition = full.groupby("condition_id").agg(
        n=("y_true", "count"),
        coverage=("covered", "mean"),
        mean_width=("interval_width", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()
    per_condition.to_csv(
        OUT_DIR / f"mc_dropout_per_condition_{model_name}_{safe_cfg}_logo_all.csv", index=False
    )
    print(per_condition, flush=True)

    sort_df = full.sort_values("y_pred").reset_index(drop=True)
    intervals_figure = MutableFigure(f"mc_dropout_intervals_{model_name}_{safe_cfg}.png", profile=FigureProfiles.DOUBLE, out_dir=OUT_DIR, metadata={"generator": "scripts/mc_dropout_generic.py"})
    fig, ax = intervals_figure.create()
    x = np.arange(len(sort_df))
    ax.fill_between(x, sort_df["lower"], sort_df["upper"], color=PublicationPalette.UNCERTAINTY, alpha=0.3, label="Nominal 95% MC-dropout epistemic band")
    ax.scatter(x, sort_df["y_true"], color=PublicationPalette.OBSERVED, s=15, zorder=3, label="Observed $R_a$")
    ax.set_xlabel("Sample index (sorted by predicted mean)")
    ax.set_ylabel(r"$R_a$ (µm)")
    ax.set_title(f"MC-dropout epistemic intervals: full LOGO ({model_name} / {config_display})")
    ax.legend()
    fig.tight_layout()
    intervals_figure.save()

    calibration_figure = MutableFigure(f"mc_dropout_calibration_{model_name}_{safe_cfg}_full_logo.png", profile=FigureProfiles.SINGLE, out_dir=OUT_DIR, metadata={"generator": "scripts/mc_dropout_generic.py"})
    fig, ax = calibration_figure.create()
    colors = np.where(full["covered"], PublicationPalette.MODEL_FAMILY["LightGBMModel"], PublicationPalette.CONDITION_7)
    ax.scatter(full["y_std"], full["abs_error"], c=colors, alpha=0.6, s=30)
    ax.set_xlabel("MC-dropout standard deviation (µm)")
    ax.set_ylabel("Absolute error (µm)")
    ax.set_title(f"MC-dropout uncertainty calibration\n{model_name} / {config_display} / full LOGO")
    max_std = full["y_std"].max()
    ax.plot([0, max_std], [0, max_std], "r--", lw=1, label="error = std")
    ax.plot([0, max_std], [0, 1.96 * max_std], "k:", lw=1, label="error = 1.96 std")
    ax.legend()
    fig.tight_layout()
    calibration_figure.save()
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
