"""MC-dropout uncertainty for a trained ResNetVibCNN on vib_spec."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from scripts.train_and_evaluate import smart_load_data, CVSplitter, scale_data_dict
from grinding_physic_fusion.models.architectures import model_factory

PublicationPlotter.set_style()

CONFIG = "vib_spec"
MODEL_NAME = "ResNetVibCNN"
FOLD = 0
REPEAT = 0
CKPT_PATH = Path(f"checkpoints/ResNetVibCNN_{CONFIG}_fold{FOLD}_repeat{REPEAT}.pt")
OUT_DIR = Path("reports/evidence/uncertainty")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")
N_MC = 50
Z = 1.96


def main():
    print("Loading data ...", flush=True)
    full_data = smart_load_data([MODEL_NAME], [CONFIG])
    groups = full_data["condition_ids"]
    splitter = CVSplitter(n_folds=16, grouped=False, logo=True, seed=42)
    gen = splitter.split(groups)
    for r, f, train_idx, val_idx, test_idx in gen:
        if f == FOLD and r == REPEAT:
            break

    print("Scaling ...", flush=True)
    scaled, _ = scale_data_dict(full_data, train_idx, scale_specs=True, scale_target=False)
    X_test = torch.from_numpy(scaled["vib_spec"][test_idx]).float().to(DEVICE)
    y_test = scaled["targets"][test_idx]
    cond_ids = scaled["condition_ids"][test_idx]

    print("Loading model ...", flush=True)
    model = model_factory(MODEL_NAME)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.train()  # keep dropout active

    print(f"Running {N_MC} stochastic forward passes ...", flush=True)
    preds = []
    with torch.no_grad():
        for _ in range(N_MC):
            out = model(X_test).squeeze(-1).cpu().numpy()
            preds.append(out)
    preds = np.stack(preds, axis=0)  # (N_MC, n_test)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    df = pd.DataFrame({
        "condition_id": cond_ids,
        "y_true": y_test,
        "y_pred": mean_pred,
        "y_std": std_pred,
        "lower": mean_pred - Z * std_pred,
        "upper": mean_pred + Z * std_pred,
    })
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
    df["interval_width"] = df["upper"] - df["lower"]

    csv_path = OUT_DIR / f"mc_dropout_{MODEL_NAME}_{CONFIG}_fold{FOLD}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

    summary = {
        "model": MODEL_NAME,
        "config": CONFIG,
        "fold": FOLD,
        "n_samples": len(df),
        "coverage_95": float(df["covered"].mean()),
        "mean_interval_width": float(df["interval_width"].mean()),
        "median_interval_width": float(df["interval_width"].median()),
        "mean_abs_error": float(df["abs_error"].mean()),
        "correlation_std_mae": float(df["y_std"].corr(df["abs_error"])),
    }
    summary_path = OUT_DIR / f"mc_dropout_summary_{MODEL_NAME}_{CONFIG}_fold{FOLD}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}", flush=True)
    print(summary, flush=True)

    managed = MutableFigure(f"mc_dropout_calibration_{MODEL_NAME}_{CONFIG}_fold{FOLD}.png", profile=FigureProfiles.SINGLE, out_dir=OUT_DIR, metadata={"generator": "scripts/mc_dropout_uncertainty.py"})
    fig, ax = managed.create()
    ax.scatter(df["y_std"], df["abs_error"], alpha=0.6, s=30, color=PublicationPalette.OBSERVED)
    ax.set_xlabel("MC-dropout standard deviation (µm)")
    ax.set_ylabel("Absolute error (µm)")
    ax.set_title(f"MC-dropout uncertainty calibration\n{MODEL_NAME} / {CONFIG} / fold {FOLD}")
    ax.plot([0, df["y_std"].max()], [0, df["y_std"].max()], "r--", lw=1, label="perfect calibration")
    ax.legend()
    fig.tight_layout()
    managed.save()
    print("Done", flush=True)

if __name__ == "__main__":
    main()
