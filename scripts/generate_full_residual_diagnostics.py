#!/usr/bin/env python3
"""Generate residual diagnostics for the recommended RF model."""
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

from grinding_physic_fusion.data.dataset import load_process_parameters
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

TABLES_DIR = ROOT / "reports" / "evidence" / "tables"
PRED_DIR = ROOT / "reports" / "evidence" / "predictions"
PLOTS_DIR = ROOT / "reports" / "evidence" / "plots" / "supp"
OVERLEAF_DIR = ROOT / "overleaf" / "images"


def load_predictions() -> pd.DataFrame:
    run = pd.read_csv(TABLES_DIR / "supp_run_order.csv")
    rms = pd.read_csv(TABLES_DIR / "supp_runorder_diagnostics.csv")
    params = load_process_parameters()
    rows: list[pd.DataFrame] = []

    for condition in range(1, 17):
        path = PRED_DIR / f"RandomForestModel_ae_logspec_vib_logspec_fold{condition - 1}_repeat0.csv"
        pred = pd.read_csv(path)
        meta = run[run["condition_id"] == condition].sort_values("sample_id").reset_index(drop=True)
        if len(pred) != len(meta):
            matched_rows = []
            used: set[int] = set()
            for y_true in pred["y_true"].to_numpy():
                diffs = (meta["Ra"] - y_true).abs()
                for idx in diffs.sort_values().index:
                    if int(idx) not in used:
                        used.add(int(idx))
                        matched_rows.append(meta.loc[idx])
                        break
            meta = pd.DataFrame(matched_rows).reset_index(drop=True)
            if len(pred) != len(meta):
                raise ValueError(f"Fold {condition}: {len(pred)} predictions for {len(meta)} metadata rows")

        pred = pred.copy()
        pred["condition_id"] = condition
        pred["sample_id"] = meta["sample_id"].to_numpy()
        pred["pass_order"] = meta["pass_order"].to_numpy()
        pred["wheel_speed"] = params[condition - 1, 0]
        pred["workpiece_speed"] = params[condition - 1, 1]
        pred["depth_of_cut"] = params[condition - 1, 2]

        train_y = run.loc[run["condition_id"] != condition, "Ra"].to_numpy()
        train_min = train_y.min()
        train_max = train_y.max()
        below = np.maximum(train_min - pred["y_true"].to_numpy(), 0)
        above = np.maximum(pred["y_true"].to_numpy() - train_max, 0)
        pred["training_target_support_distance"] = below + above
        rows.append(pred)

    out = pd.concat(rows, ignore_index=True)
    out["residual"] = out["y_pred"] - out["y_true"]
    out["abs_residual"] = out["residual"].abs()
    out = out.merge(rms, on=["condition_id", "sample_id", "pass_order"], how="left")
    return out


def main() -> int:
    PublicationPlotter.set_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions()
    out_csv = TABLES_DIR / "rf_full_residual_diagnostics.csv"
    df.to_csv(out_csv, index=False)

    panels = [
        ("y_true", "Measured $R_a$ (µm)"),
        ("y_pred", "Predicted $R_a$ (µm)"),
        ("wheel_speed", "Wheel speed (m/s)"),
        ("workpiece_speed", "Workpiece speed (r/min)"),
        ("depth_of_cut", "Depth of cut (µm)"),
        ("pass_order", "Run order"),
        ("ae_rms", "AE RMS"),
        ("vib_rms", "Vibration RMS"),
        ("training_target_support_distance", "Training-target support distance (µm)"),
    ]

    managed = MutableFigure(
        "supp_full_residual_diagnostics.png",
        profile=FigureProfiles.DIAGNOSTIC_GRID,
        out_dir=PLOTS_DIR,
        overleaf_dir=OVERLEAF_DIR,
        metadata={"generator": "scripts/generate_full_residual_diagnostics.py"},
    )
    fig, axes = managed.create()
    for ax, (col, label) in zip(axes.ravel(), panels):
        colors = np.where(df["condition_id"].to_numpy() == 7, PublicationPalette.CONDITION_7, PublicationPalette.OBSERVED)
        ax.scatter(df[col], df["residual"], s=9, c=colors, alpha=0.75, linewidths=0)
        ax.axhline(0, color="black", lw=0.7, linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Residual (µm)")
    fig.suptitle("Recommended RF residual diagnostics", y=1.01)
    fig.tight_layout()
    managed.save()
    print(f"Saved {out_csv}")
    print(f"Saved {PLOTS_DIR / 'supp_full_residual_diagnostics.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
