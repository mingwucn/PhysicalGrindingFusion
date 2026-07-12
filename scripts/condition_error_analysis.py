#!/usr/bin/env python3
"""
Condition-level error analysis.

Regenerates the LOGO splits used during training, aligns held-out predictions
with condition IDs, and computes MAE per grinding condition.

Outputs:
    reports/evidence/tables/condition_error_ranking.csv
    reports/evidence/tables/condition_best_model.csv
    reports/evidence/plots/condition_error_heatmap.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.data.dataset import load_all_data
from train_and_evaluate import CVSplitter

METRICS_DIR = ROOT / "reports" / "evidence" / "metrics"
PREDICTIONS_DIR = ROOT / "reports" / "evidence" / "predictions"
TABLES_DIR = ROOT / "reports" / "evidence" / "tables"
PLOTS_DIR = ROOT / "reports" / "evidence" / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def config_from_stem(stem: str, model: str) -> str:
    """cv_results_<Model>_<config> -> config with '+' restored."""
    prefix = f"cv_results_{model}_"
    if not stem.startswith(prefix):
        return ""
    return stem[len(prefix):].replace("_", "+")


def main() -> int:
    # Use any config to load condition_ids (same across configs)
    sample_json = next(METRICS_DIR.glob("cv_results_*.json"))
    with open(sample_json) as f:
        sample_data = json.load(f)
    sample_model = sample_data.get("model_name", sample_json.stem.split("_", 2)[1] if len(sample_json.stem.split("_", 2)) >= 2 else "")
    config = sample_data.get("config") or config_from_stem(sample_json.stem, sample_model)
    data_dict = load_all_data(config=config)
    groups = data_dict["condition_ids"]
    splitter = CVSplitter(n_folds=16, grouped=False, logo=True, seed=42)

    records: List[dict] = []

    for p in sorted(METRICS_DIR.glob("cv_results_*.json")):
        with open(p) as f:
            data = json.load(f)
        model = data.get("model_name", "")
        config = data.get("config") or config_from_stem(p.stem, model)
        if not config:
            continue
        cfg_file = config.replace("+", "_")

        for repeat_idx, fold_idx, train_idx, val_idx, test_idx in splitter.split(groups):
            pred_csv = PREDICTIONS_DIR / f"{model}_{cfg_file}_fold{fold_idx}_repeat{repeat_idx}.csv"
            if not pred_csv.exists():
                continue
            df = pd.read_csv(pred_csv)
            df["abs_error"] = np.abs(df["y_pred"] - df["y_true"])
            cond_ids = groups[test_idx]
            if len(df) != len(test_idx):
                print(f"Length mismatch for {pred_csv}: {len(df)} vs {len(test_idx)}")
                continue
            for i, cond in enumerate(cond_ids):
                records.append({
                    "model": model,
                    "config": config,
                    "fold": fold_idx,
                    "repeat": repeat_idx,
                    "condition_id": int(cond),
                    "mae": float(df["abs_error"].iloc[i]),
                })

    df = pd.DataFrame(records)
    if df.empty:
        print("No prediction records found")
        return 1

    # Aggregate per model-config-condition across folds/repeats
    grouped = df.groupby(["model", "config", "condition_id"])["mae"].mean().reset_index()

    condition_mean = grouped.groupby("condition_id")["mae"].mean().reset_index().sort_values("mae", ascending=False)
    condition_mean.to_csv(TABLES_DIR / "condition_error_ranking.csv", index=False)

    best_per_condition = grouped.loc[grouped.groupby("condition_id")["mae"].idxmin()]
    best_per_condition.to_csv(TABLES_DIR / "condition_best_model.csv", index=False)

    # Heatmap: top 20 configs by overall MAE × conditions
    config_means = grouped.groupby(["model", "config"])["mae"].mean().sort_values().head(20)
    top_cols = list(config_means.index)
    pivot = grouped.pivot(index="condition_id", columns=["model", "config"], values="mae")
    pivot_top = pivot[[c for c in top_cols if c in pivot.columns]]

    if not pivot_top.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_top.T, cmap="YlOrRd", annot=False, ax=ax, cbar_kws={"label": "MAE (µm)"})
        ax.set_title("Condition-level MAE heatmap (top 20 configs)")
        ax.set_xlabel("Condition ID")
        ax.set_ylabel("Model / Config")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "condition_error_heatmap.png", dpi=300)
        plt.close(fig)

    print(f"Wrote {TABLES_DIR / 'condition_error_ranking.csv'}")
    print(f"Wrote {TABLES_DIR / 'condition_best_model.csv'}")
    print(f"Wrote {PLOTS_DIR / 'condition_error_heatmap.png'}")
    print("\nTop 10 hardest conditions:")
    print(condition_mean.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
