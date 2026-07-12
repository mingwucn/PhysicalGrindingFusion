#!/usr/bin/env python3
"""
Pairwise statistical comparison of model/config combinations using the per-fold
MAE values stored in cv_results_*.json files.

Output: reports/evidence/tables/statistical_tests.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "reports" / "evidence" / "metrics"
PREDICTIONS_DIR = ROOT / "reports" / "evidence" / "predictions"
TABLES_DIR = ROOT / "reports" / "evidence" / "tables"


def _maes_from_predictions(stem: str) -> List[float]:
    """Compute per-fold MAE from prediction CSVs when fold records are absent."""
    maes: List[float] = []
    for csv_path in sorted(PREDICTIONS_DIR.glob(f"{stem}_fold*_repeat*.csv")):
        df = pd.read_csv(csv_path)
        mae = float(np.mean(np.abs(df["y_pred"].to_numpy() - df["y_true"].to_numpy())))
        maes.append(mae)
    if not maes:
        raise ValueError(f"No prediction CSVs found for {stem}")
    return maes


def load_fold_maes(path: Path) -> List[float]:
    with open(path, "r") as f:
        data = json.load(f)
    # Two possible schemas
    if "fold_records" in data:
        return [float(r["mae"]) for r in data["fold_records"]]
    if "folds" in data:
        return [float(r["mae"]) for r in data["folds"]]
    # Fallback: per-fold predictions on disk
    try:
        source = data.get("source_artifacts", {})
        preds = source.get("predictions", [])
        if preds:
            maes: List[float] = []
            for p in preds:
                df = pd.read_csv(p)
                mae = float(np.mean(np.abs(df["y_pred"].to_numpy() - df["y_true"].to_numpy())))
                maes.append(mae)
            return maes
    except Exception:
        pass
    # Last resort: glob predictions from filename stem
    return _maes_from_predictions(path.stem)


def find_result(model_name: str, config_contains: str = "") -> Tuple[Path, List[float]]:
    candidates = []
    for p in METRICS_DIR.glob(f"cv_results_{model_name}_*.json"):
        if config_contains and config_contains not in p.stem:
            continue
        try:
            maes = load_fold_maes(p)
        except Exception:
            continue
        candidates.append((p, maes, np.mean(maes)))
    if not candidates:
        raise FileNotFoundError(f"No result found for {model_name} {config_contains}")
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0], candidates[0][1]


def wilcoxon_report(a: List[float], b: List[float]) -> Dict[str, Any]:
    diff = np.array(a) - np.array(b)
    # Wilcoxon requires nonzero differences
    if np.all(diff == 0):
        return {"statistic": np.nan, "pvalue": 1.0, "median_diff": 0.0}
    stat, p = wilcoxon(diff, zero_method="zsplit")
    return {"statistic": float(stat), "pvalue": float(p), "median_diff": float(np.median(diff))}


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("Best DL (spectrogram)", "ResNetVibCNN", "", "Best LightGBM", "LightGBMModel", ""),
        ("Best DL (spectrogram)", "ResNetVibCNN", "", "Best DL (feature-only)", "BilinearFusionNetwork", "ae_features_vib_features_physics_pp"),
        ("Best DL (spectrogram)", "ResNetVibCNN", "", "Best ML full-input", "LightGBMModel", "ae_features_vib_features_physics_pp"),
        ("Best fusion DL", "BilinearFusionNetwork", "ae_spec_vib_spec_physics_pp", "LightGBMModel", "LightGBMModel", ""),
    ]

    rows = []
    for label_a, model_a, cfg_a, label_b, model_b, cfg_b in comparisons:
        try:
            path_a, maes_a = find_result(model_a, cfg_a)
            path_b, maes_b = find_result(model_b, cfg_b)
        except FileNotFoundError as exc:
            print(f"Skipping {label_a} vs {label_b}: {exc}")
            continue
        report = wilcoxon_report(maes_a, maes_b)
        rows.append({
            "comparison": f"{label_a} vs {label_b}",
            "model_a": model_a,
            "config_a": path_a.stem,
            "mean_mae_a": np.mean(maes_a),
            "model_b": model_b,
            "config_b": path_b.stem,
            "mean_mae_b": np.mean(maes_b),
            "median_diff": report["median_diff"],
            "wilcoxon_statistic": report["statistic"],
            "pvalue": report["pvalue"],
            "significant_05": report["pvalue"] < 0.05,
        })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "statistical_tests.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
