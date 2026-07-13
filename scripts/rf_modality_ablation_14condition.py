#!/usr/bin/env python3
"""Matched 14-condition RF modality ablation for the current dB-z pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_and_evaluate import CVSplitter, smart_load_data  # noqa: E402

CONFIG = "ae_logspec+vib_logspec"
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


def flatten(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate([array.reshape(len(array), -1) for array in arrays], axis=1)


def main() -> int:
    data = smart_load_data(["RandomForestModel"], [CONFIG])
    y = np.asarray(data["targets"])
    groups = np.asarray(data["condition_ids"])
    matrices = {
        "AE only": flatten(np.asarray(data["ae_logspec"])),
        "Vibration only": flatten(np.asarray(data["vib_logspec"])),
        "AE + vibration": flatten(np.asarray(data["ae_logspec"]), np.asarray(data["vib_logspec"])),
    }
    splitter = CVSplitter(n_folds=16, logo=True, seed=42)
    rows: list[dict] = []
    predictions: list[dict] = []
    for name, X in matrices.items():
        for _, fold, train, val, test in splitter.split(groups):
            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            mae = float(mean_absolute_error(y[test], pred))
            rows.append({
                "input": name,
                "test_condition": fold + 1,
                "validation_condition": int(np.unique(groups[val])[0]),
                "n_training_conditions": int(len(np.unique(groups[train]))),
                "mae": mae,
            })
            predictions.extend({
                "input": name,
                "sample_index": int(index),
                "condition": fold + 1,
                "measured_ra": float(target),
                "predicted_ra": float(estimate),
            } for index, target, estimate in zip(test, y[test], pred))
            print(f"{name}, condition {fold + 1:02d}: MAE={mae:.6f}", flush=True)

    evidence = ROOT / "reports" / "evidence"
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("input", sort=False).mae
        .agg(mean_mae="mean", fold_sd=lambda values: values.std(ddof=0), median_mae="median", maximum_mae="max")
        .reset_index()
    )
    folds.to_csv(evidence / "tables" / "rf_modality_ablation_14condition_folds.csv", index=False)
    summary.to_csv(evidence / "tables" / "rf_modality_ablation_14condition.csv", index=False)
    pd.DataFrame(predictions).to_csv(evidence / "predictions" / "rf_modality_ablation_14condition.csv", index=False)
    metadata = {
        "protocol": "current fixed dB-z RF; canonical outer LOGO test and validation conditions; 14 fitting conditions",
        "cache": CONFIG,
        "rf_parameters": RF_PARAMS,
        "results": summary.to_dict(orient="records"),
    }
    (evidence / "metrics" / "rf_modality_ablation_14condition.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
