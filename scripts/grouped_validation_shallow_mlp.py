#!/usr/bin/env python3
"""Fair shallow-MLP LOGO baseline using the designated validation condition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_and_evaluate import CVSplitter, smart_load_data  # noqa: E402

CONFIG = "ae_features+vib_features+pp"
GRID = (
    {"hidden_layer_sizes": (64,), "alpha": 1e-3, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-2, "learning_rate_init": 5e-4},
)
MAX_EPOCHS = 500
PATIENCE = 30


def matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([data[key].reshape(len(data[key]), -1) for key in CONFIG.split("+")], axis=1)


def fit_candidate(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: dict) -> tuple[MLPRegressor, float, int]:
    model = MLPRegressor(
        **params,
        activation="relu",
        solver="adam",
        batch_size=min(32, len(y_train)),
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        random_state=42,
    )
    best_state: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    best_mae = np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.fit(X_train, y_train)
        score = mean_absolute_error(y_val, model.predict(X_val))
        if score < best_mae - 1e-6:
            best_mae, best_epoch, stale = float(score), epoch, 0
            best_state = ([v.copy() for v in model.coefs_], [v.copy() for v in model.intercepts_])
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    assert best_state is not None
    model.coefs_, model.intercepts_ = best_state
    return model, best_mae, best_epoch


def main() -> int:
    data = smart_load_data(["ShallowMLPModel"], [CONFIG])
    X, y, groups = matrix(data), np.asarray(data["targets"]), np.asarray(data["condition_ids"])
    rows: list[dict] = []
    predictions: list[dict] = []
    splitter = CVSplitter(n_folds=16, logo=True, seed=42)
    for _, fold, train, val, test in splitter.split(groups):
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        X_train = x_scaler.fit_transform(X[train])
        X_val, X_test = x_scaler.transform(X[val]), x_scaler.transform(X[test])
        y_train = y_scaler.fit_transform(y[train, None]).ravel()
        y_val = y_scaler.transform(y[val, None]).ravel()
        fitted = [fit_candidate(X_train, y_train, X_val, y_val, params) + (params,) for params in GRID]
        model, val_mae_scaled, epoch, params = min(fitted, key=lambda item: item[1])
        pred = y_scaler.inverse_transform(model.predict(X_test)[:, None]).ravel()
        mae = float(mean_absolute_error(y[test], pred))
        rows.append({
            "test_condition": fold + 1,
            "validation_condition": int(np.unique(groups[val])[0]),
            "n_training_conditions": 14,
            "mae": mae,
            "validation_mae_scaled": val_mae_scaled,
            "selected_epoch": epoch,
            **{key: str(value) for key, value in params.items()},
        })
        predictions.extend({"sample_index": int(i), "condition": fold + 1, "measured_ra": float(yt), "predicted_ra": float(yp)} for i, yt, yp in zip(test, y[test], pred))
        print(f"Condition {fold + 1:02d}: MAE={mae:.5f}, epoch={epoch}, params={params}", flush=True)

    out = ROOT / "reports" / "evidence"
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables" / "grouped_validation_shallow_mlp.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions" / "grouped_validation_shallow_mlp.csv", index=False)
    summary = {
        "protocol": "designated whole-condition validation; train-only input and target scaling; three-candidate training-only grid",
        "config": CONFIG,
        "mean_mae": float(table.mae.mean()),
        "fold_sd": float(table.mae.std(ddof=0)),
        "median_mae": float(table.mae.median()),
        "maximum_mae": float(table.mae.max()),
        "grid": [{**p, "hidden_layer_sizes": list(p["hidden_layer_sizes"])} for p in GRID],
    }
    (out / "metrics" / "grouped_validation_shallow_mlp.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
