#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
Condition-aware conformal-ish calibration for the recommended random forest.

For each LOGO fold we train the recommended RF (ae_logspec + vib_logspec).
Using OOB predictions on the training folds we compute a per-condition
residual distribution.  For the held-out test condition we select a
calibration factor from the k-nearest training conditions in
process-parameter space (normalised wheel speed, workpiece speed, depth).
The prediction interval is  y_hat +/- q * sigma_tree, where q is the 95th
percentile of |OOB residual| / sigma_tree for the nearest conditions.

A global OOB baseline (single q across all training conditions) is also
reported for comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import (
    INTERMEDIATE_DIR,
    get_roughness_for_sample,
    load_process_parameters,
    load_surface_roughness,
)


def flatten_sample(ae: np.ndarray, vib: np.ndarray) -> np.ndarray:
    return np.concatenate([ae.reshape(-1), vib.reshape(-1)])


def oob_predictions_and_std(model: RandomForestRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return OOB mean and std for each sample using only trees where it is OOB."""
    n_samples = X.shape[0]
    n_estimators = len(model.estimators_)
    all_preds = np.full((n_samples, n_estimators), np.nan, dtype=np.float32)
    for t, tree in enumerate(model.estimators_):
        # bootstrap sample indices for this tree
        in_bag = set(model.estimators_samples_[t])
        oob_idx = np.array([i for i in range(n_samples) if i not in in_bag])
        if len(oob_idx) == 0:
            continue
        preds = tree.predict(X[oob_idx])
        all_preds[oob_idx, t] = preds
    mean = np.nanmean(all_preds, axis=1)
    std = np.nanstd(all_preds, axis=1)
    return mean, std


def tree_std_test(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """Standard deviation of tree predictions on a test set."""
    preds = np.array([tree.predict(X) for tree in model.estimators_])
    return preds.std(axis=0)


def main() -> int:
    roughness = load_surface_roughness()
    params = load_process_parameters()
    mean_cache = np.load(INTERMEDIATE_DIR / "cached_specs" / "mean_specs.npz", allow_pickle=True)
    alt_cache = np.load(INTERMEDIATE_DIR / "cached_specs" / "alternative_reps.npz", allow_pickle=True)
    condition_ids = np.asarray(mean_cache["condition_ids"])
    sample_ids = np.asarray(mean_cache["sample_ids"])
    y = np.array(
        [get_roughness_for_sample(roughness, int(cid), int(sid)) for cid, sid in zip(condition_ids, sample_ids)],
        dtype=np.float32,
    )
    X = np.array(
        [flatten_sample(ae, vib) for ae, vib in zip(alt_cache["ae_logspec"], alt_cache["vib_logspec"])]
    )

    # Normalise process parameters to [0, 1] over conditions for distance computation
    params_min = params.min(axis=0)
    params_max = params.max(axis=0)
    params_norm = (params - params_min) / (params_max - params_min + 1e-8)

    records_global = []
    records_cond = []
    records_cond_knn = []
    q_global_list = []

    for c in range(1, 17):
        train_idx = condition_ids != c
        test_idx = condition_ids == c
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1, bootstrap=True, oob_score=False
        )
        model.fit(X_train_s, y_train)

        # OOB calibration on training samples
        oob_mean, oob_std = oob_predictions_and_std(model, X_train_s)
        residuals = y_train - oob_mean
        abs_scaled = np.abs(residuals) / (oob_std + 1e-8)
        valid = np.isfinite(abs_scaled)
        abs_scaled = abs_scaled[valid]
        q_global = float(np.quantile(abs_scaled, 0.95, method="median_unbiased"))
        q_global_list.append(q_global)

        # Test predictions and tree std
        y_pred = model.predict(X_test_s)
        sigma_test = tree_std_test(model, X_test_s)

        # Global intervals
        lower_g = y_pred - q_global * sigma_test
        upper_g = y_pred + q_global * sigma_test
        covered_g = (y_test >= lower_g) & (y_test <= upper_g)
        width_g = upper_g - lower_g
        records_global.append({
            "test_condition": c,
            "coverage": float(covered_g.mean()),
            "mean_width": float(width_g.mean()),
            "mae": float(np.abs(y_test - y_pred).mean()),
        })

        # Condition-aware intervals using k-nearest training conditions in parameter space
        test_params = params_norm[c - 1]
        train_conditions = [i for i in range(1, 17) if i != c]
        dists = [(i, float(np.linalg.norm(params_norm[i - 1] - test_params))) for i in train_conditions]
        dists.sort(key=lambda x: x[1])
        k = 3
        neighbor_conditions = [i for i, _ in dists[:k]]

        # Collect scaled residuals from neighbor conditions only
        neighbor_mask = np.isin(condition_ids[train_idx], neighbor_conditions)
        if neighbor_mask.sum() < 20:
            # Fall back to all training conditions if too few samples
            neighbor_mask = np.ones_like(train_idx[train_idx], dtype=bool)
        q_cond = float(np.quantile(abs_scaled[neighbor_mask[valid]], 0.95, method="median_unbiased"))

        lower_c = y_pred - q_cond * sigma_test
        upper_c = y_pred + q_cond * sigma_test
        covered_c = (y_test >= lower_c) & (y_test <= upper_c)
        width_c = upper_c - lower_c
        records_cond_knn.append({
            "test_condition": c,
            "coverage": float(covered_c.mean()),
            "mean_width": float(width_c.mean()),
            "mae": float(np.abs(y_test - y_pred).mean()),
            "q_cond": q_cond,
            "q_global": q_global,
            "neighbors": ",".join(map(str, neighbor_conditions)),
        })

    mean_cache.close()
    alt_cache.close()

    out_dir = ROOT / "reports" / "evidence" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_global = pd.DataFrame(records_global)
    df_global.to_csv(out_dir / "rf_conformal_global.csv", index=False)
    df_cond = pd.DataFrame(records_cond_knn)
    df_cond.to_csv(out_dir / "rf_conformal_condition_aware.csv", index=False)

    print("Global OOB-like calibration")
    print(df_global.to_string(index=False))
    print(f"Overall coverage: {df_global['coverage'].mean()*100:.1f}%, mean width: {df_global['mean_width'].mean():.4f} µm")

    print("\nCondition-aware (k-NN parameter-space) calibration")
    print(df_cond.to_string(index=False))
    print(f"Overall coverage: {df_cond['coverage'].mean()*100:.1f}%, mean width: {df_cond['mean_width'].mean():.4f} µm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
