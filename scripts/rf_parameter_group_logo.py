#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
Parameter-group leave-one-level-out CV for the recommended random forest.

Instead of leaving out a single condition, we leave out all conditions
that share a level of workpiece speed (v_w) or grinding depth (a_p).
This tests generalisation to unseen settings of a process parameter.
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


def run_group_cv(condition_ids: np.ndarray, y: np.ndarray, X: np.ndarray, group_mask: np.ndarray, label: str):
    """Train on conditions where group_mask==False, test on where True."""
    train_idx = ~group_mask
    test_idx = group_mask
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    mae = float(np.abs(y_test - y_pred).mean())
    return {
        "group": label,
        "n_train": int(train_idx.sum()),
        "n_test": int(test_idx.sum()),
        "mae": mae,
    }


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

    rows = []

    # Group by workpiece speed (column 1)
    vw_levels = np.unique(params[:, 1])
    for level in vw_levels:
        test_conditions = np.where(params[:, 1] == level)[0] + 1  # condition IDs
        mask = np.isin(condition_ids, test_conditions)
        rows.append(run_group_cv(condition_ids, y, X, mask, f"v_w={level:.0f}"))

    # Group by grinding depth (column 2)
    ap_levels = np.unique(params[:, 2])
    for level in ap_levels:
        test_conditions = np.where(params[:, 2] == level)[0] + 1
        mask = np.isin(condition_ids, test_conditions)
        rows.append(run_group_cv(condition_ids, y, X, mask, f"a_p={level:.0f}"))

    mean_cache.close()
    alt_cache.close()

    out_dir = ROOT / "reports" / "evidence" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rf_parameter_group_logo.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nMean v_w-group MAE: {df[df['group'].str.startswith('v_w')]['mae'].mean():.4f} µm")
    print(f"Mean a_p-group MAE: {df[df['group'].str.startswith('a_p')]['mae'].mean():.4f} µm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
