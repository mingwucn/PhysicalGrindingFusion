#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
Comprehensive sensitivity suite for the recommended random-forest configuration.

Outputs written to reports/evidence/tables/:
    - cv_scheme_comparison.csv
    - rf_calibration_strategies.csv
    - rf_modality_ablation.csv
    - rf_frequency_ablation.csv
    - sensitivity_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import (
    INTERMEDIATE_DIR,
    get_roughness_for_sample,
    load_process_parameters,
    load_surface_roughness,
)

OUT_DIR = ROOT / "reports" / "evidence" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def flatten_sample(ae: np.ndarray, vib: np.ndarray | None) -> np.ndarray:
    parts = []
    if ae is not None:
        parts.append(ae.reshape(-1))
    if vib is not None:
        parts.append(vib.reshape(-1))
    return np.concatenate(parts) if parts else np.array([])


def rf_fit_predict(X_train, y_train, X_test, seed: int = 42):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return model, scaler, y_pred


def tree_std(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    preds = np.array([tree.predict(X) for tree in model.estimators_])
    return preds.std(axis=0)


def load_data():
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
    ae = alt_cache["ae_logspec"]
    vib = alt_cache["vib_logspec"]
    mean_cache.close()
    alt_cache.close()
    return condition_ids, sample_ids, y, ae, vib, params


# ---------------------------------------------------------------------------
# A. CV scheme comparison
# ---------------------------------------------------------------------------
def cv_scheme_comparison(condition_ids, y, X):
    records = []

    # 1. Standard 16-fold LOGO
    preds = np.zeros_like(y)
    for c in range(1, 17):
        train = condition_ids != c
        test = condition_ids == c
        _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
        preds[test] = y_pred
    records.append({"scheme": "LOGO (16 conditions)", "folds": 16, "mae": float(np.abs(preds - y).mean())})

    # 2. LOGO by workpiece speed
    params = load_process_parameters()
    vw_levels = np.unique(params[:, 1])
    preds = np.zeros_like(y)
    for level in vw_levels:
        test_conditions = np.where(params[:, 1] == level)[0] + 1
        train = ~np.isin(condition_ids, test_conditions)
        test = np.isin(condition_ids, test_conditions)
        _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
        preds[test] = y_pred
    records.append({"scheme": "LOGO by workpiece speed", "folds": len(vw_levels), "mae": float(np.abs(preds - y).mean())})

    # 3. LOGO by wheel speed
    vs_levels = np.unique(params[:, 0])
    preds = np.zeros_like(y)
    for level in vs_levels:
        test_conditions = np.where(params[:, 0] == level)[0] + 1
        train = ~np.isin(condition_ids, test_conditions)
        test = np.isin(condition_ids, test_conditions)
        _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
        preds[test] = y_pred
    records.append({"scheme": "LOGO by wheel speed", "folds": len(vs_levels), "mae": float(np.abs(preds - y).mean())})

    # 4. LOGO by grinding depth
    ap_levels = np.unique(params[:, 2])
    preds = np.zeros_like(y)
    for level in ap_levels:
        test_conditions = np.where(params[:, 2] == level)[0] + 1
        train = ~np.isin(condition_ids, test_conditions)
        test = np.isin(condition_ids, test_conditions)
        _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
        preds[test] = y_pred
    records.append({"scheme": "LOGO by grinding depth", "folds": len(ap_levels), "mae": float(np.abs(preds - y).mean())})

    # 5. Random 5-fold CV (non-grouped), repeated with 5 seeds
    maes = []
    for seed in range(42, 47):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_maes = []
        for train_idx, test_idx in kf.split(X):
            _, _, y_pred = rf_fit_predict(X[train_idx], y[train_idx], X[test_idx], seed=seed)
            fold_maes.append(float(np.abs(y[test_idx] - y_pred).mean()))
        maes.append(np.mean(fold_maes))
    records.append({"scheme": "Random 5-fold CV (mean of 5 seeds)", "folds": 5, "mae": float(np.mean(maes))})

    # 6. Random 80/20 split, 10 seeds
    maes = []
    for seed in range(42, 52):
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for train_idx, test_idx in ss.split(X):
            _, _, y_pred = rf_fit_predict(X[train_idx], y[train_idx], X[test_idx], seed=seed)
            maes.append(float(np.abs(y[test_idx] - y_pred).mean()))
    records.append({"scheme": "Random 80/20 split (mean of 10 seeds)", "folds": 10, "mae": float(np.mean(maes))})

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "cv_scheme_comparison.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# B. RF calibration strategies
# ---------------------------------------------------------------------------
def oob_residuals_and_std(model: RandomForestRegressor, X: np.ndarray):
    n_samples = X.shape[0]
    n_est = len(model.estimators_)
    all_preds = np.full((n_samples, n_est), np.nan, dtype=np.float32)
    for t, tree in enumerate(model.estimators_):
        in_bag = set(model.estimators_samples_[t])
        oob_idx = np.array([i for i in range(n_samples) if i not in in_bag])
        if len(oob_idx) == 0:
            continue
        all_preds[oob_idx, t] = tree.predict(X[oob_idx])
    mean = np.nanmean(all_preds, axis=1)
    std = np.nanstd(all_preds, axis=1)
    return mean, std


def calibration_strategies(condition_ids, y, X):
    params = load_process_parameters()
    params_norm = (params - params.min(axis=0)) / (params.max(axis=0) - params.min(axis=0) + 1e-8)

    strategies = []

    for c in range(1, 17):
        train = condition_ids != c
        test = condition_ids == c
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        cond_train_ids = condition_ids[train]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        sigma_test = tree_std(model, X_test_s)

        # 1. Global OOB scaled interval
        oob_mean, oob_std = oob_residuals_and_std(model, X_train_s)
        abs_scaled = np.abs(y_train - oob_mean) / (oob_std + 1e-8)
        q_global = float(np.quantile(abs_scaled[np.isfinite(abs_scaled)], 0.95))
        cov = float(((y_test >= y_pred - q_global * sigma_test) & (y_test <= y_pred + q_global * sigma_test)).mean())
        strategies.append({"test_condition": c, "strategy": "Global OOB scaled", "coverage": cov,
                           "mean_width": float((2 * q_global * sigma_test).mean()), "q": q_global})

        # 2. Condition-aware k-NN (3 nearest training conditions in param space)
        dists = [(i, float(np.linalg.norm(params_norm[i - 1] - params_norm[c - 1]))) for i in range(1, 17) if i != c]
        dists.sort(key=lambda x: x[1])
        neighbors = [i for i, _ in dists[:3]]
        mask = np.isin(cond_train_ids, neighbors)
        q_knn = float(np.quantile(abs_scaled[mask & np.isfinite(abs_scaled)], 0.95))
        cov = float(((y_test >= y_pred - q_knn * sigma_test) & (y_test <= y_pred + q_knn * sigma_test)).mean())
        strategies.append({"test_condition": c, "strategy": "Condition-aware k-NN", "coverage": cov,
                           "mean_width": float((2 * q_knn * sigma_test).mean()), "q": q_knn})

        # 3. Split conformal with one held-out calibration condition (nearest neighbor)
        calib_cond = neighbors[0]
        train2 = train & (condition_ids != calib_cond)
        calib = condition_ids == calib_cond
        X_train2, X_calib = X[train2], X[calib]
        y_train2, y_calib = y[train2], y[calib]

        scaler2 = StandardScaler()
        X_train2_s = scaler2.fit_transform(X_train2)
        X_calib_s = scaler2.transform(X_calib)
        X_test2_s = scaler2.transform(X_test)
        model2 = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        model2.fit(X_train2_s, y_train2)
        y_pred2 = model2.predict(X_test2_s)
        y_calib_pred = model2.predict(X_calib_s)
        sigma_calib = tree_std(model2, X_calib_s)
        sigma_test2 = tree_std(model2, X_test2_s)
        abs_scaled_calib = np.abs(y_calib - y_calib_pred) / (sigma_calib + 1e-8)
        q_split = float(np.quantile(abs_scaled_calib, 0.95))
        cov = float(((y_test >= y_pred2 - q_split * sigma_test2) & (y_test <= y_pred2 + q_split * sigma_test2)).mean())
        strategies.append({"test_condition": c, "strategy": "Split conformal (1 calib condition)", "coverage": cov,
                           "mean_width": float((2 * q_split * sigma_test2).mean()), "q": q_split})

        # 4. Naive residual interval (pooled 95th percentile, no sigma scaling)
        q_naive = float(np.quantile(np.abs(y_train - oob_mean), 0.95))
        cov = float(((y_test >= y_pred - q_naive) & (y_test <= y_pred + q_naive)).mean())
        strategies.append({"test_condition": c, "strategy": "Naive residual 95%", "coverage": cov,
                           "mean_width": float(2 * q_naive), "q": q_naive})

    df = pd.DataFrame(strategies)
    df.to_csv(OUT_DIR / "rf_calibration_strategies.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# C. Modality ablation
# ---------------------------------------------------------------------------
def modality_ablation(condition_ids, y, ae, vib):
    configs = {
        "AE only": (ae, None),
        "Vib only": (None, vib),
        "AE + Vib": (ae, vib),
    }
    records = []
    for name, (a, v) in configs.items():
        X = np.array([flatten_sample(a[i] if a is not None else None, v[i] if v is not None else None) for i in range(len(y))])
        preds = np.zeros_like(y)
        for c in range(1, 17):
            train = condition_ids != c
            test = condition_ids == c
            _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
            preds[test] = y_pred
        records.append({"config": name, "mae": float(np.abs(preds - y).mean())})
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "rf_modality_ablation.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# D. Frequency-band ablation
# ---------------------------------------------------------------------------
def frequency_ablation(condition_ids, y, ae, vib):
    # ae_logspec shape (n, 2, 300, 47); 1 MHz corresponds to bin 150
    # vib_logspec shape (n, 3, 257, 13); 2 kHz -> bin 20, 15 kHz -> bin 150
    ae_no_high = ae.copy()
    ae_no_high[:, :, 151:, :] = 0.0
    vib_no_chatter = vib.copy()
    vib_no_chatter[:, :, 20:151, :] = 0.0
    ae_both = ae_no_high.copy()
    vib_both = vib_no_chatter.copy()

    configs = {
        "Full AE + Vib": (ae, vib),
        "AE >1 MHz removed": (ae_no_high, vib),
        "Vib 2-15 kHz removed": (ae, vib_no_chatter),
        "Both removed": (ae_both, vib_both),
    }
    records = []
    for name, (a, v) in configs.items():
        X = np.array([flatten_sample(a[i], v[i]) for i in range(len(y))])
        preds = np.zeros_like(y)
        for c in range(1, 17):
            train = condition_ids != c
            test = condition_ids == c
            _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
            preds[test] = y_pred
        records.append({"config": name, "mae": float(np.abs(preds - y).mean())})
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "rf_frequency_ablation.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    condition_ids, sample_ids, y, ae, vib, params = load_data()
    X = np.array([flatten_sample(ae[i], vib[i]) for i in range(len(y))])

    print("Running CV scheme comparison ...")
    df_cv = cv_scheme_comparison(condition_ids, y, X)
    print(df_cv.to_string(index=False))

    print("\nRunning calibration strategies ...")
    df_cal = calibration_strategies(condition_ids, y, X)
    summary_cal = df_cal.groupby("strategy").agg({"coverage": "mean", "mean_width": "mean"}).reset_index()
    print(summary_cal.to_string(index=False))

    print("\nRunning modality ablation ...")
    df_mod = modality_ablation(condition_ids, y, ae, vib)
    print(df_mod.to_string(index=False))

    print("\nRunning frequency ablation ...")
    df_freq = frequency_ablation(condition_ids, y, ae, vib)
    print(df_freq.to_string(index=False))

    # Write Markdown summary
    with open(OUT_DIR / "sensitivity_summary.md", "w") as f:
        f.write("# Comprehensive sensitivity summary\n\n")
        f.write("## Cross-validation schemes (RF ae_logspec + vib_logspec)\n\n")
        f.write(df_cv.to_markdown(index=False))
        f.write("\n\n## RF uncertainty calibration strategies\n\n")
        f.write(summary_cal.to_markdown(index=False))
        f.write("\n\nPer-condition coverage:\n\n")
        f.write(df_cal.to_markdown(index=False))
        f.write("\n\n## Modality ablation (RF)\n\n")
        f.write(df_mod.to_markdown(index=False))
        f.write("\n\n## Frequency-band ablation (RF)\n\n")
        f.write(df_freq.to_markdown(index=False))
        f.write("\n")

    print("\nDone. Outputs in", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
