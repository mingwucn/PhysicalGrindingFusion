#!/usr/bin/env python3
"""RF band ablation that recomputes per-sample z-scores after band deletion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_and_evaluate import CVSplitter  # noqa: E402


def retained_zscore(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    retained = x[:, :, keep, :]
    mean = retained.mean(axis=(2, 3), keepdims=True)
    std = retained.std(axis=(2, 3), keepdims=True)
    return ((retained - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def flatten(ae: np.ndarray, vib: np.ndarray) -> np.ndarray:
    return np.concatenate((ae.reshape(len(ae), -1), vib.reshape(len(vib), -1)), axis=1)


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = []
    for _, _, train, _, test in CVSplitter(n_folds=16, logo=True, seed=42).split(groups):
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X[train], y[train])
        values.append(np.abs(y[test] - model.predict(X[test])).mean())
    return np.asarray(values)


def paired_bootstrap_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    cache_dir = ROOT / "data" / "intermediate" / "cached_specs"
    with np.load(cache_dir / "mean_specs.npz", allow_pickle=True) as data:
        ae_db, vib_db = np.asarray(data["ae_spec"]), np.asarray(data["vib_spec"])
        y, groups = np.asarray(data["targets"]), np.asarray(data["condition_ids"])
    ae_all, vib_all = np.ones(ae_db.shape[2], bool), np.ones(vib_db.shape[2], bool)
    vib_gt15 = np.arange(vib_db.shape[2]) <= 150
    vib_16_24 = ~((np.arange(vib_db.shape[2]) >= 160) & (np.arange(vib_db.shape[2]) <= 240))
    variants = {
        "Full dB-z recomputed from dB": (ae_all, vib_all),
        "Remove Vib >15 kHz before z-score": (ae_all, vib_gt15),
        "Remove Vib 16--24 kHz before z-score": (ae_all, vib_16_24),
    }
    rows, folds = [], []
    results = {}
    for name, (ae_keep, vib_keep) in variants.items():
        maes = evaluate(flatten(retained_zscore(ae_db, ae_keep), retained_zscore(vib_db, vib_keep)), y, groups)
        results[name] = maes
        rows.append({"variant": name, "mean_mae": maes.mean(), "fold_sd": maes.std(), "maximum_fold_mae": maes.max()})
        folds.extend({"variant": name, "condition": i + 1, "mae": value} for i, value in enumerate(maes))
        print(f"{name}: {maes.mean():.6f} +/- {maes.std():.6f}", flush=True)
    baseline = results["Full dB-z recomputed from dB"]
    for row in rows:
        difference = results[row["variant"]] - baseline
        row["mean_difference_vs_full"] = float(difference.mean())
        row["difference_ci95_low"], row["difference_ci95_high"] = paired_bootstrap_ci(difference)
        row["protocol"] = "band deleted from dB cache before per-sample/per-channel z-score; RF refitted on 14 conditions"
    out = ROOT / "reports" / "evidence" / "tables"
    pd.DataFrame(rows).to_csv(out / "pre_normalization_band_ablation.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "pre_normalization_band_ablation_folds.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
