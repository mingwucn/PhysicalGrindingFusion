#!/usr/bin/env python3
"""Retrain the recommended RF after removing selected dB-z frequency bands.

Unlike the existing evaluation-time masking experiment, each variant is
trained and tested with the selected feature bins replaced by zero.  Zero is
the neutral value of the per-sample dB-z representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import INTERMEDIATE_DIR, get_roughness_for_sample, load_surface_roughness  # noqa: E402
from train_and_evaluate import CVSplitter  # noqa: E402


def flatten(ae: np.ndarray, vib: np.ndarray) -> np.ndarray:
    return np.concatenate([ae.reshape(len(ae), -1), vib.reshape(len(vib), -1)], axis=1)


def bootstrap_ci(values: np.ndarray, n_boot: int = 20_000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def logo_fold_mae(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    splitter = CVSplitter(n_folds=16, logo=True, seed=42)
    mae: list[float] = []
    for _, _, train_idx, _, test_idx in splitter.split(groups):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        mae.append(float(np.abs(y[test_idx] - model.predict(X[test_idx])).mean()))
    return np.asarray(mae)


def main() -> int:
    cache = np.load(INTERMEDIATE_DIR / "cached_specs" / "alternative_reps.npz", allow_pickle=True)
    with np.load(INTERMEDIATE_DIR / "cached_specs" / "mean_specs.npz", allow_pickle=True) as metadata:
        groups = np.asarray(metadata["condition_ids"], dtype=int)
        sample_ids = np.asarray(metadata["sample_ids"], dtype=int)
    roughness = load_surface_roughness()
    y = np.asarray(
        [get_roughness_for_sample(roughness, int(condition), int(sample)) for condition, sample in zip(groups, sample_ids)],
        dtype=np.float32,
    )
    ae = np.asarray(cache["ae_logspec"], dtype=np.float32)
    vib = np.asarray(cache["vib_logspec"], dtype=np.float32)
    cache.close()

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {"Full dB-z": (ae, vib)}
    ae_removed = ae.copy()
    # 1 MHz is bin 150 for fs=4 MHz and n_fft=598; remove frequencies >1 MHz.
    ae_removed[:, :, 151:, :] = 0.0
    variants["Retrained without AE >1 MHz"] = (ae_removed, vib)
    vib_removed = vib.copy()
    # 2--15 kHz maps to bins 20--150 for fs=51.2 kHz and n_fft=512.
    vib_removed[:, :, 20:151, :] = 0.0
    variants["Retrained without Vib 2--15 kHz"] = (ae, vib_removed)
    variants["Retrained without both bands"] = (ae_removed, vib_removed)
    vib_high_removed = vib.copy()
    # Remove all vibration bins above 15 kHz (bin centres 15.1--25.6 kHz).
    vib_high_removed[:, :, 151:, :] = 0.0
    variants["Retrained without Vib >15 kHz"] = (ae, vib_high_removed)
    vib_rf_band_removed = vib.copy()
    # Remove the RF OOF TreeSHAP-dominant 16--24 kHz region (100 Hz/bin).
    vib_rf_band_removed[:, :, 160:241, :] = 0.0
    variants["Retrained without Vib 16--24 kHz"] = (ae, vib_rf_band_removed)

    fold_rows: list[dict[str, float | int | str]] = []
    results: dict[str, np.ndarray] = {}
    for name, (ae_variant, vib_variant) in variants.items():
        print(f"{name} ...", flush=True)
        maes = logo_fold_mae(flatten(ae_variant, vib_variant), y, groups)
        results[name] = maes
        fold_rows.extend({"variant": name, "condition": condition + 1, "mae": value} for condition, value in enumerate(maes))

    baseline = results["Full dB-z"]
    summary: list[dict[str, float | str]] = []
    for name, maes in results.items():
        diff = maes - baseline
        ci_lo, ci_hi = bootstrap_ci(diff)
        summary.append({
            "variant": name,
            "mean_mae": float(maes.mean()),
            "fold_sd": float(maes.std()),
            "mean_difference_vs_full": float(diff.mean()),
            "difference_ci95_low": ci_lo,
            "difference_ci95_high": ci_hi,
            "protocol": "retrained LOGO RF; removed dB-z features set to zero before fit and test",
        })

    out = ROOT / "reports" / "evidence" / "tables"
    pd.DataFrame(summary).to_csv(out / "retrained_frequency_band_ablation.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "retrained_frequency_band_ablation_folds.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
