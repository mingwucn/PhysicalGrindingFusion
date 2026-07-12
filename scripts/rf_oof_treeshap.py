#!/usr/bin/env python3
"""Out-of-fold TreeSHAP for the reproducible current RF/dB-z pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import INTERMEDIATE_DIR

OUT = ROOT / "reports" / "evidence" / "xai"
OUT.mkdir(parents=True, exist_ok=True)

AE_SHAPE = (2, 300, 47)
VIB_SHAPE = (3, 257, 13)
AE_SR, AE_NFFT = 4_000_000.0, 598
VIB_SR, VIB_NFFT = 51_200.0, 512
SEED = 42


def flatten(ae: np.ndarray, vib: np.ndarray) -> np.ndarray:
    return np.concatenate((ae.reshape(len(ae), -1), vib.reshape(len(vib), -1)), axis=1)


def frequency_profile(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Pool mean absolute SHAP over samples, channels, and STFT frames."""
    return np.abs(values.reshape(len(values), *shape)).mean(axis=(0, 1, 3))


def main() -> int:
    cache_dir = INTERMEDIATE_DIR / "cached_specs"
    mean = np.load(cache_dir / "mean_specs.npz", allow_pickle=True)
    alt = np.load(cache_dir / "alternative_reps.npz", allow_pickle=True)
    ae = np.asarray(alt["ae_logspec"], dtype=np.float32)
    vib = np.asarray(alt["vib_logspec"], dtype=np.float32)
    y = np.asarray(mean["targets"], dtype=np.float32)
    conditions = np.asarray(mean["condition_ids"], dtype=int)
    X = flatten(ae, vib)
    ae_n = int(np.prod(AE_SHAPE))

    rng = np.random.RandomState(SEED)
    fold_profiles: list[dict[str, float | int | str]] = []
    pooled_ae = np.zeros(AE_SHAPE[1], dtype=np.float64)
    pooled_vib = np.zeros(VIB_SHAPE[1], dtype=np.float64)
    balanced_ae: list[np.ndarray] = []
    balanced_vib: list[np.ndarray] = []
    weighted_n = 0
    prediction_rows = []

    for test_condition in range(1, 17):
        remaining = np.array([c for c in range(1, 17) if c != test_condition])
        validation_condition = int(rng.choice(remaining))
        train = (conditions != test_condition) & (conditions != validation_condition)
        test = conditions == test_condition

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        values = np.asarray(explainer.shap_values(X[test], approximate=True))

        ae_profile = frequency_profile(values[:, :ae_n], AE_SHAPE)
        vib_profile = frequency_profile(values[:, ae_n:], VIB_SHAPE)
        n_test = int(test.sum())
        pooled_ae += ae_profile * n_test
        pooled_vib += vib_profile * n_test
        balanced_ae.append(ae_profile / max(ae_profile.sum(), np.finfo(float).eps))
        balanced_vib.append(vib_profile / max(vib_profile.sum(), np.finfo(float).eps))
        weighted_n += n_test

        ae_bin = int(np.argmax(ae_profile))
        vib_bin = int(np.argmax(vib_profile))
        fold_profiles.append({
            "test_condition": test_condition,
            "validation_condition": validation_condition,
            "n_test": n_test,
            "fold_mae": float(np.mean(np.abs(pred - y[test]))),
            "dominant_ae_bin": ae_bin,
            "dominant_ae_khz": ae_bin * AE_SR / AE_NFFT / 1e3,
            "dominant_vib_bin": vib_bin,
            "dominant_vib_khz": vib_bin * VIB_SR / VIB_NFFT / 1e3,
        })
        for idx, yt, yp in zip(np.flatnonzero(test), y[test], pred):
            prediction_rows.append({
                "sample_index": int(idx),
                "test_condition": test_condition,
                "validation_condition": validation_condition,
                "measured_ra": float(yt),
                "predicted_ra": float(yp),
            })
        print(f"Condition {test_condition:02d}: MAE={fold_profiles[-1]['fold_mae']:.5f}", flush=True)

    pooled_ae /= weighted_n
    pooled_vib /= weighted_n
    balanced_ae_array = np.asarray(balanced_ae)
    balanced_vib_array = np.asarray(balanced_vib)
    pd.DataFrame(fold_profiles).to_csv(OUT / "rf_dbz_oof_treeshap_fold_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "rf_dbz_oof_predictions.csv", index=False)
    pd.DataFrame({
        "frequency_khz": np.arange(len(pooled_ae)) * AE_SR / AE_NFFT / 1e3,
        "mean_abs_shap": pooled_ae,
    }).to_csv(OUT / "rf_dbz_oof_treeshap_ae_profile.csv", index=False)
    pd.DataFrame({
        "frequency_khz": np.arange(len(pooled_vib)) * VIB_SR / VIB_NFFT / 1e3,
        "mean_abs_shap": pooled_vib,
    }).to_csv(OUT / "rf_dbz_oof_treeshap_vib_profile.csv", index=False)
    for suffix, ae_profile, vib_profile in (
        ("condition_balanced", balanced_ae_array.mean(axis=0), balanced_vib_array.mean(axis=0)),
        ("condition_balanced_without_condition7", np.delete(balanced_ae_array, 6, axis=0).mean(axis=0), np.delete(balanced_vib_array, 6, axis=0).mean(axis=0)),
    ):
        pd.DataFrame({"frequency_khz": np.arange(len(ae_profile)) * AE_SR / AE_NFFT / 1e3, "importance_fraction": ae_profile}).to_csv(OUT / f"rf_dbz_oof_treeshap_ae_{suffix}.csv", index=False)
        pd.DataFrame({"frequency_khz": np.arange(len(vib_profile)) * VIB_SR / VIB_NFFT / 1e3, "importance_fraction": vib_profile}).to_csv(OUT / f"rf_dbz_oof_treeshap_vib_{suffix}.csv", index=False)
    print(f"Overall OOF MAE: {pd.DataFrame(prediction_rows).eval('abs(measured_ra - predicted_ra)').mean():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
