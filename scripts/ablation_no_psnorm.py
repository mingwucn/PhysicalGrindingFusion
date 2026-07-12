#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
Strict per-sample-normalization ablation for the top RF configuration.

Compares:
  1. RF on ae_logspec + vib_logspec (per-sample/channel standardisation)
  2. RF on raw dB spectrograms (ae_spec + vib_spec)

Both use the same 16 LOGO folds and the same random-forest hyperparameters
as the main benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    """Flatten AE and Vib multi-channel spectrograms into one vector."""
    return np.concatenate([ae.reshape(-1), vib.reshape(-1)])


def logo_mae(
    X: np.ndarray,
    y: np.ndarray,
    condition_ids: np.ndarray,
    n_conditions: int = 16,
    random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """Return overall LOGO MAE and per-condition MAEs."""
    per_condition = np.full(n_conditions, np.nan)
    preds_all = np.zeros_like(y)
    # Mirror CVSplitter(logo=True, seed=42): each outer test condition also
    # reserves one deterministic whole-condition validation set. Sklearn RF
    # does not consume validation data, but canonical training excludes it.
    rng = np.random.RandomState(random_state)
    val_conditions = []
    for c in range(1, n_conditions + 1):
        remaining = np.array([candidate for candidate in range(1, n_conditions + 1) if candidate != c])
        val_c = int(rng.choice(remaining))
        val_conditions.append(val_c)
        train_idx = (condition_ids != c) & (condition_ids != val_c)
        test_idx = condition_ids == c
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Match the canonical RF spectrogram pipeline: it does not apply
        # fold-wise feature scaling to spectrogram inputs. The sole changed
        # operation between variants is per-sample dB-map standardisation.
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_all[test_idx] = preds
        per_condition[c - 1] = np.mean(np.abs(preds - y_test))

    overall = float(np.mean(np.abs(preds_all - y)))
    return overall, per_condition, np.asarray(val_conditions, dtype=int)


def main() -> int:
    # Load metadata
    roughness = load_surface_roughness()
    mean_cache = np.load(INTERMEDIATE_DIR / "cached_specs" / "mean_specs.npz", allow_pickle=True)
    condition_ids = np.asarray(mean_cache["condition_ids"])
    sample_ids = np.asarray(mean_cache["sample_ids"])
    y = np.array(
        [get_roughness_for_sample(roughness, int(cid), int(sid)) for cid, sid in zip(condition_ids, sample_ids)],
        dtype=np.float32,
    )

    # Variant 1: with per-sample/channel standardisation (cached logspec)
    alt_cache = np.load(INTERMEDIATE_DIR / "cached_specs" / "alternative_reps.npz", allow_pickle=True)
    X_with = np.array(
        [flatten_sample(ae, vib) for ae, vib in zip(alt_cache["ae_logspec"], alt_cache["vib_logspec"])]
    )
    overall_with, per_with, val_conditions = logo_mae(X_with, y, condition_ids)

    # Variant 2: without per-sample/channel standardisation (raw dB spectrograms)
    X_without = np.array(
        [flatten_sample(ae, vib) for ae, vib in zip(mean_cache["ae_spec"], mean_cache["vib_spec"])]
    )
    overall_without, per_without, val_conditions_without = logo_mae(X_without, y, condition_ids)
    if not np.array_equal(val_conditions, val_conditions_without):
        raise RuntimeError("Validation-condition sequence differs between ablation variants")

    mean_cache.close()
    alt_cache.close()

    out_dir = ROOT / "reports" / "evidence" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "variant": ["with per-sample norm", "without per-sample norm"],
            "config": ["ae_logspec+vib_logspec", "ae_spec+vib_spec (raw dB)"],
            "overall_mae": [overall_with, overall_without],
            "median_per_condition_mae": [float(np.median(per_with)), float(np.median(per_without))],
            "max_per_condition_mae": [float(np.max(per_with)), float(np.max(per_without))],
            "experiment_id": ["psnorm-strict-logo-seed42-v4", "psnorm-strict-logo-seed42-v4"],
        }
    )
    summary.to_csv(out_dir / "ablation_per_sample_normalisation.csv", index=False)
    print(summary.to_string(index=False))

    per_condition_df = pd.DataFrame(
        {
            "condition": np.arange(1, 17),
            "with_per_sample_norm": per_with,
            "without_per_sample_norm": per_without,
            "validation_condition": val_conditions,
        }
    )
    per_condition_df.to_csv(out_dir / "ablation_per_sample_normalisation_per_condition.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
