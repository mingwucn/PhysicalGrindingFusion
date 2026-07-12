"""Assess temporal-position and multi-window sensitivity from archived maps.

The canonical cache averages all 2,910 local spectrogram windows per pass.
This analysis evaluates early, middle, and late thirds separately, then
evaluates a concatenated early/middle/late descriptor. The latter preserves
coarse temporal ordering while using the same local maps and canonical RF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import INTERMEDIATE_DIR, MISSING_SAMPLE, discover_samples, load_surface_roughness  # noqa: E402
from train_and_evaluate import CVSplitter  # noqa: E402


POSITIONS = ("early", "middle", "late")
MULTI_WINDOW = "early_middle_late_concat"


def per_sample_zscore(spec: np.ndarray) -> np.ndarray:
    """Match the dB-z cache normalisation independently for each channel."""
    mean = spec.mean(axis=(1, 2), keepdims=True)
    std = spec.std(axis=(1, 2), keepdims=True) + 1e-8
    return (spec - mean) / std


def thirds(length: int) -> dict[str, slice]:
    """Return nonempty early/middle/late thirds for one cached pass."""
    first = max(1, length // 3)
    second = max(first + 1, (2 * length) // 3)
    return {
        "early": slice(0, first),
        "middle": slice(first, second),
        "late": slice(second, length),
    }


def load_position_features() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    targets = load_surface_roughness()
    features = {name: [] for name in (*POSITIONS, MULTI_WINDOW)}
    y, groups = [], []
    for cid, sid in discover_samples(config=None):
        if (cid, sid) == MISSING_SAMPLE:
            continue
        with np.load(INTERMEDIATE_DIR / f"{cid}-{sid:02d}-0_spec.npz", allow_pickle=True) as data:
            ae = data["spec_ae"]
            vib = data["spec_vib"]
            third_features = []
            for name, window in thirds(len(ae)).items():
                # The local-window axis is the first axis in both modalities.
                ae_mean = per_sample_zscore(ae[window].mean(axis=0))
                vib_mean = per_sample_zscore(vib[window].mean(axis=0))
                descriptor = np.concatenate([ae_mean.ravel(), vib_mean.ravel()])
                features[name].append(descriptor)
                third_features.append(descriptor)
            # The ordered concatenation is a multi-window representation, not
            # a prediction average. It exposes the same RF to coarse pass
            # position without changing the underlying local maps.
            features[MULTI_WINDOW].append(np.concatenate(third_features))
        y.append(targets[(cid - 1) * 20 + (sid - 1)])
        groups.append(cid)
    return {key: np.asarray(value, dtype=np.float32) for key, value in features.items()}, np.asarray(y, dtype=np.float32), np.asarray(groups)


def logo_maes(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[float]:
    maes = []
    splitter = CVSplitter(n_folds=16, logo=True, seed=42)
    for _, _, train_idx, _, test_idx in splitter.split(groups):
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=1,
            random_state=42, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        maes.append(float(np.abs(y[test_idx] - model.predict(X[test_idx])).mean()))
    return maes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=(*POSITIONS, MULTI_WINDOW),
        default=(*POSITIONS, MULTI_WINDOW),
        help="Descriptor variants to evaluate; defaults to all archived and multi-window variants.",
    )
    args = parser.parse_args()
    X_by_position, y, groups = load_position_features()
    summary, folds = [], []
    for position in args.positions:
        X = X_by_position[position]
        maes = logo_maes(X, y, groups)
        summary.append({"position": position, "mean_mae": float(np.mean(maes)), "std_mae": float(np.std(maes))})
        folds.extend({"position": position, "condition": i + 1, "mae": value} for i, value in enumerate(maes))
        print(f"{position}: {np.mean(maes):.6f}", flush=True)
    out = ROOT / "reports/evidence/tables"
    summary_path = out / "pass_position_sensitivity.csv"
    folds_path = out / "pass_position_sensitivity_folds.csv"
    summary_df = pd.DataFrame(summary)
    folds_df = pd.DataFrame(folds)
    # Partial runs update only the requested variants, preserving previously
    # generated early/middle/late baselines in the same auditable artifact.
    if set(args.positions) != set((*POSITIONS, MULTI_WINDOW)):
        if summary_path.exists():
            previous = pd.read_csv(summary_path)
            summary_df = pd.concat(
                [previous.loc[~previous["position"].isin(args.positions)], summary_df],
                ignore_index=True,
            )
        if folds_path.exists():
            previous_folds = pd.read_csv(folds_path)
            folds_df = pd.concat(
                [previous_folds.loc[~previous_folds["position"].isin(args.positions)], folds_df],
                ignore_index=True,
            )
    summary_df.sort_values("position").to_csv(summary_path, index=False)
    folds_df.sort_values(["position", "condition"]).to_csv(folds_path, index=False)


if __name__ == "__main__":
    main()
