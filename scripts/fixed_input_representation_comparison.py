"""LOGO comparison that holds the exact input configuration fixed by family.

All four tabular learners use each available identical configuration. This
does not tune models per representation; defaults/hyperparameters are held
fixed within learner to isolate the representation/input choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "src"), str(ROOT / "scripts")])

from train_and_evaluate import (  # noqa: E402
    CVSplitter,
    SimpleGrindingDataset,
    model_factory,
    prepare_sklearn_array,
    scale_data_dict,
    smart_load_data,
)


# The sklearn MLP baseline fails numerically on these flattened matrices when
# trained on the raw target. Exclude it until a separately target-scaled and
# tuned comparison can be reported as a distinct experiment.
LEARNERS = ["RandomForestModel", "RidgeRegressionModel", "LightGBMModel"]
CONFIGS = {
    "dB": "ae_spec+vib_spec",
    "dB-z": "ae_logspec+vib_logspec",
    "log-mel": "ae_mel+vib_mel",
    "WST": "ae_wst+vib_wst",
    "time-domain": "ae_features+vib_features",
    "process parameters": "pp",
}


def matrix(data: dict, indices: np.ndarray, config: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = SimpleGrindingDataset(data, indices, config)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs, targets, _ = next(iter(loader))
    return prepare_sklearn_array(inputs), targets.numpy().ravel()


def data_for_config(full: dict, config: str) -> dict:
    """Keep preprocessing scoped to the exact inputs declared for one cell."""
    selected = {
        "targets": full["targets"],
        "condition_ids": full["condition_ids"],
        "sample_ids": full.get("sample_ids"),
    }
    for key in config.split("+"):
        selected[key] = full[key]
    return selected


def main() -> None:
    full = smart_load_data(LEARNERS, list(CONFIGS.values()))
    groups = full["condition_ids"]
    splitter = CVSplitter(n_folds=16, logo=True, seed=42)
    rows = []
    fold_rows = []
    for family, config in CONFIGS.items():
        config_data = data_for_config(full, config)
        for learner in LEARNERS:
            fold_maes = []
            for _, fold, train_idx, _, test_idx in splitter.split(groups):
                # Fold-wise scaling is identical across learners/configurations.
                scaled, _ = scale_data_dict(config_data, train_idx, scale_specs=True, scale_target=False)
                X_train, y_train = matrix(scaled, train_idx, config)
                X_test, y_test = matrix(scaled, test_idx, config)
                # Parallel tree construction changes wall time, not the seeded
                # estimator or the fixed-input comparison definition.
                kwargs = {"n_jobs": -1} if learner == "RandomForestModel" else {}
                model = model_factory(learner, **kwargs)
                model.fit(X_train, y_train)
                mae = float(np.abs(y_test - model.predict(X_test)).mean())
                fold_maes.append(mae)
                fold_rows.append({"family": family, "config": config, "learner": learner, "fold": fold + 1, "mae": mae})
            rows.append({
                "family": family,
                "config": config,
                "learner": learner,
                "mean_mae": float(np.mean(fold_maes)),
                "std_mae": float(np.std(fold_maes)),
            })
            print(f"{learner:22s} {family:18s} {np.mean(fold_maes):.4f}", flush=True)
    out = ROOT / "reports/evidence/tables/fixed_input_representation_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT / "reports/evidence/tables/fixed_input_representation_comparison_folds.csv", index=False)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
