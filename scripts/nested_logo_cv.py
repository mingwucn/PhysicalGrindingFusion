#!/usr/bin/env python3
"""
Nested leave-one-condition-out cross-validation for tree-based models.

Outer loop: LOGO (16 folds, each held-out condition is the test set).
Inner loop: grouped 5-fold CV on the remaining conditions for hyperparameter
selection.

Outputs:
    reports/evidence/tables/nested_logo_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.data.dataset import load_all_data
from grinding_physic_fusion.models.architectures import MODEL_REGISTRY
from train_and_evaluate import (
    CVSplitter,
    InputPreparer,
    SimpleGrindingDataset,
    is_sklearn_wrapper,
    model_factory,
    prepare_sklearn_array,
    scale_data_dict,
)
from torch.utils.data import DataLoader

METRICS_DIR = ROOT / "reports" / "evidence" / "metrics"
TABLES_DIR = ROOT / "reports" / "evidence" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def build_model(model_name: str, hparams: Dict[str, Any]) -> Any:
    """Instantiate a sklearn-compatible model with given hparams."""
    cls = MODEL_REGISTRY[model_name]
    return cls(**hparams)


def fit_sklearn_model(model: Any, data_dict: Dict[str, Any], train_idx: np.ndarray, config: str) -> Any:
    """Fit a sklearn model on train indices."""
    from torch.utils.data import DataLoader
    train_ds = SimpleGrindingDataset(data_dict, train_idx, config)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    inputs_dict, target, _ = next(iter(train_loader))
    X = prepare_sklearn_array(inputs_dict)
    y = target.numpy().ravel()
    model.fit(X, y)
    return model


def eval_sklearn_model(model: Any, data_dict: Dict[str, Any], test_idx: np.ndarray, config: str) -> float:
    """Return MAE on test indices."""
    test_ds = SimpleGrindingDataset(data_dict, test_idx, config)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    inputs_dict, target, _ = next(iter(test_loader))
    X = prepare_sklearn_array(inputs_dict)
    y = target.numpy().ravel()
    y_pred = model.predict(X)
    return float(np.mean(np.abs(y - y_pred)))


def inner_cv_score(
    model_name: str,
    config: str,
    hparams: Dict[str, Any],
    data_dict: Dict[str, Any],
    train_val_idx: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Return mean inner CV MAE for a hyperparameter setting."""
    groups = data_dict["condition_ids"][train_val_idx]
    inner_gkf = GroupKFold(n_splits=n_splits)
    maes = []
    for inner_train, inner_val in inner_gkf.split(train_val_idx, groups=groups):
        inner_train_idx = train_val_idx[inner_train]
        inner_val_idx = train_val_idx[inner_val]
        scaled, _ = scale_data_dict(data_dict, inner_train_idx, scale_specs=False, scale_target=False)
        model = build_model(model_name, hparams)
        model = fit_sklearn_model(model, scaled, inner_train_idx, config)
        mae = eval_sklearn_model(model, scaled, inner_val_idx, config)
        maes.append(mae)
    return float(np.mean(maes))


def nested_logo_cv(
    model_name: str,
    config: str,
    hparam_grid: List[Dict[str, Any]],
    full_data: Dict[str, Any],
    seed: int = 42,
    outer_folds: Optional[set[int]] = None,
) -> Dict[str, Any]:
    """Run nested LOGO CV and return per-fold results."""
    splitter = CVSplitter(n_folds=16, logo=True, seed=seed)
    groups = full_data["condition_ids"]

    outer_maes = []
    fold_records = []

    for repeat_idx, fold_idx, train_idx, val_idx, test_idx in splitter.split(groups):
        if outer_folds is not None and (fold_idx + 1) not in outer_folds:
            continue
        print(f"  Outer fold {fold_idx + 1}/16 ...", end=" ")
        sys.stdout.flush()

        # Inner CV hyperparameter selection
        best_mae = float("inf")
        best_hparams = hparam_grid[0]
        for hparams in hparam_grid:
            mae = inner_cv_score(model_name, config, hparams, full_data, train_idx, seed=seed)
            if mae < best_mae:
                best_mae = mae
                best_hparams = hparams

        # Train on full train_val with best hparams, evaluate on outer test
        scaled, _ = scale_data_dict(full_data, train_idx, scale_specs=False, scale_target=False)
        model = build_model(model_name, best_hparams)
        model = fit_sklearn_model(model, scaled, train_idx, config)
        test_mae = eval_sklearn_model(model, scaled, test_idx, config)

        outer_maes.append(test_mae)
        fold_records.append({
            "fold": fold_idx,
            "test_condition": int(np.unique(groups[test_idx])[0]),
            "best_hparams": json.dumps(best_hparams),
            "inner_val_mae": best_mae,
            "outer_test_mae": test_mae,
        })
        print(f"outer MAE={test_mae:.4f}, best_hparams={best_hparams}")

    return {
        "mean_mae": float(np.mean(outer_maes)),
        "std_mae": float(np.std(outer_maes)),
        "fold_records": fold_records,
    }


def main() -> int:
    # Top configs to evaluate with nested CV
    experiments = [
        ("RandomForestModel", "ae_spec+vib_spec"),
        ("RandomForestModel", "ae_logspec+vib_logspec"),
        ("RandomForestModel", "ae_mel+vib_mel"),
        ("LightGBMModel", "ae_spec+vib_spec"),
        ("LightGBMModel", "vib_spec"),
        ("RandomForestModel", "ae_features+vib_features+physics+pp"),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="MODEL:CONFIG",
        help="Run only exact MODEL:CONFIG experiment identifiers.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Estimator worker count; set a bounded value for concurrent tmux runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "nested_logo_results.csv",
        help="CSV destination; use separate paths for concurrent experiment workers.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        nargs="+",
        metavar="FOLD",
        help="Optional 1-based outer LOGO fold subset for a tmux worker.",
    )
    args = parser.parse_args()
    if args.only:
        requested = {tuple(item.split(":", 1)) for item in args.only}
        experiments = [item for item in experiments if item in requested]
        if not experiments:
            parser.error("--only did not match any configured experiment")
    if args.outer_folds and any(fold < 1 or fold > 16 for fold in args.outer_folds):
        parser.error("--outer-folds accepts only 1-based folds in [1, 16]")

    hparam_grids = {
        "RandomForestModel": [
            {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": args.n_jobs},
            {"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": args.n_jobs},
            {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": args.n_jobs},
            # Canonical RF setting; it must be selectable in the nested run.
            {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": args.n_jobs},
        ],
        "LightGBMModel": [
            {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "n_jobs": args.n_jobs, "verbose": -1},
            {"n_estimators": 200, "learning_rate": 0.05, "random_state": 42, "n_jobs": args.n_jobs, "verbose": -1},
            {"n_estimators": 500, "learning_rate": 0.05, "random_state": 42, "n_jobs": args.n_jobs, "verbose": -1},
        ],
    }

    all_results = []
    all_fold_records = []
    for model_name, config in experiments:
        print(f"\nNested LOGO CV: {model_name} / {config}")
        full_data = load_all_data(config=config)
        result = nested_logo_cv(
            model_name,
            config,
            hparam_grids[model_name],
            full_data,
            outer_folds=set(args.outer_folds) if args.outer_folds else None,
        )
        all_results.append({
            "model": model_name,
            "config": config,
            "mean_mae": result["mean_mae"],
            "std_mae": result["std_mae"],
            "n_folds": len(result["fold_records"]),
        })
        all_fold_records.extend(
            {"model": model_name, "config": config, **record}
            for record in result["fold_records"]
        )
        print(f"  -> Mean outer MAE: {result['mean_mae']:.6f} ± {result['std_mae']:.6f}")

    df = pd.DataFrame(all_results)
    out_path = args.output
    if args.only and out_path.exists():
        keys = ["model", "config"]
        previous = pd.read_csv(out_path)
        previous = previous.merge(df[keys], on=keys, how="left", indicator=True)
        previous = previous.loc[previous["_merge"] == "left_only"].drop(columns="_merge")
        df = pd.concat([previous, df], ignore_index=True)
    df.sort_values(["model", "config"], inplace=True)
    df.to_csv(out_path, index=False)
    fold_path = out_path.with_name(f"{out_path.stem}_folds.csv")
    pd.DataFrame(all_fold_records).sort_values(["model", "config", "fold"]).to_csv(
        fold_path, index=False
    )
    print(f"\nSaved {out_path}")
    print(f"Saved {fold_path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
