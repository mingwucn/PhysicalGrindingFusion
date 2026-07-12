#!/usr/bin/env python3
"""
Uncertainty quantification for the best deep-learning model.

Produces:
- MC-dropout prediction intervals on a held-out test set.
- Reliability/calibration plot (predicted std vs. observed absolute error).
- CSV of predictions with mean, std, lower/upper nominal 95% MC-dropout epistemic interval, and true value.

Usage:
    python scripts/uncertainty_best_model.py
    python scripts/uncertainty_best_model.py --model ResNetVibCNN --config vib_spec

If --model/--config are omitted, the script picks the best DL result from
reports/evidence/tables/full_results.csv (or cv_results JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from grinding_physic_fusion.data.dataset import parse_config
from grinding_physic_fusion.models.architectures import MODEL_REGISTRY, model_factory
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from train_and_evaluate import (
    InputPreparer,
    SimpleGrindingDataset,
    is_sklearn_wrapper,
    scale_data_dict,
    smart_load_data,
)
from tune_dl_logo import build_model_kwargs, train_pytorch_model_hparams

DEVICE = torch.device("cpu")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "reports" / "evidence" / "metrics"
TABLES_DIR = PROJECT_ROOT / "reports" / "evidence" / "tables"
OUT_DIR = PROJECT_ROOT / "reports" / "evidence" / "uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

DL_MODELS = {
    "ResNetVibCNN", "ResNetAECNN", "MultiscaleSpectrogramCNN", "ChannelAttentionCNN",
    "ResNetFusion", "BilinearFusionNetwork", "CrossModalTransformer",
    "PhysicsInformedFusionNet", "GatedMultimodalFusionNet",
    "MultiHeadAttentionFusion", "AttentionMLP", "FeatureOnlyModel",
    "PhysicsMLP", "ParamsMLP", "TabNetRegressor", "TabTransformerRegressor",
}


def find_best_dl_result() -> Tuple[str, str, Optional[Path]]:
    """Return (model_name, config, checkpoint_path) for the best DL result."""
    best_mae = float("inf")
    best = ("", "", None)

    # Prefer full_results.csv if already aggregated
    full_results = TABLES_DIR / "full_results.csv"
    if full_results.exists():
        df = pd.read_csv(full_results)
        dl_df = df[df["model"].isin(DL_MODELS)].copy()
        if not dl_df.empty:
            row = dl_df.sort_values("mae_mean").iloc[0]
            model, config = row["model"], row["config"]
            cfg_file = config.replace("+", "_")
            ckpt = CHECKPOINT_DIR / f"{model}_{cfg_file}_best.pt"
            return model, config, ckpt if ckpt.exists() else None

    # Fallback to cv_results JSON files
    for path in METRICS_DIR.glob("cv_results_*.json"):
        with open(path) as f:
            data = json.load(f)
        model = data.get("model_name", "")
        if model not in DL_MODELS:
            continue
        mae = data.get("metrics", {}).get("mean_mae", float("inf"))
        if mae < best_mae:
            best_mae = mae
            config = data.get("config", "")
            cfg_file = config.replace("+", "_")
            ckpt = CHECKPOINT_DIR / f"{model}_{cfg_file}_best.pt"
            best = (model, config, ckpt if ckpt.exists() else None)
    return best


def load_model_from_checkpoint(model_name: str, config: str, ckpt_path: Path) -> nn.Module:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        model_kwargs = state.get("model_kwargs", {}) or {}
        model_state = state["model_state"]
    else:
        model_kwargs = {}
        model_state = state

    if not model_kwargs:
        # Reconstruct kwargs from config so model_factory can build a compatible instance
        model_kwargs = build_model_kwargs(model_name, config, {})
    model = model_factory(model_name, **model_kwargs)
    model.load_state_dict(model_state)
    model.eval()
    return model


def train_best_model(
    model_name: str,
    config: str,
    data_dict: Dict[str, Optional[np.ndarray]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> nn.Module:
    """Retrain a model using the best hyperparameters recorded in its cv_results JSON."""
    cfg_file = config.replace("+", "_")
    cv_path = METRICS_DIR / f"cv_results_{model_name}_{cfg_file}.json"
    if cv_path.exists():
        with open(cv_path) as f:
            cv_data = json.load(f)
        best_hparams = cv_data.get("best_hparams", {})
    else:
        best_hparams = {}

    model_hparams = best_hparams.get("model", {})
    training_hparams = best_hparams.get("training", {})

    model_kwargs = build_model_kwargs(model_name, config, model_hparams)
    model = model_factory(model_name, **model_kwargs)

    scaled, _ = scale_data_dict(data_dict, train_idx, scale_specs=False, scale_target=False)

    class _ConfigDataset(SimpleGrindingDataset):
        _KEY_MAP = {"physics": "physics_vector"}

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            target = torch.tensor(self.data["targets"][real_idx], dtype=torch.float32)
            condition_id = int(self.data["condition_ids"][real_idx])
            inputs: Dict[str, torch.Tensor] = {}
            for part in self.parts:
                key = self._KEY_MAP.get(part, part)
                arr = self.data.get(key)
                if arr is not None:
                    inputs[part] = torch.from_numpy(arr[real_idx].astype(np.float32))
            return inputs, target, condition_id

    train_ds = _ConfigDataset(scaled, train_idx, config)
    val_ds = _ConfigDataset(scaled, val_idx, config)
    bs = int(training_hparams.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=min(bs, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(bs, len(val_ds)), shuffle=False)

    preparer = InputPreparer(model)
    train_pytorch_model_hparams(
        model,
        train_loader,
        val_loader,
        preparer,
        DEVICE,
        lr=float(training_hparams.get("lr", 1e-3)),
        weight_decay=float(training_hparams.get("weight_decay", 1e-4)),
        epochs=int(training_hparams.get("epochs", 150)),
        patience=int(training_hparams.get("patience", 20)),
    )
    return model


def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    preparer: InputPreparer,
    n_passes: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_mean, y_std) from T stochastic forward passes."""
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for _ in range(n_passes):
        preds, targets = [], []
        model.train()  # keep dropout active
        with torch.no_grad():
            for batch in loader:
                inputs_dict, target, _ = batch
                kwargs, missing = preparer.prepare(inputs_dict, DEVICE)
                if missing:
                    continue
                out = model(**kwargs)
                pred = out["roughness"] if isinstance(out, dict) else out
                preds.append(pred.cpu().numpy())
                targets.append(target.numpy())
        all_preds.append(np.concatenate(preds))
        all_targets.append(np.concatenate(targets).ravel())

    y_true = all_targets[0]
    preds_arr = np.stack(all_preds, axis=0)  # (T, N)
    y_mean = preds_arr.mean(axis=0)
    y_std = preds_arr.std(axis=0, ddof=1)
    return y_true, y_mean, y_std


def split_train_val(groups: np.ndarray, train_idx: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Reserve one random condition from train_idx as validation (LOGO style)."""
    rng = np.random.RandomState(seed)
    train_conditions = np.unique(groups[train_idx])
    val_condition = rng.choice(train_conditions)
    val_mask = groups[train_idx] == val_condition
    val_idx = train_idx[val_mask]
    subtrain_idx = train_idx[~val_mask]
    return subtrain_idx, val_idx


def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin samples by predicted std and return bin centers, mean std, and mean absolute error."""
    abs_err = np.abs(y_pred - y_true)
    order = np.argsort(y_std)
    y_std_s, abs_err_s = y_std[order], abs_err[order]
    n = len(y_std_s)
    bin_size = max(1, n // n_bins)
    bin_centers, mean_std, mean_err = [], [], []
    for i in range(0, n, bin_size):
        sl = slice(i, min(i + bin_size, n))
        bin_centers.append((y_std_s[sl][0] + y_std_s[sl][-1]) / 2)
        mean_std.append(y_std_s[sl].mean())
        mean_err.append(abs_err_s[sl].mean())
    return np.array(bin_centers), np.array(mean_std), np.array(mean_err)


def main() -> int:
    parser = argparse.ArgumentParser(description="Uncertainty quantification for best DL model")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--config", type=str, default=None, help="Config string")
    parser.add_argument("--n_passes", type=int, default=50, help="MC dropout forward passes")
    parser.add_argument("--device", default="cpu", help="torch device")
    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)

    if args.model and args.config:
        model_name, config = args.model, args.config
        cfg_file = config.replace("+", "_")
        ckpt_path = CHECKPOINT_DIR / f"{model_name}_{cfg_file}_best.pt"
        checkpoint = ckpt_path if ckpt_path.exists() else None
    else:
        model_name, config, checkpoint = find_best_dl_result()

    if not model_name:
        print("No DL results found. Train some DL models first.")
        return 1

    print(f"Best DL result: {model_name} / {config}")
    print(f"Checkpoint: {checkpoint}")

    print("Loading data ...")
    data_dict = smart_load_data([model_name], [config])
    groups = data_dict["condition_ids"]
    unique = np.unique(groups)
    rng = np.random.RandomState(42)
    n_test = max(1, int(0.2 * len(unique)))
    test_conditions = rng.choice(unique, n_test, replace=False)
    test_mask = np.isin(groups, test_conditions)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    scaled, _ = scale_data_dict(data_dict, train_idx, scale_specs=False, scale_target=False)

    class _ConfigDataset(SimpleGrindingDataset):
        _KEY_MAP = {"physics": "physics_vector"}

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            target = torch.tensor(self.data["targets"][real_idx], dtype=torch.float32)
            condition_id = int(self.data["condition_ids"][real_idx])
            inputs: Dict[str, torch.Tensor] = {}
            for part in self.parts:
                key = self._KEY_MAP.get(part, part)
                arr = self.data.get(key)
                if arr is not None:
                    inputs[part] = torch.from_numpy(arr[real_idx].astype(np.float32))
            return inputs, target, condition_id

    if checkpoint is not None:
        print("Loading model from checkpoint ...")
        model = load_model_from_checkpoint(model_name, config, checkpoint)
    else:
        print("No checkpoint found; retraining best configuration ...")
        subtrain_idx, val_idx = split_train_val(groups, train_idx)
        model = train_best_model(model_name, config, data_dict, subtrain_idx, val_idx)
        # Persist the retrained model for reproducibility
        ckpt_name = f"{model_name}_{config.replace('+', '_')}_uncertainty.pt"
        torch.save({"model_state": model.state_dict(), "config": config}, CHECKPOINT_DIR / ckpt_name)
        print(f"Saved retrained model to {CHECKPOINT_DIR / ckpt_name}")

    if is_sklearn_wrapper(model):
        print("Uncertainty script is intended for PyTorch models.")
        return 0

    preparer = InputPreparer(model)
    test_ds = _ConfigDataset(scaled, test_idx, config)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"Running MC dropout with {args.n_passes} passes ...")
    y_true, y_mean, y_std = mc_dropout_predict(model, test_loader, preparer, n_passes=args.n_passes)

    # Prediction intervals
    z = 1.96
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = float(in_interval.mean())
    mae = float(mean_absolute_error(y_true, y_mean))
    mean_width = float((upper - lower).mean())

    safe_model = model_name.replace(" ", "_")
    safe_cfg = config.replace("+", "_")

    # Save CSV
    df = pd.DataFrame({
        "y_true": y_true,
        "y_mean": y_mean,
        "y_std": y_std,
        "pi_lower_95": lower,
        "pi_upper_95": upper,
        "in_interval": in_interval,
    })
    csv_path = OUT_DIR / f"mc_dropout_{safe_model}_{safe_cfg}.csv"
    df.to_csv(csv_path, index=False)

    # Save summary
    summary = {
        "model": model_name,
        "config": config,
        "n_test": len(y_true),
        "mae": mae,
        "mean_std": float(y_std.mean()),
        "coverage_95": coverage,
        "mean_pi_width": mean_width,
        "n_passes": args.n_passes,
    }
    with open(OUT_DIR / f"mc_dropout_{safe_model}_{safe_cfg}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMC dropout summary ({model_name} / {config}):")
    print(f"  MAE          = {mae:.5f}")
    print(f"  Mean std     = {float(y_std.mean()):.5f}")
    print(f"  95% coverage = {coverage:.3f}")
    print(f"  Mean PI width= {mean_width:.5f}")

    # Calibration plot
    bin_centers, mean_std, mean_err = reliability_diagram(y_true, y_mean, y_std)
    calibration_figure = MutableFigure(f"mc_dropout_calibration_{safe_model}_{safe_cfg}.png", profile=FigureProfiles.SQUARE, out_dir=OUT_DIR, metadata={"generator": "scripts/uncertainty_best_model.py"})
    fig, ax = calibration_figure.create()
    ax.plot(mean_std, mean_err, "o-", label="Observed error")
    ax.plot([0, max(mean_std.max(), mean_err.max())], [0, max(mean_std.max(), mean_err.max())], "k--", label="MAE = predicted standard deviation reference")
    ax.set_xlabel("Predicted std (µm)")
    ax.set_ylabel("Mean absolute error (µm)")
    ax.set_title(f"MC Dropout Calibration – {model_name}\n{config}")
    ax.legend()
    fig.tight_layout()
    calibration_figure.save()

    # Prediction interval plot (sorted by y_mean)
    order = np.argsort(y_mean)
    intervals_figure = MutableFigure(f"mc_dropout_intervals_{safe_model}_{safe_cfg}.png", profile=FigureProfiles.DOUBLE, out_dir=OUT_DIR, metadata={"generator": "scripts/uncertainty_best_model.py"})
    fig, ax = intervals_figure.create()
    ax.fill_between(range(len(order)), lower[order], upper[order], alpha=0.3, color=PublicationPalette.UNCERTAINTY, label="Nominal 95% MC-dropout epistemic band")
    ax.plot(range(len(order)), y_mean[order], color=PublicationPalette.PREDICTED, label="Predicted mean")
    ax.scatter(range(len(order)), y_true[order], c=PublicationPalette.OBSERVED, s=15, label="True", zorder=3)
    ax.set_xlabel("Sample index (sorted by prediction)")
    ax.set_ylabel("Surface roughness (µm)")
    ax.set_title(f"MC Dropout Predictions – {model_name}\n{config}")
    ax.legend()
    fig.tight_layout()
    intervals_figure.save()

    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
