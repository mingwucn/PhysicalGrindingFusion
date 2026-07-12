#!/usr/bin/env python3
"""
Comprehensive Training and Evaluation Pipeline for Grinding Surface Roughness Prediction.

Supports:
- Grouped cross-validation (default, split by condition_id) and LOGO (16 folds).
- Ordinary KFold (opt-out only; produces optimistic estimates for this dataset).
- Multiple input configurations (spectrograms, features, process parameters, fusion).
- Traditional ML (RandomForest, XGBoost, LightGBM) and Deep Learning models.
- Ablation studies (H1, H2, H3, H5).
- Automatic checkpointing, metrics logging, and CSV report generation.

Usage:
    python scripts/train_and_evaluate.py --models all --configs all --cv_folds 5 --epochs 200
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grinding_physic_fusion.data.dataset import (
    INTERMEDIATE_DIR,
    build_datasets,
    get_available_configs,
    load_all_data,
    parse_config,
)
from grinding_physic_fusion.models.architectures import (
    MODEL_REGISTRY,
    count_parameters,
    model_factory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "evidence" / "metrics"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "evidence" / "tables"
PREDICTIONS_DIR = PROJECT_ROOT / "reports" / "evidence" / "predictions"

RANDOM_SEED = 42
DEVICE = torch.device("cpu")

# Models requested in requirements
DEFAULT_MODELS = [
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "ResNetAECNN",
    "ResNetVibCNN",
    "TabNetRegressor",
    "LSTMPhysicsModel",
    "TCNPhysicsModel",
    "CrossModalTransformer",
    "PhysicsInformedFusionNet",
    "MultiHeadAttentionFusion",
    "BilinearFusionNetwork",
    "GraphNeuralNetworkFusion",
    "GatedMultimodalFusionNet",
    "TabTransformerRegressor",
    "FeatureOnlyModel",
]

# Configs from dataset module (plus a special full-physics config for fusion models)
VALID_DATASET_CONFIGS = get_available_configs()
FULL_PHYSICS_CONFIG = "full_physics"  # pseudo-config: loads everything via config=None

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (CHECKPOINT_DIR, REPORTS_METRICS_DIR, REPORTS_TABLES_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset & Scaling helpers
# ---------------------------------------------------------------------------


class SimpleGrindingDataset(Dataset):
    """Lightweight dataset that returns (inputs_dict, target, condition_id)."""

    def __init__(
        self,
        data_dict: Dict[str, Optional[np.ndarray]],
        indices: np.ndarray,
        config: Optional[str],
    ):
        self.data = data_dict
        self.indices = np.asarray(indices)
        self.config = config
        self.parts = parse_config(config) if config is not None else set()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        real_idx = self.indices[idx]
        target = torch.tensor(self.data["targets"][real_idx], dtype=torch.float32)
        condition_id = int(self.data["condition_ids"][real_idx])
        inputs: Dict[str, torch.Tensor] = {}
        for key in self.parts:
            arr = self.data.get(key)
            if arr is not None:
                inputs[key] = torch.from_numpy(arr[real_idx].astype(np.float32))
        return inputs, target, condition_id


class FullPhysicsDataset(Dataset):
    """
    Dataset for the full-physics pseudo-config.
    Returns keys: ae_spec, vib_spec, physics_vector, pp, ae_features, vib_features.
    """

    ALL_KEYS = [
        "ae_spec",
        "vib_spec",
        "ae_features",
        "vib_features",
        "physics_vector",
        "pp",
    ]

    def __init__(self, data_dict: Dict[str, Optional[np.ndarray]], indices: np.ndarray):
        self.data = data_dict
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        real_idx = self.indices[idx]
        target = torch.tensor(self.data["targets"][real_idx], dtype=torch.float32)
        condition_id = int(self.data["condition_ids"][real_idx])
        inputs: Dict[str, torch.Tensor] = {}
        for key in self.ALL_KEYS:
            arr = self.data.get(key)
            if arr is not None:
                inputs[key] = torch.from_numpy(arr[real_idx].astype(np.float32))
        return inputs, target, condition_id


def scale_data_dict(
    data_dict: Dict[str, Optional[np.ndarray]],
    train_indices: np.ndarray,
    scale_specs: bool = False,
    scale_target: bool = False,
) -> Tuple[Dict[str, Optional[np.ndarray]], Optional[StandardScaler]]:
    """
    Fit StandardScalers on train_indices and transform the entire dictionary.
    Spectrograms are optionally flattened and scaled.
    Targets are optionally standardized.
    Returns (scaled_dict, target_scaler).
    """
    scaled: Dict[str, Optional[np.ndarray]] = {
        "condition_ids": data_dict["condition_ids"].copy(),
    }
    sample_ids = data_dict.get("sample_ids")
    if sample_ids is not None:
        scaled["sample_ids"] = sample_ids.copy()

    target_scaler: Optional[StandardScaler] = None
    if scale_target:
        target_scaler = StandardScaler()
        target_scaler.fit(data_dict["targets"][train_indices].reshape(-1, 1))
        scaled["targets"] = target_scaler.transform(
            data_dict["targets"].reshape(-1, 1)
        ).reshape(-1)
    else:
        scaled["targets"] = data_dict["targets"].copy()

    spectrogram_keys = ["ae_spec", "vib_spec", "ae_mel", "vib_mel", "ae_logspec", "vib_logspec", "ae_wst", "vib_wst"]
    for key in spectrogram_keys + ["ae_features", "vib_features", "physics_vector", "pp"]:
        arr = data_dict.get(key)
        if arr is None:
            scaled[key] = None
            continue

        if key in spectrogram_keys and not scale_specs:
            scaled[key] = arr.copy()
            continue

        scaler = StandardScaler()
        train_arr = arr[train_indices]
        if train_arr.ndim == 1:
            scaler.fit(train_arr.reshape(-1, 1))
            scaled[key] = scaler.transform(arr.reshape(-1, 1)).reshape(-1)
        else:
            orig_shape = arr.shape
            scaler.fit(train_arr.reshape(len(train_indices), -1))
            scaled[key] = scaler.transform(arr.reshape(len(arr), -1)).reshape(orig_shape)
    return scaled, target_scaler


# ---------------------------------------------------------------------------
# Cross-validation splitter
# ---------------------------------------------------------------------------


class CVSplitter:
    """
    Cross-validation splitter supporting:
    - Standard KFold / RepeatedKFold
    - GroupKFold / Repeated GroupKFold
    - Leave-One-Group-Out (LOGO)
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_repeats: int = 1,
        grouped: bool = False,
        logo: bool = False,
        seed: int = RANDOM_SEED,
    ):
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.grouped = grouped
        self.logo = logo
        self.seed = seed

    @property
    def total_folds(self) -> int:
        if self.logo:
            return self.n_folds
        return self.n_folds * self.n_repeats

    def split(
        self, groups: np.ndarray
    ):
        """
        Yield (repeat_idx, fold_idx, train_idx, val_idx, test_idx) for each fold.
        """
        np.random.seed(self.seed)
        n_samples = len(groups)

        if self.logo:
            # Leave-one-group-out (original behaviour)
            unique_groups = np.unique(groups)
            for test_g in unique_groups:
                test_mask = groups == test_g
                remaining = unique_groups[unique_groups != test_g]
                val_g = np.random.choice(remaining)
                val_mask = groups == val_g
                train_mask = ~(test_mask | val_mask)
                yield (
                    0,
                    int(test_g) - 1,
                    np.where(train_mask)[0],
                    np.where(val_mask)[0],
                    np.where(test_mask)[0],
                )
            return

        if self.grouped:
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            if n_groups < self.n_folds:
                raise ValueError(
                    f"Number of groups ({n_groups}) must be >= n_folds ({self.n_folds})"
                )

            for repeat in range(self.n_repeats):
                rng = np.random.RandomState(self.seed + repeat)
                shuffled_groups = unique_groups.copy()
                rng.shuffle(shuffled_groups)

                # Assign groups to folds round-robin so every condition lands in
                # exactly one test fold per repeat.
                group_to_fold = {
                    g: i % self.n_folds for i, g in enumerate(shuffled_groups)
                }

                for fold in range(self.n_folds):
                    test_groups = shuffled_groups[
                        np.arange(n_groups) % self.n_folds == fold
                    ]
                    test_mask = np.isin(groups, test_groups)

                    # Validation must also respect groups: hold out one whole
                    # condition from the remaining training groups.
                    remaining_groups = shuffled_groups[
                        ~np.isin(shuffled_groups, test_groups)
                    ]
                    rng_val = np.random.RandomState(
                        self.seed + repeat * 1000 + fold
                    )
                    val_group = rng_val.choice(remaining_groups)
                    val_mask = groups == val_group

                    train_mask = ~(test_mask | val_mask)
                    yield (
                        repeat,
                        fold,
                        np.where(train_mask)[0],
                        np.where(val_mask)[0],
                        np.where(test_mask)[0],
                    )
        else:
            if self.n_repeats == 1:
                kf = KFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.seed
                )
                splits = list(kf.split(np.arange(n_samples)))
                for fold, (train_idx, test_idx) in enumerate(splits):
                    rng = np.random.RandomState(self.seed + fold)
                    perm = rng.permutation(len(train_idx))
                    val_size = max(1, int(0.2 * len(train_idx)))
                    val_idx = train_idx[perm[:val_size]]
                    train_idx = train_idx[perm[val_size:]]
                    yield (0, fold, train_idx, val_idx, test_idx)
            else:
                rkf = RepeatedKFold(
                    n_splits=self.n_folds,
                    n_repeats=self.n_repeats,
                    random_state=self.seed,
                )
                for i, (train_idx, test_idx) in enumerate(
                    rkf.split(np.arange(n_samples))
                ):
                    repeat = i // self.n_folds
                    fold = i % self.n_folds
                    rng = np.random.RandomState(self.seed + repeat * 1000 + fold)
                    perm = rng.permutation(len(train_idx))
                    val_size = max(1, int(0.2 * len(train_idx)))
                    val_idx = train_idx[perm[:val_size]]
                    train_idx = train_idx[perm[val_size:]]
                    yield (repeat, fold, train_idx, val_idx, test_idx)


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------


class InputPreparer:
    """
    Inspect a model's forward signature and map dataset inputs to model arguments.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_args: List[str] = []
        self.required_args: List[str] = []
        self.optional_args: List[str] = []
        self._inspect()

    def _inspect(self) -> None:
        sig = inspect.signature(self.model.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            self.forward_args.append(name)
            if param.default is inspect.Parameter.empty:
                self.required_args.append(name)
            else:
                self.optional_args.append(name)

    def prepare(
        self, inputs_dict: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Build kwargs dict for model.forward().
        Returns (kwargs, missing_required_args).
        """
        kwargs: Dict[str, torch.Tensor] = {}
        missing: List[str] = []

        # Determine batch size from any available tensor
        batch_size: Optional[int] = None
        for t in inputs_dict.values():
            batch_size = t.size(0)
            break

        for arg in self.forward_args:
            if arg == "x":
                # Single input: if only one tensor is available and it's multi-dimensional,
                # pass it directly (for CNNs). Otherwise flatten and concatenate.
                parts: List[torch.Tensor] = []
                for key in sorted(inputs_dict.keys()):
                    t = inputs_dict[key]
                    if t.dim() > 2:
                        t = t.view(t.size(0), -1)
                    elif t.dim() == 1:
                        t = t.unsqueeze(1)
                    parts.append(t)
                if len(inputs_dict) == 1:
                    # Only one modality: pass it directly without flattening if >2D
                    single_key = list(inputs_dict.keys())[0]
                    single_t = inputs_dict[single_key]
                    if single_t.dim() > 2:
                        kwargs["x"] = single_t.to(device)
                    else:
                        if single_t.dim() == 1:
                            single_t = single_t.unsqueeze(1)
                        kwargs["x"] = single_t.to(device)
                elif parts:
                    kwargs["x"] = torch.cat(parts, dim=1).to(device)
                else:
                    missing.append("x")

            elif arg == "ae_spec":
                if "ae_spec" in inputs_dict:
                    kwargs["ae_spec"] = inputs_dict["ae_spec"].to(device)
                elif arg in self.required_args:
                    missing.append("ae_spec")

            elif arg == "vib_spec":
                if "vib_spec" in inputs_dict:
                    kwargs["vib_spec"] = inputs_dict["vib_spec"].to(device)
                elif arg in self.required_args:
                    missing.append("vib_spec")

            elif arg == "physics":
                if "physics_vector" in inputs_dict:
                    kwargs["physics"] = inputs_dict["physics_vector"].to(device)
                elif "ae_features" in inputs_dict or "vib_features" in inputs_dict:
                    # Build physics vector from ae_features + vib_features
                    parts = []
                    if "ae_features" in inputs_dict:
                        parts.append(inputs_dict["ae_features"])
                    if "vib_features" in inputs_dict:
                        parts.append(inputs_dict["vib_features"])
                    physics = torch.cat(parts, dim=1)
                    # Pad to 44-D if needed (models expect 44-D physics vector)
                    if physics.size(-1) < 44:
                        pad = torch.zeros(physics.size(0), 44 - physics.size(-1), device=physics.device, dtype=physics.dtype)
                        physics = torch.cat([physics, pad], dim=1)
                    kwargs["physics"] = physics.to(device)
                else:
                    if arg in self.required_args:
                        missing.append("physics")

            elif arg == "params":
                if "pp" in inputs_dict:
                    kwargs["params"] = inputs_dict["pp"].to(device)
                else:
                    if arg in self.required_args:
                        missing.append("params")

            elif arg in ("bdi_st", "lengths"):
                # Optional / not available in current dataset
                pass

            elif arg == "ae_features":
                if "ae_features" in inputs_dict:
                    kwargs["ae_features"] = inputs_dict["ae_features"].to(device)
                elif arg in self.required_args:
                    missing.append("ae_features")

            elif arg == "vib_features":
                if "vib_features" in inputs_dict:
                    kwargs["vib_features"] = inputs_dict["vib_features"].to(device)
                elif arg in self.required_args:
                    missing.append("vib_features")

            else:
                # Unknown argument – try to map by exact name
                if arg in inputs_dict:
                    kwargs[arg] = inputs_dict[arg].to(device)
                elif arg in self.required_args:
                    missing.append(arg)

        return kwargs, missing


# ---------------------------------------------------------------------------
# Sklearn helpers
# ---------------------------------------------------------------------------


def is_sklearn_wrapper(model: nn.Module) -> bool:
    """Return True if model is one of the sklearn wrappers in the registry."""
    return hasattr(model, "fit") and hasattr(model, "predict")


def prepare_sklearn_array(inputs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """Concatenate all inputs into a 2-D numpy array for sklearn models."""
    parts: List[np.ndarray] = []
    for key in sorted(inputs_dict.keys()):
        t = inputs_dict[key]
        if t.dim() > 2:
            t = t.view(t.size(0), -1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        parts.append(t.cpu().numpy())
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def train_sklearn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> float:
    """Fit sklearn wrapper and return validation MAE."""
    X_train, y_train = [], []
    for batch in train_loader:
        inputs_dict, target, _ = batch
        X_train.append(prepare_sklearn_array(inputs_dict))
        y_train.append(target.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0).ravel()

    model.fit(X_train, y_train)

    # Validation
    X_val, y_val = [], []
    for batch in val_loader:
        inputs_dict, target, _ = batch
        X_val.append(prepare_sklearn_array(inputs_dict))
        y_val.append(target.numpy())
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0).ravel()

    y_pred = model.predict(X_val)
    return float(mean_absolute_error(y_val, y_pred))


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    preparer: InputPreparer,
    device: torch.device,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    batch_size_hint: int = 32,
) -> Tuple[float, Dict[str, Any]]:
    """
    Train a PyTorch model with AdamW, MSE loss, and early stopping on val MAE.
    Returns (best_val_mae, history).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs_dict, target, _ = batch
            kwargs, missing = preparer.prepare(inputs_dict, device)
            if missing:
                raise RuntimeError(f"Missing required forward args: {missing}")

            optimizer.zero_grad()
            out = model(**kwargs)
            pred = out["roughness"] if isinstance(out, dict) else out
            loss = criterion(pred, target.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs_dict, target, _ = batch
                kwargs, missing = preparer.prepare(inputs_dict, device)
                out = model(**kwargs)
                pred = out["roughness"] if isinstance(out, dict) else out
                val_preds.append(pred.cpu().numpy())
                val_targets.append(target.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets).ravel()
        val_mae = float(mean_absolute_error(val_targets, val_preds))

        history["train_loss"].append(avg_train_loss)
        history["val_mae"].append(val_mae)

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_mae, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    preparer: InputPreparer,
    device: torch.device,
    target_scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate model on test set and return metrics + predictions."""
    if is_sklearn_wrapper(model):
        X_test, y_test = [], []
        for batch in test_loader:
            inputs_dict, target, _ = batch
            X_test.append(prepare_sklearn_array(inputs_dict))
            y_test.append(target.numpy())
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0).ravel()
        y_pred = model.predict(X_test)
    else:
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs_dict, target, _ = batch
                kwargs, missing = preparer.prepare(inputs_dict, device)
                out = model(**kwargs)
                pred = out["roughness"] if isinstance(out, dict) else out
                preds.append(pred.cpu().numpy())
                targets.append(target.numpy())
        y_pred = np.concatenate(preds)
        y_test = np.concatenate(targets).ravel()

    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


# ---------------------------------------------------------------------------
# Model / Config compatibility matrix
# ---------------------------------------------------------------------------


def get_compatible_configs(model_name: str) -> List[str]:
    """
    Return dataset configs that a model can plausibly consume.
    The actual compatibility is verified at runtime by InputPreparer.
    """
    spectrogram_models = {"ResNetAECNN", "ResNetVibCNN", "SelfSupervisedPretrainedCNN"}
    feature_models = {
        "RandomForestModel",
        "XGBoostModel",
        "LightGBMModel",
        "TabNetRegressor",
        "TabTransformerRegressor",
    }
    sequence_models = {"LSTMPhysicsModel", "TCNPhysicsModel"}
    strict_fusion = {
        "CrossModalTransformer",
        "PhysicsInformedFusionNet",
        "GatedMultimodalFusionNet",
        "FeatureOnlyModel",
    }
    flexible_fusion = {
        "MultiHeadAttentionFusion",
        "BilinearFusionNetwork",
        "GraphNeuralNetworkFusion",
        "TransformerEncoderFusion",
        "CrossAttentionFusionV1",
        "SqueezeExcitationFusionV1",
        "AttentionMLP",
    }
    legacy_fusion = {"FusionModel"}

    if model_name in spectrogram_models:
        return [c for c in VALID_DATASET_CONFIGS if "spec" in c]
    if model_name in feature_models:
        return [c for c in VALID_DATASET_CONFIGS if "spec" not in c]
    if model_name in sequence_models:
        # Can accept flat vectors; best with full physics or feature configs
        return [c for c in VALID_DATASET_CONFIGS if "spec" not in c] + [FULL_PHYSICS_CONFIG]
    if model_name in strict_fusion:
        return [FULL_PHYSICS_CONFIG]
    if model_name in flexible_fusion:
        # Can work with any config that provides at least one modality they support
        return VALID_DATASET_CONFIGS + [FULL_PHYSICS_CONFIG]
    if model_name in legacy_fusion:
        return [FULL_PHYSICS_CONFIG]

    # Default: try all standard configs plus full_physics
    return VALID_DATASET_CONFIGS + [FULL_PHYSICS_CONFIG]


def select_batch_size(model_name: str) -> int:
    """Heuristic batch size based on model complexity."""
    if model_name in ("ResNetAECNN", "ResNetVibCNN", "CrossModalTransformer"):
        return 16
    if model_name == "TransferLearningFusionNet":
        return 8  # ResNet18 at 224x224 is heavy on CPU memory
    if model_name in ("RandomForestModel", "XGBoostModel", "LightGBMModel"):
        return 64  # sklearn can handle larger batches for data collection
    return 32


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


def run_single_experiment(
    model_name: str,
    config: str,
    full_data: Dict[str, Optional[np.ndarray]],
    splitter: CVSplitter,
    device: torch.device,
    epochs: int = 200,
    patience: int = 20,
    save_checkpoints: bool = True,
    scale_specs: bool = False,
    scale_target: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run one model × config combination across all CV folds.
    Returns a result dict or None if the experiment fails.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Config: {config}")
    print(f"{'='*60}")

    # Determine compatible dataset keys
    if config == FULL_PHYSICS_CONFIG:
        dataset_cls = FullPhysicsDataset
        data_keys = FullPhysicsDataset.ALL_KEYS
    else:
        # Use a concrete class to avoid lambda pickling issues with DataLoader
        class _ConfigDataset(SimpleGrindingDataset):
            def __init__(self, data_dict, indices):
                super().__init__(data_dict, indices, config)
        dataset_cls = _ConfigDataset
        data_keys = list(parse_config(config))

    fold_results: List[Dict[str, Any]] = []
    groups = full_data["condition_ids"]
    source_artifacts: Dict[str, List[str]] = {"predictions": [], "checkpoints": []}
    start_time = time.time()

    try:
        model_template = model_factory(model_name)
        n_params = count_parameters(model_template) if not is_sklearn_wrapper(model_template) else 0
        print(f"  Parameters: {n_params:,}")
    except Exception as exc:
        print(f"  [SKIP] Failed to instantiate model: {exc}")
        return None

    preparer = InputPreparer(model_template)
    print(f"  Forward args: {preparer.forward_args}")
    print(f"  Required args: {preparer.required_args}")

    # Verify that required data modalities exist in full_data
    required_data_keys: set = set()
    KEY_MAP = {"physics": "physics_vector"}
    for arg in preparer.required_args:
        if arg == "x":
            required_data_keys.update(KEY_MAP.get(k, k) for k in data_keys)
        elif arg == "ae_spec":
            required_data_keys.add("ae_spec")
        elif arg == "vib_spec":
            required_data_keys.add("vib_spec")
        elif arg == "physics":
            required_data_keys.add("physics_vector")
        elif arg == "params":
            required_data_keys.add("pp")

    for key in required_data_keys:
        if full_data.get(key) is None:
            print(f"  [SKIP] Missing data for key '{key}'")
            return None

    for repeat_idx, fold_idx, train_idx, val_idx, test_idx in splitter.split(groups):
        total_folds_display = splitter.total_folds
        fold_label = fold_idx + 1
        if splitter.n_repeats > 1:
            fold_label = f"{fold_idx + 1} (repeat {repeat_idx + 1})"
        print(f"  Fold {fold_label}/{total_folds_display} ...", end=" ")
        sys.stdout.flush()

        try:
            # Scale data
            scaled, target_scaler = scale_data_dict(
                full_data, train_idx, scale_specs=scale_specs, scale_target=scale_target
            )

            # Build datasets
            train_ds = dataset_cls(scaled, train_idx)
            val_ds = dataset_cls(scaled, val_idx)
            test_ds = dataset_cls(scaled, test_idx)

            batch_size = select_batch_size(model_name)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Instantiate fresh model
            model = model_factory(model_name)

            # Verify input compatibility on first batch
            sample_batch = next(iter(train_loader))
            sample_inputs, _, _ = sample_batch
            kwargs, missing = preparer.prepare(sample_inputs, device)
            if missing:
                print(f"[SKIP] Missing required args: {missing}")
                return None

            # Train
            if is_sklearn_wrapper(model):
                val_mae = train_sklearn_model(model, train_loader, val_loader)
            else:
                val_mae, _ = train_pytorch_model(
                    model, train_loader, val_loader, preparer, device,
                    epochs=epochs, patience=patience,
                )

            # Evaluate on test set
            metrics = evaluate_model(model, test_loader, preparer, device, target_scaler=target_scaler)
            metrics["fold"] = fold_idx
            metrics["repeat"] = repeat_idx
            metrics["val_mae"] = val_mae
            if target_scaler is not None:
                metrics["target_scale_mean"] = float(target_scaler.mean_[0])
                metrics["target_scale_std"] = float(target_scaler.scale_[0])
            else:
                metrics["target_scale_mean"] = 0.0
                metrics["target_scale_std"] = 1.0
            fold_results.append(metrics)

            # Save per-fold predictions
            pred_df = pd.DataFrame({
                "y_true": metrics["y_true"],
                "y_pred": metrics["y_pred"],
            })
            pred_filename = f"{model_name}_{config.replace('+', '_')}_fold{fold_idx}_repeat{repeat_idx}.csv"
            pred_path = PREDICTIONS_DIR / pred_filename
            pred_df.to_csv(pred_path, index=False)
            source_artifacts["predictions"].append(str(pred_path))

            # Save checkpoint
            if save_checkpoints:
                ckpt_name = f"{model_name}_{config.replace('+', '_')}_fold{fold_idx}_repeat{repeat_idx}.pt"
                ckpt_path = CHECKPOINT_DIR / ckpt_name
                if is_sklearn_wrapper(model):
                    ckpt_path = ckpt_path.with_suffix(".pkl")
                    with open(ckpt_path, "wb") as f:
                        pickle.dump(model, f)
                else:
                    torch.save(model.state_dict(), ckpt_path)
                source_artifacts["checkpoints"].append(str(ckpt_path))

            print(f"Test MAE={metrics['mae']:.4f} MSE={metrics['mse']:.4f} R2={metrics['r2']:.4f}")

        except Exception as exc:
            print(f"[FAIL] {exc}")
            # Continue to next fold instead of aborting whole experiment
            continue

    if not fold_results:
        print(f"  [SKIP] No successful folds.")
        return None

    # Aggregate across folds
    maes = [f["mae"] for f in fold_results]
    mses = [f["mse"] for f in fold_results]
    r2s = [f["r2"] for f in fold_results]

    result = {
        "model": model_name,
        "config": config,
        "n_params": n_params,
        "n_folds_completed": len(fold_results),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "folds": fold_results,
    }
    print(f"  Aggregated -> MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")

    # Save CVResult JSON
    split_strategy = "GroupKFold" if splitter.grouped else "KFold"
    if splitter.logo:
        split_strategy = "LOGO"
    cv_result = {
        "model_name": model_name,
        "model_key": model_name,
        "validation_design": {
            "n_outer": splitter.n_folds,
            "n_inner": splitter.n_repeats,
            "total_folds": splitter.total_folds,
            "split_strategy": split_strategy,
        },
        "metrics": {
            "mean_mae": result["mae_mean"],
            "std_mae": result["mae_std"],
            "mean_mse": result["mse_mean"],
            "std_mse": result["mse_std"],
            "mean_r2": result["r2_mean"],
            "std_r2": result["r2_std"],
        },
        "runtime": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(time.time() - start_time, 2),
        },
        "fold_scale_factors": [
            {
                "fold": f["fold"],
                "repeat": f["repeat"],
                "target_mean": f.get("target_scale_mean", 0.0),
                "target_std": f.get("target_scale_std", 1.0),
            }
            for f in fold_results
        ],
        "source_artifacts": source_artifacts,
    }
    cv_path = REPORTS_METRICS_DIR / f"cv_results_{model_name}_{config.replace('+', '_')}.json"
    with open(cv_path, "w") as f:
        json.dump(cv_result, f, indent=2)
    print(f"  Saved CV result to {cv_path}")

    return result


# ---------------------------------------------------------------------------
# Ablation study helpers
# ---------------------------------------------------------------------------


def build_ablation_tables(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Generate DataFrames for ablation hypotheses from completed results."""
    if not results:
        return {}

    df = pd.DataFrame([
        {
            "model": r["model"],
            "config": r["config"],
            "mae_mean": r["mae_mean"],
            "mae_std": r["mae_std"],
            "mse_mean": r["mse_mean"],
            "mse_std": r["mse_std"],
            "r2_mean": r["r2_mean"],
            "r2_std": r["r2_std"],
            "n_folds": r["n_folds_completed"],
        }
        for r in results
    ])

    tables: Dict[str, pd.DataFrame] = {}
    tables["full_matrix"] = df.copy()

    # H1: Fusion vs Single-modality
    # Fusion = configs with '+' ; Single = no '+'
    h1 = df.copy()
    h1["modality_count"] = h1["config"].apply(lambda c: len(parse_config(c)) if c != FULL_PHYSICS_CONFIG else 5)
    h1["category"] = h1["modality_count"].apply(lambda n: "fusion" if n > 1 else "single")
    tables["H1_fusion_vs_single"] = (
        h1.groupby(["model", "category"])
        .agg({"mae_mean": ["mean", "std", "count"]})
        .reset_index()
    )

    # H2: Physics-informed proxy (physics_vector vs feature-only)
    h2 = df[df["config"].isin([FULL_PHYSICS_CONFIG, "ae_features+vib_features+pp"])].copy()
    h2["physics_informed"] = h2["config"].apply(
        lambda c: "with_physics_vector" if c == FULL_PHYSICS_CONFIG else "without_physics_vector"
    )
    tables["H2_physics_informed"] = h2[["model", "physics_informed", "mae_mean", "r2_mean"]].copy()

    # H3: Spectrogram vs Features
    h3 = df.copy()
    def _spectrogram_vs_features(cfg: str) -> str:
        if cfg == FULL_PHYSICS_CONFIG:
            return "full"
        has_spec = "spec" in cfg
        has_feat = "features" in cfg or cfg == "pp"
        if has_spec and not has_feat:
            return "spectrogram_only"
        if has_feat and not has_spec:
            return "feature_only"
        if has_spec and has_feat:
            return "mixed"
        return "other"
    h3["category"] = h3["config"].apply(_spectrogram_vs_features)
    tables["H3_spectrogram_vs_features"] = (
        h3.groupby(["model", "category"])
        .agg({"mae_mean": ["mean", "std", "count"]})
        .reset_index()
    )

    # H5: Edge deployment (FeatureOnlyModel vs full fusion models)
    h5 = df[df["model"].isin(["FeatureOnlyModel", "CrossModalTransformer", "GatedMultimodalFusionNet"])].copy()
    tables["H5_edge_deployment"] = h5[["model", "config", "mae_mean", "r2_mean", "n_folds"]].copy()

    return tables


# ---------------------------------------------------------------------------
# Smart data loading (avoids slow paths when possible)
# ---------------------------------------------------------------------------


def _needs_specs(selected_models: List[str], selected_configs: List[str]) -> bool:
    """Return True if any selected config actually includes spectrograms.
    Models that support spectrograms but are run on feature-only configs
    should not trigger spectrogram loading."""
    spectrogram_tokens = {"ae_spec", "vib_spec", "ae_mel", "vib_mel", "ae_logspec", "vib_logspec", "ae_wst", "vib_wst"}
    if any(any(t in c.split("+") for t in spectrogram_tokens) for c in selected_configs):
        return True
    if FULL_PHYSICS_CONFIG in selected_configs:
        return True
    if "TransferLearningFusionNet" in selected_models:
        return True
    return False


def _needs_physics_vector(selected_models: List[str], selected_configs: List[str]) -> bool:
    """Return True if any selected model/config needs the 44-D physics vector."""
    physics_models = {
        "CrossModalTransformer", "PhysicsInformedFusionNet",
        "GatedMultimodalFusionNet", "FeatureOnlyModel",
        "FusionModel", "LSTMPhysicsModel", "TCNPhysicsModel",
        "MultiHeadAttentionFusion", "BilinearFusionNetwork",
        "GraphNeuralNetworkFusion", "CrossAttentionFusionV1",
        "SqueezeExcitationFusionV1", "TransformerEncoderFusion",
        "AttentionMLP", "PhysicsGuidedAttentionNetwork",
        "HierarchicalAttentionNetwork", "TabTransformerRegressor",
    }
    if any(m in physics_models for m in selected_models):
        return True
    if FULL_PHYSICS_CONFIG in selected_configs:
        return True
    if any("physics" in c or c == "all" for c in selected_configs):
        return True
    return False


def load_physics_vector_fast() -> Dict[str, np.ndarray]:
    """
    Load only the aggregated physics_vector without touching spectrograms.
    This is orders of magnitude faster than load_all_data(config=None).
    """
    from grinding_physic_fusion.data.dataset import (
        INTERMEDIATE_DIR,
        MISSING_SAMPLE,
        aggregate_physics_features,
        discover_samples,
        load_process_parameters,
        load_surface_roughness,
    )

    pairs = discover_samples(config=None)
    params = load_process_parameters()
    roughness = load_surface_roughness()

    physics_vectors = []
    condition_ids = []
    sample_ids = []
    targets = []

    for cid, sid in pairs:
        if (cid, sid) == MISSING_SAMPLE:
            continue
        path = INTERMEDIATE_DIR / f"{cid}-{sid:02d}-0_physics.npz"
        data = np.load(path, allow_pickle=True)
        pv = aggregate_physics_features(data)
        data.close()
        physics_vectors.append(pv)
        condition_ids.append(cid)
        sample_ids.append(sid)
        targets.append(roughness[(cid - 1) * 20 + (sid - 1)])

    return {
        "physics_vector": np.stack(physics_vectors).astype(np.float32),
        "targets": np.array(targets, dtype=np.float32),
        "condition_ids": np.array(condition_ids, dtype=np.int64),
        "sample_ids": np.array(sample_ids, dtype=np.int64),
    }


def smart_load_data(
    selected_models: List[str],
    selected_configs: List[str],
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load only the data modalities that are actually needed.
    Merges outputs from multiple targeted load_all_data calls.
    """
    need_specs = _needs_specs(selected_models, selected_configs)
    need_physics = _needs_physics_vector(selected_models, selected_configs)
    need_features_pp = True  # Almost always needed; loads fast anyway

    full_data: Dict[str, Optional[np.ndarray]] = {}

    # 1. Fast feature + pp load (avoids spectrograms and raw-data fallback)
    if need_features_pp:
        print("Loading features + process parameters (fast path) ...")
        t0 = time.time()
        feat_data = load_all_data(config="ae_features+vib_features+pp")
        t1 = time.time()
        print(f"  -> {t1 - t0:.1f}s")
        full_data.update(feat_data)

    # 2. Spectrogram load (use cached mean spectrograms when available)
    if need_specs:
        cache_path = INTERMEDIATE_DIR / "cached_specs" / "mean_specs.npz"
        if cache_path.exists():
            print("Loading cached mean spectrograms ...")
            t0 = time.time()
            spec_cache = np.load(cache_path, allow_pickle=True)
            if not (
                np.array_equal(spec_cache["condition_ids"], full_data["condition_ids"])
                and np.array_equal(spec_cache["sample_ids"], full_data["sample_ids"])
            ):
                spec_cache.close()
                raise RuntimeError("Cached spectrograms are not aligned with feature data.")
            for k in ("ae_spec", "vib_spec"):
                full_data[k] = spec_cache[k]
            spec_cache.close()
            print(f"  -> {time.time() - t0:.1f}s")
        else:
            print("Loading spectrograms (slow; large npz files) ...")
            t0 = time.time()
            # Use a config that explicitly includes ae_features and vib_features
            # to avoid the raw-data fallback bug in load_single_sample.
            spec_data = load_all_data(config="ae_spec+ae_features+vib_spec+vib_features")
            t1 = time.time()
            print(f"  -> {t1 - t0:.1f}s")
            for k in ("ae_spec", "vib_spec"):
                full_data[k] = spec_data.get(k)

        # Load alternative time-frequency representations if any config requests them
        alt_tokens = {"ae_mel", "vib_mel", "ae_logspec", "vib_logspec"}
        requested_alt = set()
        for c in selected_configs:
            requested_alt.update(set(c.split("+")) & alt_tokens)
        if requested_alt:
            alt_cache_path = INTERMEDIATE_DIR / "cached_specs" / "alternative_reps.npz"
            if alt_cache_path.exists():
                print("Loading cached alternative representations ...")
                t0 = time.time()
                alt_cache = np.load(alt_cache_path, allow_pickle=True)
                for k in requested_alt:
                    if k in alt_cache:
                        full_data[k] = alt_cache[k]
                alt_cache.close()
                print(f"  -> {time.time() - t0:.1f}s")

        # Load Wavelet Scattering Transform features if requested
        wst_tokens = {"ae_wst", "vib_wst"}
        requested_wst = set()
        for c in selected_configs:
            requested_wst.update(set(c.split("+")) & wst_tokens)
        if requested_wst:
            wst_cache_path = INTERMEDIATE_DIR / "cached_specs" / "wst_features.npz"
            if wst_cache_path.exists():
                print("Loading cached WST features ...")
                t0 = time.time()
                wst_cache = np.load(wst_cache_path, allow_pickle=True)
                if not (
                    np.array_equal(wst_cache["condition_ids"], full_data["condition_ids"])
                    and np.array_equal(wst_cache["sample_ids"], full_data["sample_ids"])
                ):
                    wst_cache.close()
                    raise RuntimeError("Cached WST features are not aligned with feature data.")
                for k in requested_wst:
                    if k in wst_cache:
                        full_data[k] = wst_cache[k]
                wst_cache.close()
                print(f"  -> {time.time() - t0:.1f}s")

    # 3. Physics vector (fast custom loader)
    if need_physics:
        print("Loading physics vectors (fast custom loader) ...")
        t0 = time.time()
        phys_data = load_physics_vector_fast()
        t1 = time.time()
        print(f"  -> {t1 - t0:.1f}s")
        full_data.update(phys_data)

    # Ensure common keys exist
    for k in ("targets", "condition_ids", "sample_ids"):
        if k not in full_data or full_data[k] is None:
            raise RuntimeError(f"Missing essential key '{k}' after data loading.")

    return full_data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grinding Surface Roughness Prediction Pipeline")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help='Comma-separated model names or "all".',
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help='Comma-separated config names or "all".',
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5). Use 16 for LOGO.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Number of repeats for repeated K-fold CV (default: 1).",
    )
    parser.add_argument(
        "--grouped_cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GroupKFold split by condition_id (default: True). "
             "Use --no-grouped_cv to run ordinary KFold (leaks condition-level information).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Max training epochs (default: 200).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable saving model checkpoints.",
    )
    parser.add_argument(
        "--scale_specs",
        action="store_true",
        help="Enable scaling of spectrogram inputs.",
    )
    parser.add_argument(
        "--scale_target",
        action="store_true",
        default=True,
        help="Standardize target values using training-set statistics (default: True).",
    )
    parser.add_argument(
        "--no_scale_target",
        action="store_true",
        help="Disable target standardization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Select models
    # ------------------------------------------------------------------
    if args.models.lower() == "all":
        selected_models = DEFAULT_MODELS.copy()
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    # Validate
    invalid_models = [m for m in selected_models if m not in MODEL_REGISTRY]
    if invalid_models:
        print(f"Warning: unknown models ignored: {invalid_models}")
        selected_models = [m for m in selected_models if m in MODEL_REGISTRY]

    # ------------------------------------------------------------------
    # Select configs
    # ------------------------------------------------------------------
    if args.configs.lower() == "all":
        selected_configs = VALID_DATASET_CONFIGS.copy()
    else:
        selected_configs = [c.strip() for c in args.configs.split(",")]

    # ------------------------------------------------------------------
    # Load dataset (smart conditional loading)
    # ------------------------------------------------------------------
    try:
        full_data = smart_load_data(selected_models, selected_configs)
    except Exception as exc:
        print(f"Failed to load data: {exc}")
        sys.exit(1)

    n_samples = len(full_data["targets"])
    print(f"Total samples: {n_samples}")
    for k, v in sorted(full_data.items()):
        if v is not None and k not in ("targets", "condition_ids", "sample_ids"):
            print(f"  {k}: shape {v.shape}")

    # ------------------------------------------------------------------
    # CV splitter
    # ------------------------------------------------------------------
    logo = args.cv_folds == 16
    splitter = CVSplitter(
        n_folds=args.cv_folds,
        n_repeats=args.n_repeats,
        grouped=args.grouped_cv,
        logo=logo,
        seed=args.seed,
    )
    if logo:
        print(f"CV strategy: LOGO (16 folds)")
    else:
        strategy = "GroupKFold" if args.grouped_cv else "KFold"
        if not args.grouped_cv:
            print(
                "WARNING: Using ordinary KFold on this dataset leaks condition-level "
                "information across train/test folds. Use --grouped_cv or --cv_folds 16."
            )
        if args.n_repeats > 1:
            print(f"CV strategy: {args.cv_folds}-fold {strategy} repeated {args.n_repeats} times ({splitter.total_folds} total folds)")
        else:
            print(f"CV strategy: {args.cv_folds}-fold {strategy}")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []
    total_experiments = len(selected_models) * len(selected_configs)
    experiment_counter = 0

    for model_name in selected_models:
        compatible = get_compatible_configs(model_name)
        for config in selected_configs:
            experiment_counter += 1
            print(f"\n[{experiment_counter}/{total_experiments}] Running {model_name} / {config}")

            # Skip configs that are obviously incompatible
            if config not in compatible and config != FULL_PHYSICS_CONFIG:
                # Still try – runtime check will catch real incompatibilities
                pass

            scale_target = args.scale_target and not args.no_scale_target
            result = run_single_experiment(
                model_name=model_name,
                config=config,
                full_data=full_data,
                splitter=splitter,
                device=DEVICE,
                epochs=args.epochs,
                patience=args.patience,
                save_checkpoints=not args.no_checkpoints,
                scale_specs=args.scale_specs,
                scale_target=scale_target,
            )
            if result is not None:
                all_results.append(result)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = REPORTS_METRICS_DIR / "training_results.json"
    existing_results = []
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []
    # Deduplicate by model+config
    seen = set()
    deduped = []
    for r in existing_results + all_results:
        key = (r.get("model"), r.get("config"))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    with open(results_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # ------------------------------------------------------------------
    # Generate tables
    # ------------------------------------------------------------------
    tables = build_ablation_tables(all_results)
    for name, table in tables.items():
        csv_path = REPORTS_TABLES_DIR / f"{name}.csv"
        table.to_csv(csv_path, index=False)
        print(f"Saved table '{name}' to {csv_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    if all_results:
        summary_df = pd.DataFrame([
            {
                "model": r["model"],
                "config": r["config"],
                "mae": f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}",
                "mse": f"{r['mse_mean']:.4f} ± {r['mse_std']:.4f}",
                "r2": f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}",
                "folds": r["n_folds_completed"],
            }
            for r in all_results
        ])
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
    else:
        print("\nNo successful experiments completed.")


if __name__ == "__main__":
    main()
