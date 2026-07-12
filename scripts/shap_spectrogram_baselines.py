"""SHAP-style global explanations for linear/tree baselines on ae_spec+vib_spec."""
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Allow imports of both `src` and `scripts`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import torch

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

from scripts.train_and_evaluate import (
    smart_load_data,
    scale_data_dict,
    InputPreparer,
    model_factory,
)
from src.grinding_physic_fusion.data.dataset import load_all_data

CONFIG = "ae_spec+vib_spec"
MODELS = ["RidgeRegressionModel", "LightGBMModel"]
SEED = 42
N_BACKGROUND = 30
N_TEST = 50
OUT_DIR = Path("reports/evidence/xai")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()


def flatten_config(full_data: Dict[str, np.ndarray], config: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, ...]]]:
    """Flatten the modalities in `config` the same way InputPreparer does for sklearn `x`."""
    from src.grinding_physic_fusion.data.dataset import parse_config
    keys = sorted(parse_config(config))
    parts = []
    shapes = []
    for key in keys:
        t = torch.from_numpy(full_data[key])
        if t.dim() > 2:
            t = t.view(t.size(0), -1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        parts.append(t.numpy())
        shapes.append(full_data[key].shape[1:])  # (C, F, T)
    X = np.concatenate(parts, axis=1)
    return X, keys, shapes


def train_and_explain(model_name: str, X: np.ndarray, y: np.ndarray, key_shapes: List[Tuple[int, ...]], keys: List[str]):
    print(f"\n=== {model_name} ===", flush=True)
    model = model_factory(model_name)
    model.fit(X, y)

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X), size=min(N_BACKGROUND, len(X)), replace=False)
    test_idx = rng.choice(len(X), size=min(N_TEST, len(X)), replace=False)
    X_bg = X[bg_idx]
    X_test = X[test_idx]

    if model_name == "RidgeRegressionModel":
        # Global linear explanation: coefficient magnitudes
        importance = np.abs(model.model.coef_)
    elif model_name == "RandomForestModel":
        # TreeSHAP is too expensive for 38k spectrogram features with RF depth 8;
        # use the model's built-in Gini-based feature importance as a global proxy.
        importance = model.model.feature_importances_
    else:
        # TreeSHAP for LightGBM (shallow trees -> tractable)
        explainer = shap.TreeExplainer(model.model, X_bg)
        sv_test = explainer.shap_values(X_test)
        importance = np.abs(sv_test).mean(axis=0)

    # Map back to modalities
    records = []
    offset = 0
    for key, shape in zip(keys, key_shapes):
        n_features = int(np.prod(shape))
        imp = importance[offset : offset + n_features].reshape(shape)
        # Average importance over time for each (channel, freq)
        imp_time = np.abs(imp).mean(axis=-1)  # (C, F)
        for c in range(imp_time.shape[0]):
            for f in range(imp_time.shape[1]):
                records.append({
                    "modality": key,
                    "channel": c,
                    "freq_bin": f,
                    "importance": float(imp_time[c, f]),
                })
        offset += n_features

    df = pd.DataFrame(records)
    csv_path = OUT_DIR / f"shap_importance_{model_name}_{CONFIG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

    # Top-20 bar plot per modality
    top_parts = []
    for key in keys:
        sub = df[df["modality"] == key].nlargest(20, "importance")
        top_parts.append(sub)
    top = pd.concat(top_parts, ignore_index=True)
    if len(keys) != 2:
        raise ValueError("The final SHAP baseline figure requires AE and vibration panels.")
    managed = MutableFigure(
        f"shap_importance_{model_name}_{CONFIG}.png",
        profile=FigureProfiles.TWO_PANEL_ROW,
        out_dir=OUT_DIR,
        overleaf_dir=Path("overleaf/images"),
        metadata={"generator": "scripts/shap_spectrogram_baselines.py"},
    )
    fig, axes = managed.create()
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        sub = top[top["modality"] == key].sort_values("importance", ascending=True)
        ylabels = [f"ch{r['channel']}_f{r['freq_bin']}" for _, r in sub.iterrows()]
        ax.barh(range(len(sub)), sub["importance"].values, color=PublicationPalette.model(model_name))
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(ylabels)
        ax.set_title(f"{model_name} / {key}")
        ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    managed.save()
    print(f"Saved SHAP plot for {model_name}", flush=True)

    return df


def main():
    print("Loading data ...", flush=True)
    full_data = smart_load_data(MODELS, [CONFIG])
    print("Scaling ...", flush=True)
    scaled, _ = scale_data_dict(full_data, np.arange(len(full_data["targets"])), scale_specs=True, scale_target=False)
    X, keys, shapes = flatten_config(scaled, CONFIG)
    y = scaled["targets"]
    print(f"X shape: {X.shape}, keys: {keys}, shapes: {shapes}", flush=True)

    for model_name in MODELS:
        try:
            train_and_explain(model_name, X, y, shapes, keys)
        except Exception as exc:
            print(f"Failed for {model_name}: {exc}", flush=True)

if __name__ == "__main__":
    main()
