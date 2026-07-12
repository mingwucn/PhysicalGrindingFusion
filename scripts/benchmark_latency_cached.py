#!/usr/bin/env python3
"""
CPU inference benchmark using cached mean/alternative/WST features.

Avoids loading the huge per-sample _spec.npz files, so it can measure latency
for spectrogram-based models quickly.

Outputs:
    reports/evidence/tables/edge_latency_benchmark.csv
    reports/evidence/tables/edge_latency_benchmark_per_fold.csv
    reports/evidence/plots/latency_vs_accuracy.png

The benchmark pools timings over the 16 canonical repeat-0 outer-fold
checkpoints so that each aggregate latency row refers to the same fitted-model
family as its 16-fold benchmark MAE.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from train_and_evaluate import model_factory, prepare_sklearn_array

PublicationPlotter.set_style()

CHECKPOINT_DIR = ROOT / "checkpoints"
TABLES_DIR = ROOT / "reports" / "evidence" / "tables"
PLOTS_DIR = ROOT / "reports" / "evidence" / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CACHEDIR = ROOT / "data" / "intermediate" / "cached_specs"
N_BLOCKS = 5
BLOCK_RUNS = 200


def load_caches() -> Dict[str, np.ndarray]:
    caches: Dict[str, np.ndarray] = {}
    with np.load(CACHEDIR / "mean_specs.npz", allow_pickle=True) as d:
        caches["ae_spec"] = d["ae_spec"]
        caches["vib_spec"] = d["vib_spec"]
    with np.load(CACHEDIR / "alternative_reps.npz", allow_pickle=True) as d:
        caches["ae_logspec"] = d["ae_logspec"]
        caches["vib_logspec"] = d["vib_logspec"]
        caches["ae_mel"] = d["ae_mel"]
        caches["vib_mel"] = d["vib_mel"]
    with np.load(CACHEDIR / "wst_features.npz", allow_pickle=True) as d:
        caches["ae_wst"] = d["ae_wst"]
        caches["vib_wst"] = d["vib_wst"]
    return caches


def find_checkpoint(model: str, config: str) -> Path | None:
    cfg_file = config.replace("+", "_")
    candidates = [
        CHECKPOINT_DIR / f"{model}_{cfg_file}_fold0_repeat0.pkl",
        CHECKPOINT_DIR / f"{model}_{cfg_file}_fold0_repeat0.pt",
        CHECKPOINT_DIR / f"{model}_{cfg_file}_fold0.pt",
        CHECKPOINT_DIR / f"{model}_{cfg_file}_best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in CHECKPOINT_DIR.glob(f"{model}_{cfg_file}*"):
        return c
    return None


def find_fold_checkpoints(model: str, config: str, n_folds: int = 16) -> list[tuple[int, Path]]:
    """Return the canonical repeat-0 checkpoint for every outer LOGO fold."""
    cfg_file = config.replace("+", "_")
    checkpoints: list[tuple[int, Path]] = []
    for fold in range(n_folds):
        suffix = "pkl" if model in {"RandomForestModel", "LightGBMModel"} else "pt"
        path = CHECKPOINT_DIR / f"{model}_{cfg_file}_fold{fold}_repeat0.{suffix}"
        if not path.exists():
            return []
        checkpoints.append((fold, path))
    return checkpoints


def count_parameters(model: Any) -> int:
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    return 0


def force_single_thread(model: Any) -> None:
    """Pin wrapped sklearn/LightGBM estimators to one inference thread."""
    estimator = getattr(model, "model", model)
    if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return
    params = estimator.get_params(deep=False)
    updates = {name: 1 for name in ("n_jobs", "num_threads", "nthread") if name in params}
    if updates:
        estimator.set_params(**updates)


def build_sample(config: str, caches: Dict[str, np.ndarray], idx: int = 0) -> Dict[str, torch.Tensor]:
    parts = config.split("+")
    sample: Dict[str, torch.Tensor] = {}
    for token in parts:
        if token == "pp":
            # pp is (3,); use a dummy vector
            sample["pp"] = torch.tensor([25.0, 0.5, 0.02], dtype=torch.float32)
        elif token == "physics":
            sample["physics_vector"] = torch.zeros(1, 44, dtype=torch.float32)
        elif token in caches:
            arr = caches[token][idx]
            sample[token] = torch.from_numpy(arr).float().unsqueeze(0)
        else:
            # fallback zeros with a plausible shape
            shape = {
                "ae_features": (8,), "vib_features": (12,),
            }.get(token, (1,))
            sample[token] = torch.zeros(1, *shape, dtype=torch.float32)
    return sample


def _measure_times(func, n_blocks: int = N_BLOCKS, block_runs: int = BLOCK_RUNS) -> np.ndarray:
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        times: list[float] = []
        for _ in range(n_blocks):
            for _ in range(10):
                func()
            for _ in range(block_runs):
                t0 = time.perf_counter()
                func()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        return np.asarray(times, dtype=np.float64)
    finally:
        if gc_was_enabled:
            gc.enable()


def _summarise_times(times: np.ndarray) -> Dict[str, float]:
    return {
        "p50_ms": float(np.quantile(times, 0.50)),
        "median_ms": float(np.quantile(times, 0.50)),
        "p90_ms": float(np.quantile(times, 0.90)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "p99_ms": float(np.quantile(times, 0.99)),
        "max_ms": float(np.max(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
    }


def benchmark_sklearn(model: Any, X_sample: np.ndarray, n_blocks: int = N_BLOCKS, block_runs: int = BLOCK_RUNS) -> Dict[str, float]:
    times = _measure_times(lambda: model.predict(X_sample), n_blocks=n_blocks, block_runs=block_runs)
    return _summarise_times(times)


def benchmark_pytorch(model: torch.nn.Module, sample_input: Any, n_blocks: int = N_BLOCKS, block_runs: int = BLOCK_RUNS) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        def run_once():
            if isinstance(sample_input, dict):
                model(**sample_input)
            else:
                model(sample_input)

        times = _measure_times(run_once, n_blocks=n_blocks, block_runs=block_runs)
    return _summarise_times(times)


def render_latency_figure(out: pd.DataFrame) -> None:
    """Render a latency figure from a frozen benchmark table."""
    out = out.sort_values(["median_ms", "mae_mean"]).reset_index(drop=True).copy()
    out["plot_id"] = np.arange(1, len(out) + 1)
    managed = MutableFigure(
        "latency_vs_accuracy.png",
        profile=FigureProfiles.DOUBLE,
        out_dir=PLOTS_DIR,
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/benchmark_latency_cached.py", "mode": "render_only"},
    )
    fig, ax = managed.create()
    # p95 is an asymmetric latency tail, so encode it horizontally. Avoid
    # symmetric fold-SD bars for MAE because they can extend below zero and
    # are not confidence intervals.
    ax.hlines(
        out["mae_mean"],
        out["median_ms"],
        out["p95_ms"],
        color=PublicationPalette.NEUTRAL,
        linewidth=1.3,
        zorder=1,
        label="p50 point; right whisker = p95",
    )
    cap_half_height = 0.00035
    ax.vlines(
        out["p95_ms"],
        out["mae_mean"] - cap_half_height,
        out["mae_mean"] + cap_half_height,
        color=PublicationPalette.NEUTRAL,
        linewidth=1.3,
        zorder=1,
    )
    ax.scatter(
        out["median_ms"],
        out["mae_mean"],
        s=PublicationPlotter.POINT_SIZE,
        color=PublicationPalette.OBSERVED,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for _, row in out.iterrows():
        ax.text(row["median_ms"], row["mae_mean"], str(int(row["plot_id"])), fontsize=7, ha="center", va="center", color="white", fontweight="bold", zorder=4)
    config_labels = {
        "ae_logspec+vib_logspec": "AE-dB-z + Vib-dB-z",
        "ae_spec+vib_spec": "AE-dB + Vib-dB",
        "vib_logspec": "Vib-dB-z",
        "vib_spec": "Vib-dB",
        "ae_spec": "AE-dB",
        "vib_wst": "Vib-WST",
        "ae_spec+vib_spec+physics+process": "AE-dB + Vib-dB + physics + process",
        "ae_spec+vib_spec+physics+pp": "AE-dB + Vib-dB + physics + PP",
    }
    model_labels = {
        "RandomForestModel": "Random forest",
        "ResNetVibCNN": "ResNetVibCNN",
        "ResNetAECNN": "ResNetAECNN",
        "BilinearFusionNetwork": "Bilinear fusion",
        "LightGBMModel": "LightGBM",
    }
    key_labels = [
        f"{int(row['plot_id'])}: {model_labels.get(row['model'], row['model'].replace('Model', ''))} / "
        f"{config_labels.get(row['config'], row['config'])}"
        for _, row in out.iterrows()
    ]
    ax.set_xlabel("Median inference latency (ms)")
    ax.set_ylabel("Mean LOGO MAE (µm)")
    ax.set_title("Feature-precomputed latency vs. accuracy")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=float(out["p95_ms"].max()) * 1.15)
    ax.legend(loc="lower right", fontsize=PublicationPlotter.LEGEND_SIZE, frameon=False)
    midpoint = (len(key_labels) + 1) // 2
    fig.text(0.12, 0.02, "\n".join(key_labels[:midpoint]), ha="left", va="bottom", fontsize=6)
    fig.text(0.54, 0.02, "\n".join(key_labels[midpoint:]), ha="left", va="bottom", fontsize=6)
    fig.subplots_adjust(bottom=0.29)
    managed.save()


def main() -> int:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true", help="Render the canonical latency table without rerunning timings.")
    args = parser.parse_args()
    table_path = TABLES_DIR / "edge_latency_benchmark.csv"
    if args.render_only:
        render_latency_figure(pd.read_csv(table_path))
        return 0
    full_results = TABLES_DIR / "full_results_logo_only.csv"
    if not full_results.exists():
        print("full_results_logo_only.csv not found")
        return 1

    df = pd.read_csv(full_results)
    # diverse top models; prefer those with checkpoints
    candidates = [
        ("RandomForestModel", "ae_logspec+vib_logspec"),
        ("RandomForestModel", "ae_spec+vib_spec"),
        ("LightGBMModel", "vib_wst"),
        ("LightGBMModel", "vib_logspec"),
        ("ResNetVibCNN", "vib_spec"),
        ("ResNetAECNN", "ae_spec"),
        ("BilinearFusionNetwork", "ae_spec+vib_spec+physics+pp"),
    ]

    caches = load_caches()
    with np.load(CACHEDIR / "mean_specs.npz", allow_pickle=True) as metadata:
        condition_ids = np.asarray(metadata["condition_ids"], dtype=int)
    rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for model_name, config in candidates:
        # verify this result exists
        row_match = df[(df["model"] == model_name) & (df["config"] == config)]
        if row_match.empty:
            continue
        row = row_match.iloc[0]
        print(f"Benchmarking {model_name} / {config} ...", flush=True)
        fold_checkpoints = find_fold_checkpoints(model_name, config)
        if len(fold_checkpoints) != 16:
            print("  Complete 16-fold repeat-0 checkpoint set not found; skipping")
            continue

        try:
            all_times: list[np.ndarray] = []
            checkpoint_sizes: list[float] = []
            parameter_counts: list[int] = []
            for fold, ckpt in fold_checkpoints:
                test_condition = fold + 1
                test_indices = np.flatnonzero(condition_ids == test_condition)
                if len(test_indices) == 0:
                    raise ValueError(f"No cached sample for test condition {test_condition}")
                sample = build_sample(config, caches, idx=int(test_indices[0]))
                if ckpt.suffix == ".pkl":
                    with open(ckpt, "rb") as handle:
                        model = pickle.load(handle)
                    force_single_thread(model)
                    X_sample = prepare_sklearn_array(sample)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="X does not have valid feature names",
                            category=UserWarning,
                        )
                        times = _measure_times(lambda: model.predict(X_sample))
                    n_params = 0
                else:
                    state = torch.load(ckpt, map_location="cpu", weights_only=True)
                    model_kwargs = state.get("model_kwargs", {}) if isinstance(state, dict) else {}
                    model_state = state.get("model_state", state) if isinstance(state, dict) else state
                    model = model_factory(model_name, **model_kwargs)
                    model.load_state_dict(model_state)
                    from train_and_evaluate import InputPreparer
                    preparer = InputPreparer(model)
                    kwargs, missing = preparer.prepare(sample, torch.device("cpu"))
                    if missing:
                        raise ValueError(f"Missing model inputs for fold {fold}: {missing}")
                    model.eval()
                    with torch.no_grad():
                        times = _measure_times(lambda: model(**kwargs))
                    n_params = count_parameters(model)
                fold_summary = _summarise_times(times)
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                fold_rows.append({
                    "model": model_name,
                    "config": config,
                    "fold": fold,
                    "test_condition": test_condition,
                    "sample_index": int(test_indices[0]),
                    "checkpoint_path": str(ckpt),
                    "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                    "checkpoint_size_mb": size_mb,
                    "timing_blocks": N_BLOCKS,
                    "runs_per_block": BLOCK_RUNS,
                    **fold_summary,
                })
                all_times.append(times)
                checkpoint_sizes.append(size_mb)
                parameter_counts.append(n_params)
            bench = _summarise_times(np.concatenate(all_times))
            n_params = parameter_counts[0]
            if any(value != n_params for value in parameter_counts):
                raise ValueError("Parameter count differs across fold checkpoints")
        except Exception as exc:
            print(f"  Benchmark failed: {exc}")
            import traceback
            traceback.print_exc()
            continue

        bench.update({
            "model": model_name,
            "config": config,
            "mae_mean": row["mae_mean"],
            "mae_std": row["mae_std"],
            "n_parameters": n_params,
            "checkpoint_count": len(fold_checkpoints),
            "checkpoint_size_mb": float(np.median(checkpoint_sizes)),
            "checkpoint_size_min_mb": float(np.min(checkpoint_sizes)),
            "checkpoint_size_max_mb": float(np.max(checkpoint_sizes)),
            "timing_scope": "pooled over 16 canonical repeat-0 LOGO checkpoints",
            "timing_blocks": N_BLOCKS,
            "runs_per_block": BLOCK_RUNS,
        })
        rows.append(bench)
        print(
            f"  p50={bench['p50_ms']:.3f} ms, p95={bench['p95_ms']:.3f} ms, max={bench['max_ms']:.3f} ms",
            flush=True,
        )

    if not rows:
        print("No benchmarks succeeded")
        return 1

    out = pd.DataFrame(rows)
    cols = ["model", "config", "mae_mean", "mae_std", "n_parameters", "checkpoint_size_mb",
            "checkpoint_count", "checkpoint_size_min_mb", "checkpoint_size_max_mb", "timing_scope",
            "timing_blocks", "runs_per_block", "p50_ms", "median_ms", "p90_ms", "p95_ms",
            "p99_ms", "max_ms", "mean_ms", "std_ms"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(TABLES_DIR / "edge_latency_benchmark.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(TABLES_DIR / "edge_latency_benchmark_per_fold.csv", index=False)

    render_latency_figure(out)

    print("\n" + out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
