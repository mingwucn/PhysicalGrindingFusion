#!/usr/bin/env python3
# Output mapping: see docs/figure_script_toc.md
"""
CPU inference latency benchmark for the top-performing model/config pairs.

For each top model/config, uses the first available fold checkpoint
(fold0_repeat0) and measures single-sample inference latency on CPU.

Outputs:
    reports/evidence/tables/edge_latency_benchmark.csv
    reports/evidence/plots/latency_vs_accuracy.png
"""

from __future__ import annotations

import os

# Pin CPU threading for reproducible single-core inference timing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.data.dataset import (
    GrindingDataset,
    INTERMEDIATE_DIR,
    aggregate_physics_features,
    get_roughness_for_sample,
    load_all_data,
    load_process_parameters,
    load_surface_roughness,
    parse_config,
)
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from train_and_evaluate import is_sklearn_wrapper, model_factory, prepare_sklearn_array

# Single-threaded PyTorch inference for reproducible CPU timing.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

PublicationPlotter.set_style()

CHECKPOINT_DIR = ROOT / "checkpoints"
TABLES_DIR = ROOT / "reports" / "evidence" / "tables"
PLOTS_DIR = ROOT / "reports" / "evidence" / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 1000
N_WARMUP = 100


def load_benchmark_data(config: str) -> Dict[str, Any]:
    """Fast cache-based loader for inference benchmarking.

    Avoids loading the large per-sample raw ``*_spec.npz`` files; instead it
    uses the pre-computed mean-spec and alternative-representation caches.
    """
    parts = parse_config(config)
    process_params = load_process_parameters()
    roughness_array = load_surface_roughness()

    data: Dict[str, Any] = {}

    # Mean raw spectrograms (always available for the benchmark cache)
    mean_cache_path = INTERMEDIATE_DIR / "cached_specs" / "mean_specs.npz"
    if mean_cache_path.exists():
        with np.load(mean_cache_path, allow_pickle=True) as mean_cache:
            data["condition_ids"] = np.asarray(mean_cache["condition_ids"])
            data["sample_ids"] = np.asarray(mean_cache["sample_ids"])
            if "ae_spec" in parts or config is None:
                data["ae_spec"] = np.asarray(mean_cache["ae_spec"])
            if "vib_spec" in parts or config is None:
                data["vib_spec"] = np.asarray(mean_cache["vib_spec"])
    else:
        return load_all_data(config=config)

    # Alternative time-frequency representations (log-mel, mel, logspec)
    alt_tokens = {"ae_mel", "vib_mel", "ae_logspec", "vib_logspec"}
    requested_alt = parts & alt_tokens
    if requested_alt:
        alt_cache_path = INTERMEDIATE_DIR / "cached_specs" / "alternative_reps.npz"
        if alt_cache_path.exists():
            with np.load(alt_cache_path, allow_pickle=True) as alt_cache:
                for token in requested_alt:
                    if token in alt_cache:
                        data[token] = np.asarray(alt_cache[token])
        else:
            return load_all_data(config=config)

    # WST representations
    wst_tokens = {"ae_wst", "vib_wst"}
    requested_wst = parts & wst_tokens
    if requested_wst:
        wst_cache_path = INTERMEDIATE_DIR / "cached_specs" / "wst_features.npz"
        if wst_cache_path.exists():
            with np.load(wst_cache_path, allow_pickle=True) as wst_cache:
                for token in requested_wst:
                    if token in wst_cache:
                        data[token] = np.asarray(wst_cache[token])
        else:
            return load_all_data(config=config)

    # Process parameters
    if "pp" in parts or config == "pp":
        data["pp"] = process_params[data["condition_ids"] - 1]

    # Physics features are stored in small per-sample files; load them
    # directly rather than falling back to the heavy raw-spec loader.
    if "physics" in parts:
        physics_vectors = []
        for cid, sid in zip(data["condition_ids"], data["sample_ids"]):
            physics_path = INTERMEDIATE_DIR / f"{int(cid)}-{int(sid):02d}-0_physics.npz"
            if physics_path.exists():
                with np.load(physics_path, allow_pickle=True) as pd:
                    physics_vectors.append(aggregate_physics_features(pd))
            else:
                physics_vectors.append(np.zeros(44, dtype=np.float32))
        data["physics_vector"] = np.stack(physics_vectors)

    # Targets
    data["targets"] = np.array(
        [
            get_roughness_for_sample(roughness_array, int(cid), int(sid))
            for cid, sid in zip(data["condition_ids"], data["sample_ids"])
        ],
        dtype=np.float32,
    )

    # Fallback for any modality not provided by the caches (e.g. time-domain
    # features that are not cached).
    token_to_key = {
        "ae_spec": "ae_spec",
        "vib_spec": "vib_spec",
        "ae_mel": "ae_mel",
        "vib_mel": "vib_mel",
        "ae_logspec": "ae_logspec",
        "vib_logspec": "vib_logspec",
        "ae_wst": "ae_wst",
        "vib_wst": "vib_wst",
        "ae_features": "ae_features",
        "vib_features": "vib_features",
        "physics": "physics_vector",
        "pp": "pp",
    }
    needed = {token_to_key[t] for t in parts} - {"pp"} - set(data.keys())
    if needed:
        full = load_all_data(config=config)
        for key in needed:
            if key in full and full[key] is not None:
                data[key] = full[key]
        # Ensure shared metadata is present from the cache-backed loader
        for meta_key in ("targets", "condition_ids", "sample_ids"):
            if meta_key not in data or data[meta_key] is None:
                data[meta_key] = full[meta_key]

    return data


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
    # Fallback: any checkpoint for this model/config
    for c in CHECKPOINT_DIR.glob(f"{model}_{cfg_file}*"):
        return c
    return None


def count_parameters(model: Any) -> int:
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    return 0


def benchmark_sklearn(model: Any, X_sample: np.ndarray, n_runs: int = N_RUNS) -> Dict[str, float]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N_WARMUP):
            model.predict(X_sample)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.predict(X_sample)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "p99_ms": float(np.quantile(times, 0.99)),
    }


def benchmark_pytorch(model: torch.nn.Module, sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]], n_runs: int = N_RUNS) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(**sample_input) if isinstance(sample_input, dict) else model(sample_input)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(**sample_input) if isinstance(sample_input, dict) else model(sample_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "p99_ms": float(np.quantile(times, 0.99)),
    }


def main() -> int:
    full_results = TABLES_DIR / "full_results_logo_only.csv"
    if not full_results.exists():
        print("full_results_logo_only.csv not found")
        return 1

    df = pd.read_csv(full_results)

    # Representative model--configuration pairs for the deployment table.
    representative = [
        ("RandomForestModel", "ae_logspec+vib_logspec"),
        ("RandomForestModel", "ae_spec+vib_spec"),
        ("LightGBMModel", "vib_logspec"),
        ("ResNetVibCNN", "vib_spec"),
        ("ResNetAECNN", "ae_spec"),
        ("ChannelAttentionCNN", "ae_spec"),
        ("BilinearFusionNetwork", "ae_spec+vib_spec+physics+pp"),
    ]
    selected_rows: List[pd.Series] = []
    for model_name, config in representative:
        match = df[(df["model"] == model_name) & (df["config"] == config)]
        if match.empty:
            print(f"Representative entry not found: {model_name}/{config}")
            continue
        selected_rows.append(match.iloc[0])
    if not selected_rows:
        print("No representative entries found; falling back to top 8")
        selected_rows = [r for _, r in df.nsmallest(8, "mae_mean").iterrows()]

    selected = pd.DataFrame(selected_rows)

    # Load dummy data for sample construction
    sample_config = selected.iloc[0]["config"]
    data_dict = load_benchmark_data(sample_config)

    rows: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        model_name = row["model"]
        config = row["config"]
        print(f"Benchmarking {model_name}/{config} ...")
        ckpt = find_checkpoint(model_name, config)
        if ckpt is None:
            print(f"  Checkpoint not found for {model_name}/{config}")
            continue

        try:
            if ckpt.suffix == ".pkl":
                with open(ckpt, "rb") as f:
                    model = pickle.load(f)
                # Build sample input
                cfg_data = load_benchmark_data(config)
                test_ds = GrindingDataset(cfg_data, config, [0])
                inputs_dict, _, _ = test_ds[0]
                X_sample = prepare_sklearn_array({k: v.unsqueeze(0) for k, v in inputs_dict.items()})
                bench = benchmark_sklearn(model, X_sample)
                n_params = 0
            else:
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                model_kwargs = state.get("model_kwargs", {}) if isinstance(state, dict) else {}
                model_state = state.get("model_state", state) if isinstance(state, dict) else state
                model = model_factory(model_name, **model_kwargs)
                model.load_state_dict(model_state)
                cfg_data = load_benchmark_data(config)
                test_ds = GrindingDataset(cfg_data, config, [0])
                inputs_dict, _, _ = test_ds[0]
                sample_input = {k: v.unsqueeze(0) for k, v in inputs_dict.items()}
                # Use InputPreparer to map to model forward
                from train_and_evaluate import InputPreparer
                preparer = InputPreparer(model)
                kwargs, missing = preparer.prepare(sample_input, torch.device("cpu"))
                if missing:
                    print(f"Missing args for {model_name}/{config}: {missing}")
                    continue
                bench = benchmark_pytorch(model, kwargs)
                n_params = count_parameters(model)
        except Exception as exc:
            print(f"Benchmark failed for {model_name}/{config}: {exc}")
            import traceback
            traceback.print_exc()
            continue

        bench.update({
            "model": model_name,
            "config": config,
            "mae_mean": row["mae_mean"],
            "mae_std": row["mae_std"],
            "n_parameters": n_params,
            "checkpoint_size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
            "checkpoint_path": str(ckpt),
        })
        rows.append(bench)

    if not rows:
        print("No benchmarks succeeded")
        return 1

    out = pd.DataFrame(rows)
    cols = ["model", "config", "mae_mean", "mae_std", "n_parameters", "checkpoint_size_mb",
            "median_ms", "mean_ms", "std_ms", "p95_ms", "p99_ms"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(TABLES_DIR / "edge_latency_benchmark.csv", index=False)

    out = out.sort_values(["median_ms", "mae_mean"]).reset_index(drop=True)
    out["plot_id"] = np.arange(1, len(out) + 1)

    managed = MutableFigure("latency_vs_accuracy.png", profile=FigureProfiles.DOUBLE, out_dir=PLOTS_DIR, overleaf_dir=ROOT / "overleaf" / "images", metadata={"generator": "scripts/benchmark_latency.py"})
    fig, ax = managed.create()
    ax.errorbar(out["median_ms"], out["mae_mean"], yerr=out["mae_std"], fmt="o", ms=7, capsize=3, color=PublicationPalette.OBSERVED)
    for _, r in out.iterrows():
        ax.text(r["median_ms"], r["mae_mean"], str(int(r["plot_id"])),
                fontsize=7, ha="center", va="center", color="white", fontweight="bold")
    legend = "\n".join(
        f"{int(r['plot_id'])}: {r['model'].replace('Model', '')} / {r['config']}"
        for _, r in out.iterrows()
    )
    ax.text(1.02, 0.5, legend, transform=ax.transAxes, va="center", ha="left", fontsize=6)
    ax.set_xlabel("Median inference latency (ms)")
    ax.set_ylabel("Mean MAE (µm)")
    ax.set_title("Latency vs. accuracy trade-off")
    ax.set_xlim(left=0, right=float(out["median_ms"].max()) * 1.15)
    fig.subplots_adjust(right=0.68)
    managed.save()

    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
