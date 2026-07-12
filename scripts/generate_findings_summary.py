#!/usr/bin/env python3
"""
Generate a concise findings summary markdown from the authoritative per-campaign
cv_results_*.json files (single-repeat LOGO metrics). This avoids mixing repeats
that are aggregated in full_results.csv.

Usage:
    python scripts/generate_findings_summary.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "reports" / "evidence" / "metrics"
TABLES_DIR = PROJECT_ROOT / "reports" / "evidence" / "tables"
OUT_PATH = PROJECT_ROOT / "reports" / "evidence" / "findings_summary.md"

ML_MODELS = {"RandomForestModel", "XGBoostModel", "LightGBMModel", "RidgeRegressionModel", "ShallowMLPModel"}
DL_MODELS = {
    "ResNetVibCNN", "ResNetAECNN", "MultiscaleSpectrogramCNN", "ChannelAttentionCNN",
    "ResNetFusion", "BilinearFusionNetwork", "CrossModalTransformer",
    "PhysicsInformedFusionNet", "GatedMultimodalFusionNet",
    "MultiHeadAttentionFusion", "AttentionMLP", "FeatureOnlyModel",
    "PhysicsMLP", "ParamsMLP", "TabNetRegressor", "TabTransformerRegressor",
    "GraphNeuralNetworkFusion", "TransferFeatureMLP", "TrajectoryCNN",
}


def _config_from_path(path: Path, model_name: str) -> str:
    """Infer the '+'-separated config from the cv_results filename."""
    VALID_TOKENS = {"ae_spec", "vib_spec", "ae_logspec", "vib_logspec", "ae_mel", "vib_mel",
                    "ae_wst", "vib_wst", "ae_features", "vib_features", "physics", "pp", "all"}
    prefix = f"cv_results_{model_name}_"
    suffix = path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem
    parts = suffix.split("_")
    tokens = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            candidate = f"{parts[i]}_{parts[i+1]}"
            if candidate in VALID_TOKENS:
                tokens.append(candidate)
                i += 2
                continue
        if parts[i] in VALID_TOKENS:
            tokens.append(parts[i])
        i += 1
    return "+".join(tokens)


def load_cv_results() -> pd.DataFrame:
    rows = []
    for path in METRICS_DIR.glob("cv_results_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        model = data.get("model_name", "")
        metrics = data.get("metrics", {})
        config = data.get("config")
        if not config or config == model:
            config = _config_from_path(path, model)
        # Skip summaries / aggregates
        if not metrics or "mean_mae" not in metrics:
            continue
        rows.append({
            "model": model,
            "config": config,
            "mae_mean": float(metrics["mean_mae"]),
            "mae_std": float(metrics.get("std_mae", 0.0)),
            "r2_mean": float(metrics.get("mean_r2", 0.0)),
            "path": path.name,
        })
    return pd.DataFrame(rows)


def main() -> int:
    if not METRICS_DIR.exists():
        print("No metrics directory found.")
        return 1

    df = load_cv_results()
    if df.empty:
        print("No cv_results JSON files found.")
        return 1

    ml_mask = df["model"].isin(ML_MODELS)
    dl_mask = df["model"].isin(DL_MODELS)

    best_overall = df.loc[df["mae_mean"].idxmin()]
    best_ml = df.loc[df[ml_mask]["mae_mean"].idxmin()] if ml_mask.any() else None
    best_dl = df.loc[df[dl_mask]["mae_mean"].idxmin()] if dl_mask.any() else None

    lines = [
        "# Concise Findings Summary",
        "",
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "*Based on single-repeat LOGO cv_results JSONs (comparable across model families).*",
        "",
        "## Best results so far",
        "",
        "| Rank | Note | Model | Config | MAE (µm) | Std (µm) | R² |",
        "|------|------|-------|--------|----------|----------|-----|",
    ]

    def row_fmt(row: pd.Series) -> str:
        return (
            f"{row['model']} | `{row['config']}` | "
            f"{row['mae_mean']:.5f} | {row['mae_std']:.5f} | {row['r2_mean']:.3f} |"
        )

    lines.append(f"| 1 | Overall best | {row_fmt(best_overall)}")
    if best_ml is not None:
        lines.append(f"| 2 | Best transparent ML | {row_fmt(best_ml)}")
    if best_dl is not None:
        lines.append(f"| 3 | Best deep learning | {row_fmt(best_dl)}")

    lines.extend(["", "## Head-to-head comparison", ""])

    if best_ml is not None and best_dl is not None:
        delta = best_dl["mae_mean"] - best_ml["mae_mean"]
        pct = 100 * delta / best_ml["mae_mean"]
        lines.append(
            f"- **Best DL** (`{best_dl['model']} / {best_dl['config']}`): "
            f"MAE = {best_dl['mae_mean']:.5f} µm ± {best_dl['mae_std']:.5f} µm."
        )
        lines.append(
            f"- **Best ML** (`{best_ml['model']} / {best_ml['config']}`): "
            f"MAE = {best_ml['mae_mean']:.5f} µm ± {best_ml['mae_std']:.5f} µm."
        )
        lines.append(
            f"- **Gap:** DL is {delta:+.5f} µm ({pct:+.1f}%) relative to best ML."
        )
    else:
        lines.append("- Results are incomplete; rerun after all DL experiments finish.")

    lines.extend(["", "## Statistical significance", ""])
    stat_path = TABLES_DIR / "statistical_tests_updated.csv"
    if stat_path.exists() and stat_path.stat().st_size > 0:
        try:
            stat_df = pd.read_csv(stat_path)
            if not stat_df.empty:
                lines.append("| Comparison | MAE A | MAE B | Median Δ | p raw | p Holm | Significant (α=0.05) |")
                lines.append("|------------|-------|-------|----------|-------|--------|----------------------|")
                for _, r in stat_df.iterrows():
                    sig = "Yes" if r["significant_05"] else "No"
                    lines.append(
                        f"| {r['comparison']} | {r['mean_mae_a']:.5f} | {r['mean_mae_b']:.5f} | "
                        f"{r['median_diff']:.5f} | {r['pvalue_raw']:.4g} | {r['pvalue_holm']:.4g} | {sig} |"
                    )
            else:
                lines.append("- Statistical tests pending.")
        except Exception:
            lines.append("- Statistical tests pending.")
    else:
        lines.append("- Statistical tests pending.")

    lines.extend(["", "## Interpretability & uncertainty", ""])
    plots_dir = PROJECT_ROOT / "reports" / "evidence" / "plots"
    unc_dir = PROJECT_ROOT / "reports" / "evidence" / "uncertainty"
    shap_plots = list(plots_dir.glob("shap_summary_*.png"))
    gradcam_plots = list(plots_dir.glob("gradcam_*.png"))
    cal_plots = list(unc_dir.glob("mc_dropout_calibration_*.png"))
    lines.append(f"- SHAP summary plots: {len(shap_plots)} generated.")
    lines.append(f"- Grad-CAM overlays: {len(gradcam_plots)} generated.")
    lines.append(f"- MC-dropout calibration plots: {len(cal_plots)} generated.")

    lines.extend(["", "## Signal-representation comparison", ""])
    sig_path = TABLES_DIR / "signal_representation_comparison.csv"
    if sig_path.exists():
        try:
            sig_df = pd.read_csv(sig_path)
            # The comparison table stores MAE as a 'mean ± std' string.
            sig_df["mae_mean"] = sig_df["MAE (µm)"].astype(str).str.split("±").str[0].astype(float)
            sig_df["mae_std"] = sig_df["MAE (µm)"].astype(str).str.split("±").str[1].astype(float)
            top = sig_df.nsmallest(10, "mae_mean").reset_index(drop=True)
            lines.append("| Rank | Model | Config | MAE (µm) | Std (µm) | Representation |")
            lines.append("|------|-------|--------|----------|----------|----------------|")
            for i, r in top.iterrows():
                rep = "WST" if "wst" in r['config'] else "Spectrogram" if any(t in r['config'] for t in ['spec','mel','logspec']) else "Other"
                lines.append(
                    f"| {i+1} | {r['model']} | `{r['config']}` | {r['mae_mean']:.5f} | {r['mae_std']:.5f} | {rep} |"
                )
        except Exception:
            lines.append("- Signal-representation comparison pending.")
    else:
        lines.append("- Signal-representation comparison pending.")

    lines.extend(["", "## Condition-level failure analysis", ""])
    cond_path = TABLES_DIR / "condition_error_ranking.csv"
    if cond_path.exists():
        try:
            cond_df = pd.read_csv(cond_path)
            worst = cond_df.nlargest(5, "mae").reset_index(drop=True)
            lines.append("| Condition ID | Mean MAE (µm) |")
            lines.append("|--------------|---------------|")
            for _, r in worst.iterrows():
                lines.append(f"| {int(r['condition_id'])} | {r['mae']:.5f} |")
        except Exception:
            lines.append("- Condition error ranking pending.")
    else:
        lines.append("- Condition error ranking pending.")

    lines.extend(["", "## Take-away for the paper", ""])
    if best_dl is not None and best_ml is not None:
        if best_dl["mae_mean"] < best_ml["mae_mean"]:
            lines.append(
                f"A deep-learning model (`{best_dl['model']} / {best_dl['config']}`) now outperforms "
                f"the best transparent ML model (`{best_ml['model']} / {best_ml['config']}`) on this "
                "small-data grinding benchmark. The next step is to add interpretability and uncertainty "
                "evidence and frame the contribution as a rigorous, reproducible LOGO benchmark."
            )
        else:
            lines.append(
                f"Transparent ML (`{best_ml['model']} / {best_ml['config']}`) holds the lowest "
                f"MAE and outperforms the best DL model (`{best_dl['model']} / {best_dl['config']}`) by "
                f"{abs(delta):.5f} µm ({abs(pct):.1f}%). The paper should frame the contribution around "
                "physics-aware signal representations, transparent and interpretable models, and "
                "uncertainty-aware deployment rather than raw DL accuracy alone."
            )
    else:
        lines.append("Awaiting completed DL experiments before drawing conclusions.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
