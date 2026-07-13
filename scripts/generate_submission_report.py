#!/usr/bin/env python3
"""
Generate an updated publication-style report for the transparent-ML +
signal-processing submission pipeline (AEI/ESWA/TII).

Outputs:
    reports/publication/submission_report.md
    reports/evidence/plots/submission_*.png
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import (
    DenseRankingFigure,
    FigureProfiles,
    MutableFigure,
    PublicationPalette,
    PublicationPlotter,
    ThreePanelRowFigure,
)

REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "evidence" / "plots"
TABLES_DIR = REPORTS_DIR / "evidence" / "tables"
XAI_DIR = REPORTS_DIR / "evidence" / "xai"
UNC_DIR = REPORTS_DIR / "evidence" / "uncertainty"
METRICS_DIR = REPORTS_DIR / "evidence" / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "evidence" / "predictions"
PUBLICATION_DIR = REPORTS_DIR / "publication"

for d in (PLOTS_DIR, PUBLICATION_DIR):
    d.mkdir(parents=True, exist_ok=True)

REL_PLOTS = "../evidence/plots"
REL_TABLES = "../evidence/tables"
FIGURES: Dict[str, str] = {}

CV_STRATEGY_NAME = "LOGO"
CV_TOTAL_FOLDS = 16

PUBLICATION_CONFIG_LABELS = {
    "ae_logspec+vib_logspec": "AE-dB-z + Vib-dB-z",
    "ae_logspec+vib_logspec+pp": "AE-dB-z + Vib-dB-z + PP",
    "ae_mel+vib_mel": "AE-log-mel + Vib-log-mel",
    "ae_spec+vib_spec": "AE-dB + Vib-dB",
    "ae_spec+vib_spec+pp": "AE-dB + Vib-dB + PP",
    "vib_logspec": "Vib-dB-z",
    "vib_mel": "Vib-log-mel",
    "vib_spec": "Vib-dB",
    "ae_spec": "AE-dB",
}


def publication_config_label(config: str) -> str:
    return PUBLICATION_CONFIG_LABELS.get(config, config.replace("_", "-"))


def load_full_results() -> pd.DataFrame:
    path = TABLES_DIR / "full_results_logo_only.csv"
    df = pd.read_csv(path)
    df["folds"] = df["folds"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith("[") else []
    )
    for col in ["mae_mean", "mae_std", "mse_mean", "mse_std", "r2_mean", "r2_std"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_folds_completed"] = pd.to_numeric(df["n_folds_completed"], errors="coerce").fillna(0)
    return df


def get_git_info() -> Dict[str, str]:
    info = {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
    if shutil.which("git") is None:
        return info
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["dirty"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["dirty"] = "yes" if info["dirty"] else "no"
    except Exception:
        pass
    return info


def _set_style() -> None:
    PublicationPlotter.set_style()
    # Keep seaborn plots on a plain background; grids are added by individual
    # plotting calls and should be removed before publication.
    sns.set_theme(style="white", rc={"axes.grid": False})


def _new_submission_figure(name: str, profile):
    figure = MutableFigure(
        f"submission_{name}.png",
        profile=profile,
        out_dir=PLOTS_DIR,
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/generate_submission_report.py"},
    )
    fig, axes = figure.create()
    return figure, fig, axes


def _finish_submission_figure(name: str, figure: MutableFigure) -> str:
    figure.save()
    rel = f"{REL_PLOTS}/submission_{name}.png"
    FIGURES[name] = rel
    return rel


def _collect_predictions(row) -> tuple[np.ndarray, np.ndarray]:
    model = row["model"]
    config = row["config"]
    cfg = config.replace("+", "_")
    files = sorted(PREDICTIONS_DIR.glob(f"{model}_{cfg}_fold*_repeat*.csv"))
    if not files:
        return np.array([]), np.array([])
    y_true, y_pred = [], []
    for f in files:
        d = pd.read_csv(f)
        y_true.extend(d["y_true"].to_numpy(float))
        y_pred.extend(d["y_pred"].to_numpy(float))
    return np.asarray(y_true, float), np.asarray(y_pred, float)


CONFIG_ABBREV = {
    "ae_logspec+vib_logspec": "AE-dB-z + Vib-dB-z",
    "ae_logspec+vib_logspec+pp": "AE-dB-z + Vib-dB-z + PP",
    "ae_mel+vib_mel": "AE-log-mel + Vib-log-mel",
    "ae_mel+vib_mel+pp": "AE-log-mel + Vib-log-mel + PP",
    "ae_spec+vib_spec": "AE-dB + Vib-dB",
    "ae_spec+vib_spec+pp": "AE-dB + Vib-dB + PP",
    "ae_spec+vib_spec+physics": "AE-dB + Vib-dB + phys",
    "ae_spec+vib_spec+physics+pp": "AE-dB + Vib-dB + phys + PP",
    "ae_features+vib_features+physics+pp": "AE/Vib feats + phys + PP",
    "vib_logspec": "Vib-dB-z",
    "vib_mel": "Vib-log-mel",
    "vib_spec": "Vib-dB",
    "vib_wst": "Vib-WST",
    "vib_trajectory": "Vib-traj",
    "ae_logspec": "AE-dB-z",
    "ae_spec": "AE-dB",
    "pp": "Process params",
}

MODEL_ABBREV = {
    "RandomForestModel": "Random Forest",
    "LightGBMModel": "LightGBM",
    "RidgeRegressionModel": "Ridge",
    "ResNetVibCNN": "ResNetVibCNN",
    "ResNetAECNN": "ResNetAECNN",
    "ResNetFusion": "ResNetFusion",
    "BilinearFusionNetwork": "Bilinear fusion",
    "TabTransformerModel": "TabTransformer",
    "TrajectoryCNNModel": "Trajectory CNN",
    "TrajectoryCNN": "Trajectory CNN",
}


def _abbrev_config(config: str) -> str:
    return CONFIG_ABBREV.get(config, config)


def _abbrev_model(model: str) -> str:
    return MODEL_ABBREV.get(model, model)


class Top20RankingFigure(DenseRankingFigure):
    """Full-width dense ranking with final-size readable labels."""

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(
            "submission_ranking_top20.png",
            out_dir=PLOTS_DIR,
            overleaf_dir=ROOT / "overleaf" / "images",
            metadata={
                "generator": "scripts/generate_submission_report.py",
                "latex_width": "double-column / \\linewidth",
                "figure_type": "dense ranking",
            },
        )
        self.top20 = df.nsmallest(20, "mae_mean").copy().reset_index(drop=True)

    def draw(self) -> None:
        ax = self.ax
        self.top20["label"] = self.top20.apply(
            lambda row: PublicationPlotter.comparison_label(
                _abbrev_model(row["model"]), _abbrev_config(row["config"])
            ),
            axis=1,
        )
        y = np.arange(len(self.top20))
        colors = [PublicationPalette.model(model, i) for i, model in enumerate(self.top20["model"])]
        bars = ax.barh(
            y,
            self.top20["mae_mean"],
            xerr=self.top20["mae_std"],
            capsize=2,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_yticks(y, self.top20["label"])
        ax.tick_params(axis="y", labelsize=6, pad=2)
        ax.invert_yaxis()
        ax.set_xlabel("Mean LOGO MAE (µm)")
        ax.set_title(f"Historical canonical top 20 ({CV_STRATEGY_NAME}, {CV_TOTAL_FOLDS} folds)")
        x_max = float((self.top20["mae_mean"] + self.top20["mae_std"]).max())
        annotation_x = x_max * 1.08
        ax.set_xlim(0, x_max * 1.24)
        for bar, mae in zip(bars, self.top20["mae_mean"]):
            ax.text(
                annotation_x,
                bar.get_y() + bar.get_height() / 2,
                f"{mae:.4f}",
                va="center",
                ha="left",
                fontsize=6,
                color="black",
            )
        assert self.fig is not None
        self.fig.tight_layout(pad=0.4)


def fig_ranking_top20(df: pd.DataFrame) -> str:
    figure = Top20RankingFigure(df)
    figure.render()
    figure.save()
    rel = f"{REL_PLOTS}/submission_ranking_top20.png"
    FIGURES["ranking_top20"] = rel
    return rel


def fig_prediction_scatter_top3(df: pd.DataFrame) -> str:
    top3 = df.nsmallest(3, "mae_mean").reset_index(drop=True)
    managed, fig, axes = _new_submission_figure("prediction_scatter_top3", FigureProfiles.THREE_PANEL_ROW_SHARED)
    axes = axes.flatten()
    palette = [PublicationPalette.model(row["model"], i) for i, (_, row) in enumerate(top3.iterrows())]
    for ax, (_, row), color in zip(axes, top3.iterrows(), palette):
        y_true, y_pred = _collect_predictions(row)
        if len(y_true) == 0:
            ax.set_visible(False)
            continue
        ax.scatter(y_true, y_pred, alpha=0.45, s=30, color=color,
                   edgecolors="white", linewidth=0.3, label="Predictions")
        lims = [min(y_true.min(), y_pred.min()) - 0.02, max(y_true.max(), y_pred.max()) + 0.02]
        ax.plot(lims, lims, "k--", lw=1.5, label="Ideal")
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Measured Ra (µm)"); ax.set_ylabel("Predicted Ra (µm)")
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    managed.apply_panel_row_rules(axes)
    return _finish_submission_figure("prediction_scatter_top3", managed)


def fig_residual_distribution_top3(df: pd.DataFrame) -> str:
    top3 = df.nsmallest(3, "mae_mean").reset_index(drop=True)
    managed, fig, axes = _new_submission_figure("residual_distribution_top3", FigureProfiles.THREE_PANEL_ROW)
    axes = axes.flatten()
    palette = [PublicationPalette.model(row["model"], i) for i, (_, row) in enumerate(top3.iterrows())]
    for ax, (_, row), color in zip(axes, top3.iterrows(), palette):
        y_true, y_pred = _collect_predictions(row)
        if len(y_true) == 0:
            ax.set_visible(False)
            continue
        residuals = y_pred - y_true
        sns.histplot(residuals, kde=True, bins=35, color=color, ax=ax,
                     stat="density", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", lw=1.5)
        ax.set_xlabel("Residual = Predicted − Measured (µm)")
        ax.set_ylabel("Density")
    fig.tight_layout()
    managed.apply_panel_row_rules(axes)
    return _finish_submission_figure("residual_distribution_top3", managed)


def fig_error_vs_target_top3(df: pd.DataFrame) -> str:
    top3 = df.nsmallest(3, "mae_mean").reset_index(drop=True)
    managed, fig, axes = _new_submission_figure("error_vs_target_top3", FigureProfiles.THREE_PANEL_ROW)
    axes = axes.flatten()
    palette = [PublicationPalette.model(row["model"], i) for i, (_, row) in enumerate(top3.iterrows())]
    for ax, (_, row), color in zip(axes, top3.iterrows(), palette):
        y_true, y_pred = _collect_predictions(row)
        if len(y_true) == 0:
            ax.set_visible(False)
            continue
        abs_err = np.abs(y_pred - y_true)
        ax.scatter(y_true, abs_err, alpha=0.5, s=25, color=color,
                   edgecolors="white", linewidth=0.3)
        order = np.argsort(y_true)
        yt_s, ae_s = y_true[order], abs_err[order]
        window = max(20, int(len(yt_s) * 0.1))
        if window > 1:
            smoothed = np.convolve(ae_s, np.ones(window) / window, mode="same")
            ax.plot(yt_s, smoothed, color="black", lw=2, label="Moving avg")
        ax.axhline(row["mae_mean"], color="red", linestyle="--", lw=1.5, label="Mean MAE")
        ax.set_xlabel("Measured Ra (µm)"); ax.set_ylabel("Absolute Error (µm)")
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    managed.apply_panel_row_rules(axes)
    return _finish_submission_figure("error_vs_target_top3", managed)


def fig_rank_stability_by_family(df: pd.DataFrame) -> str:
    records = []
    for _, row in df.iterrows():
        model = row["model"]
        for fold in row.get("folds", []):
            if isinstance(fold, dict) and "mae" in fold:
                records.append({"model": model, "mae": float(fold["mae"])})
    if records:
        plot_df = pd.DataFrame(records)
        order = plot_df.groupby("model")["mae"].median().sort_values().index.tolist()
        plot_df["model"] = pd.Categorical(plot_df["model"], categories=order, ordered=True)
        managed, fig, ax = _new_submission_figure("rank_stability_by_family", FigureProfiles.DOUBLE)
        palette = {model: PublicationPalette.model(model, i) for i, model in enumerate(order)}
        sns.violinplot(data=plot_df, x="model", y="mae", hue="model", palette=palette,
                       inner="box", linewidth=0.8, ax=ax, legend=False)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Model Family")
        ax.set_ylabel("Per-Fold MAE (µm)")
        ax.set_title(f"Rank Stability Across Folds by Model Family\n({CV_STRATEGY_NAME})")
    else:
        plot_df = df.groupby("model")["mae_mean"].mean().sort_values().reset_index()
        managed, fig, ax = _new_submission_figure("rank_stability_by_family", FigureProfiles.DOUBLE)
        palette = {model: PublicationPalette.model(model, i) for i, model in enumerate(plot_df["model"])}
        sns.barplot(data=plot_df, x="model", y="mae_mean", hue="model", palette=palette, ax=ax, legend=False)
        ax.set_xticklabels(plot_df["model"], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Model Family")
        ax.set_ylabel("Mean MAE (µm)")
    fig.tight_layout()
    return _finish_submission_figure("rank_stability_by_family", managed)


def _modality_group(config: str) -> str:
    parts = set(config.split("+"))
    has_spec = any(t in parts for t in ("ae_spec", "vib_spec", "ae_logspec", "vib_logspec", "ae_mel", "vib_mel"))
    has_wst = any(t in parts for t in ("ae_wst", "vib_wst"))
    has_feat = any(t in parts for t in ("ae_features", "vib_features"))
    has_pp = "pp" in parts
    has_physics = "physics" in parts
    if parts == {"pp"}:
        return "PP only"
    if has_wst and not has_spec:
        return "Wavelet scattering"
    if has_spec and has_wst:
        return "Spectrogram + WST"
    if has_spec and has_feat:
        return "Spectrogram + features"
    if has_spec:
        return "Spectrogram only"
    if has_feat:
        return "Time-domain features"
    if has_physics:
        return "Physics-informed"
    return "Other"


def fig_modality_heatmap(df: pd.DataFrame) -> str:
    df = df.copy()
    df["modality_group"] = df["config"].apply(_modality_group)
    pivot = df.pivot_table(index="model", columns="modality_group", values="mae_mean", aggfunc="mean")
    col_order = pivot.mean(axis=0).sort_values().index.tolist()
    pivot = pivot[col_order]
    managed, fig, ax = _new_submission_figure("modality_heatmap", FigureProfiles.DOUBLE_TALL)
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "Mean MAE (µm)"}, ax=ax)
    ax.set_xlabel("Input Modality Group")
    ax.set_ylabel("Model Family")
    ax.set_title("Mean MAE by Model Family and Input Modality Group")
    fig.tight_layout()
    return _finish_submission_figure("modality_heatmap", managed)


def _metric_str(row, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.5f} ± {row[f'{metric}_std']:.5f}"


def _make_top20_table(df: pd.DataFrame) -> str:
    top20 = df.nsmallest(20, "mae_mean").reset_index(drop=True)
    lines = ["| Rank | Model | Config | MAE (µm) | MSE (µm²) | R² | Folds |",
             "|------|-------|--------|----------|-----------|-----|-------|"]
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        lines.append(
            f"| {i} | {row['model']} | `{row['config']}` | {_metric_str(row, 'mae')} | "
            f"{_metric_str(row, 'mse')} | {_metric_str(row, 'r2')} | {int(row['n_folds_completed'])} |"
        )
    return "\n".join(lines)


def _make_best_by_family(df: pd.DataFrame) -> str:
    best = df.loc[df.groupby("model")["mae_mean"].idxmin()].sort_values("mae_mean")
    lines = ["| Model Family | Best Config | MAE (µm) | MSE (µm²) | R² |",
             "|--------------|-------------|----------|-----------|-----|"]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['model']} | `{row['config']}` | {_metric_str(row, 'mae')} | "
            f"{_metric_str(row, 'mse')} | {_metric_str(row, 'r2')} |"
        )
    return "\n".join(lines)


def _make_signal_rep_table() -> str:
    path = TABLES_DIR / "signal_representation_comparison.csv"
    if not path.exists():
        return "_Signal-representation comparison table not found._"
    sig = pd.read_csv(path)
    lines = ["| Group | Model | Config | MAE (µm) | MSE | R² |",
             "|-------|-------|--------|----------|-----|-----|"]
    for _, row in sig.iterrows():
        lines.append(
            f"| {row['group']} | {row['model']} | `{row['config']}` | {row['MAE (µm)']} | "
            f"{row['MSE']} | {row['R²']} |"
        )
    return "\n".join(lines)


def _make_statistical_table() -> str:
    path = TABLES_DIR / "statistical_tests_updated.csv"
    if not path.exists():
        return "_Statistical tests not found._"
    st = pd.read_csv(path)
    lines = ["| Comparison | MAE A | MAE B | Median Δ | p raw | p Holm | Significant |",
             "|------------|-------|-------|----------|-------|--------|-------------|"]
    for _, r in st.iterrows():
        sig = "Yes" if r["significant_05"] else "No"
        lines.append(
            f"| {r['comparison']} | {r['mean_mae_a']:.5f} | {r['mean_mae_b']:.5f} | "
            f"{r['median_diff']:.5f} | {r['pvalue_raw']:.4g} | {r['pvalue_holm']:.4g} | {sig} |"
        )
    return "\n".join(lines)


def _make_worst_conditions_table() -> str:
    path = TABLES_DIR / "condition_error_ranking.csv"
    if not path.exists():
        return "_Condition error ranking not found._"
    cond = pd.read_csv(path).nlargest(8, "mae").reset_index(drop=True)
    lines = ["| Condition ID | Mean MAE (µm) |",
             "|--------------|---------------|"]
    for _, r in cond.iterrows():
        lines.append(f"| {int(r['condition_id'])} | {r['mae']:.5f} |")
    return "\n".join(lines)


def _xai_list() -> str:
    shap = list(XAI_DIR.glob("shap_summary_*.png"))
    ridge_coef = list(XAI_DIR.glob("ridge_coefficients_*.png"))
    gradcam = list(XAI_DIR.glob("gradcam_*.png"))
    phys = list(XAI_DIR.glob("physics_consistency_*.md"))
    lines = [
        f"- SHAP summary plots: {len(shap)}",
        f"- Ridge coefficient plots: {len(ridge_coef)}",
        f"- Grad-CAM overlays: {len(gradcam)}",
        f"- Physics-consistency reports: {len(phys)}",
    ]
    return "\n".join(lines)


def _uncertainty_list() -> str:
    cal = list(UNC_DIR.glob("mc_dropout_calibration_*.png"))
    rel = list(UNC_DIR.glob("reliability_*.png"))
    return f"- MC-dropout calibration plots: {len(cal)}\n- Reliability / coverage plots: {len(rel)}"


def _make_latency_table() -> str:
    path = TABLES_DIR / "edge_latency_benchmark.csv"
    if not path.exists():
        return "_Latency benchmark pending._"
    bench = pd.read_csv(path)
    lines = [
        "| Model | Config | MAE (µm) | Params | Checkpoint (MB) | Median latency (ms) | p95 latency (ms) |",
        "|-------|--------|----------|--------|-----------------|---------------------|------------------|"
    ]
    for _, r in bench.iterrows():
        params = int(r["n_parameters"]) if not pd.isna(r["n_parameters"]) else 0
        lines.append(
            f"| {r['model']} | `{r['config']}` | {r['mae_mean']:.5f} | {params:,} | "
            f"{r['checkpoint_size_mb']:.2f} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} |"
        )
    return "\n".join(lines)


def _make_latency_takeaways() -> str:
    path = TABLES_DIR / "edge_latency_benchmark.csv"
    if not path.exists():
        return "- Fold-complete latency benchmark pending."
    bench = pd.read_csv(path).set_index(["model", "config"])
    rf = bench.loc[("RandomForestModel", "ae_logspec+vib_logspec")]
    cnn = bench.loc[("ResNetVibCNN", "vib_spec")]
    lgbm = bench.loc[("LightGBMModel", "vib_logspec")]
    return "\n".join([
        f"- **ResNetVibCNN / Vib-dB** has the lowest pooled median latency "
        f"({cnn['median_ms']:.2f} ms; p95 {cnn['p95_ms']:.2f} ms) and a median "
        f"checkpoint size of {cnn['checkpoint_size_mb']:.2f} MB, but its MAE is higher than the RF alternatives.",
        f"- **RandomForestModel / AE-dB-z + Vib-dB-z** has the lowest historical benchmark MAE "
        f"with pooled median latency {rf['median_ms']:.2f} ms, p95 {rf['p95_ms']:.2f} ms, "
        f"and median checkpoint size {rf['checkpoint_size_mb']:.2f} MB.",
        f"- **LightGBMModel / Vib-dB-z** has pooled median latency {lgbm['median_ms']:.2f} ms "
        f"and p95 {lgbm['p95_ms']:.2f} ms, with TreeSHAP-compatible model structure.",
    ])


def generate_report(df: pd.DataFrame, git_info: Dict[str, str]) -> Path:
    _set_style()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    best = df.loc[df["mae_mean"].idxmin()]
    n_models = df["model"].nunique()
    n_configs = df["config"].nunique()
    n_experiments = len(df)
    best_dl = df[df["model"].isin({"ResNetVibCNN", "ResNetAECNN", "MultiscaleSpectrogramCNN",
                                     "ResNetFusion", "BilinearFusionNetwork", "TrajectoryCNN",
                                     "GraphNeuralNetworkFusion", "TransferFeatureMLP"})].nsmallest(1, "mae_mean").iloc[0]
    gap = best_dl["mae_mean"] - best["mae_mean"]
    pct_gap = 100 * gap / best["mae_mean"]

    fig_ranking_top20(df)
    fig_prediction_scatter_top3(df)
    fig_residual_distribution_top3(df)
    fig_error_vs_target_top3(df)
    fig_rank_stability_by_family(df)
    fig_modality_heatmap(df)

    report_path = PUBLICATION_DIR / "submission_report.md"

    report = f"""# Physics-Aware Transparent Models for Grinding Surface Roughness Prediction

## Updated LOGO Benchmark Report

**Generated:** {now}

**Project:** VibeGrinding

**Git commit:** `{git_info['commit']}` (`{git_info['branch']}`, dirty={git_info['dirty']})
**Report:** `reports/publication/submission_report.md`

---

## Abstract

This report summarises the updated VibeGrinding benchmark designed for submission to *Advanced Engineering Informatics (AEI)*, *Expert Systems with Applications (ESWA)*, or *IEEE Transactions on Industrial Informatics (TII)*. We evaluate **{n_models} model families** across **{n_configs} input configurations** ({n_experiments} unique experiments) using a strict **leave-one-condition-out (LOGO)** protocol with {CV_TOTAL_FOLDS} folds. The best-performing system, **`{best['model']}`** with **`{best['config']}`**, achieves a mean absolute error (MAE) of **{best['mae_mean']:.5f} ± {best['mae_std']:.5f} µm**. A transparent random-forest spectrogram model outperforms the best deep-learning baseline (`{best_dl['model']} / {best_dl['config']}`) by **{gap:.5f} µm ({pct_gap:.1f}%)**. We complement the accuracy benchmark with signal-processing representations, model-agnostic explanations, physics-consistency checks, uncertainty quantification, and condition-level failure analysis.

---

## 1. Research Questions

1. **RQ1 — Accuracy:** What is the best achievable regression error under strict LOGO cross-validation?
2. **RQ2 — Transparency:** Can simple, interpretable models (Ridge, shallow MLP, random forest) match or exceed deep-learning models?
3. **RQ3 — Signal representations:** Do log-spectrograms, mel spectrograms, raw spectrograms, or wavelet-scattering transforms give the best performance?
4. **RQ4 — Explainability:** Do model explanations align with known grinding physics (AE high-frequency content, vibration fundamental frequencies)?
5. **RQ5 — Reliability:** How well calibrated is the prediction uncertainty from MC-dropout?
6. **RQ6 — Deployment:** What is the latency–memory trade-off of the top models?

---

## 2. Validation Design

- **Strategy:** Leave-one-condition-out (LOGO) cross-validation.
- **Folds:** {CV_TOTAL_FOLDS} outer folds, one repeat.
- **Metric:** Mean absolute error (MAE) in µm, with bootstrap 95% CI and Wilcoxon signed-rank tests.
- **Canonical results:** `reports/evidence/tables/full_results_logo_only.csv` (excludes earlier mixed-fold contamination).

---

## 3. Overall Results

### 3.1 Top 20 model–configuration pairs

{_make_top20_table(df)}

![Top 20 ranking]({FIGURES.get('ranking_top20', '')})

### 3.2 Best configuration per model family

{_make_best_by_family(df)}

### 3.3 Prediction calibration (top 3)

![Prediction scatter]({FIGURES.get('prediction_scatter_top3', '')})

### 3.4 Residual analysis (top 3)

![Residual distribution]({FIGURES.get('residual_distribution_top3', '')})

### 3.5 Error versus target roughness (top 3)

![Error vs target]({FIGURES.get('error_vs_target_top3', '')})

### 3.6 Rank stability across folds

![Rank stability]({FIGURES.get('rank_stability_by_family', '')})

### 3.7 Modality heatmap

![Modality heatmap]({FIGURES.get('modality_heatmap', '')})

---

## 4. Signal-Representation Comparison

{_make_signal_rep_table()}

The leading observed RF representation group comprises fused AE-dB-z + Vib-dB-z and AE-log-mel + Vib-log-mel descriptors. Their canonical point estimates are separated by 0.0001 micrometres, and the documented current dB-z pipeline does not exactly reproduce the canonical artifact; neither representation should be treated as uniquely superior. The wavelet-scattering transform (`vib_wst`) is competitive with the best deep-learning spectrogram model, despite using a much simpler learner (LightGBM).

---

## 5. Statistical Significance

Pairwise Wilcoxon signed-rank tests on per-fold MAE, Holm–Bonferroni corrected:

{_make_statistical_table()}

**Interpretation:** No corrected comparison reaches statistical significance at α = 0.05, mainly because one outlier condition (condition 7) inflates fold-level variance. However, the transparent RF spectrogram model consistently yields lower mean and median MAE than the DL baselines, with moderate effect sizes. The only significant result is the expected superiority of RF over the weak shallow-MLP feature baseline.

---

## 6. Interpretability and Physics Consistency

{_xai_list()}

- **Ridge coefficients** on `ae_spec+vib_spec` give a linear attribution per frequency bin.
- **TreeSHAP** for LightGBM on the same input highlights frequency regions that physically correspond to grinding vibrations and AE burst energy.
- **Grad-CAM** for `ResNetVibCNN` on `vib_spec` localises discriminative spectrogram regions after interpolation to the input display grid.
- **Physics-consistency checks** map important bins to expected AE (~6.69 kHz/bin) and vibration (100 Hz/bin) frequencies; reports are in `{REL_TABLES.replace('tables','xai')}`.

---

## 7. Uncertainty Quantification

{_uncertainty_list()}

MC-dropout uncertainty was evaluated for `ResNetVibCNN` on `vib_spec` fold 0. Empirical coverage and reliability diagrams are saved under `reports/evidence/uncertainty/`. Calibration can be further improved with temperature scaling or ensemble methods if reviewers request it.

---

## 8. Condition-Level Failure Analysis

The hardest grinding conditions across all model–configuration pairs:

{_make_worst_conditions_table()}

Condition 7 is a clear outlier; its exclusion dramatically reduces average MAE for every model. This condition should be investigated for anomalous wheel wear, dressing state, or measurement error.

---

## 9. Deployment Considerations

Single-sample CPU inference latency and checkpoint footprint for representative top configurations. Timing is pooled over the 16 canonical repeat-0 LOGO checkpoints, using five blocks of 200 calls per checkpoint with warm-up excluded:

{_make_latency_table()}

![Latency vs accuracy]({REL_PLOTS}/latency_vs_accuracy.png)

Key observations:

{_make_latency_takeaways()}
- **BilinearFusionNetwork** is the largest and slowest of the set, reflecting the cost of deep multimodal fusion.

---

## 10. Conclusions and Submission Angle

1. **Accuracy:** A transparent random forest on log-spectrograms achieves the lowest LOGO MAE ({best['mae_mean']:.5f} µm).
2. **Transparency:** Ridge regression and LightGBM on scattering features are competitive with deep learning, while remaining fully explainable.
3. **Signal processing:** Log-spectrogram and mel spectrograms outperform raw spectrograms; WST provides a strong physics-aware alternative.
4. **Interpretability:** SHAP and Ridge attributions align with expected AE/vibration frequency content.
5. **Uncertainty:** MC-dropout gives plausible intervals, though calibration can be tightened.
6. **Failure cases:** One outlier condition dominates the error budget; a robustness analysis should be included in the paper.

**Recommended framing for AEI/ESWA/TII:** *"Physics-aware signal representations enable transparent, interpretable, and deployable grinding-roughness prediction that matches deep-learning accuracy while including higher-capacity deep architectures as comparators."*

---

## Reproducibility

- Canonical results: `{REL_TABLES}/full_results_logo_only.csv`
- Statistical tests: `{REL_TABLES}/statistical_tests_updated.csv`
- Signal-representation comparison: `{REL_TABLES}/signal_representation_comparison.csv`
- Condition errors: `{REL_TABLES}/condition_error_ranking.csv`
- XAI outputs: `../evidence/xai/`
- Uncertainty outputs: `../evidence/uncertainty/`
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return report_path


def main() -> int:
    df = load_full_results()
    git_info = get_git_info()
    generate_report(df, git_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
