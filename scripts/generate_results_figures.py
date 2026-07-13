# Output mapping: see docs/figure_script_toc.md
"""Generate real-data visual illustrations for Results and Discussion sections.

Figures produced:
- results_statistical_comparison_pairs.png   (4.2 Statistical comparison)
- results_statistical_comparison_forest.png  (4.2 Statistical comparison)
- results_interpretability_bandmass.png      (4.4 Interpretability and physics consistency)
- results_uncertainty_calibration.png        (4.5 Uncertainty quantification)
- results_condition_error_bars.png           (4.6 Condition-level failure analysis)
- results_deployment_tradeoffs.png           (4.7 Deployment characteristics)
- discussion_complexity_accuracy.png         (5.1 Why transparent models perform strongly)

All figures are saved to reports/evidence/plots/results and copied to overleaf/images.
"""
import ast
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import (
    FigureProfiles,
    MutableFigure,
    PublicationPalette,
    PublicationPlotter,
)

OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

CONFIG_LABELS = {
    "ae_logspec+vib_logspec": "AE-dB-z + Vib-dB-z",
    "ae_mel+vib_mel": "AE-log-mel + Vib-log-mel",
    "ae_spec+vib_spec": "AE-dB + Vib-dB",
    "ae_spec+vib_spec+physics+pp": "AE-dB + Vib-dB + phys + PP",
    "vib_logspec": "Vib-dB-z",
    "vib_spec": "Vib-dB",
    "ae_spec": "AE-dB",
}


def load_full_results(path: Path):
    df = pd.read_csv(path)
    df["maes"] = df["folds"].apply(lambda s: [d["mae"] for d in ast.literal_eval(s)])
    return df


def make_figure(name: str, profile, *, metadata: dict | None = None):
    """Create a registered OOP figure; drawing remains local to each plot."""
    figure = MutableFigure(
        name,
        profile=profile,
        out_dir=OUT_DIR,
        metadata={"generator": "scripts/generate_results_figures.py", **(metadata or {})},
    )
    fig, axes = figure.create()
    return figure, fig, axes


def get_fold_maes(model: str, config: str):
    """Return per-fold MAEs for a model/config, preferring full_results, else predictions."""
    full_path = ROOT / "reports" / "evidence" / "tables" / "full_results_logo_only.csv"
    if full_path.exists():
        full = load_full_results(full_path)
        row = full[(full["model"] == model) & (full["config"] == config)]
        if len(row) == 1:
            return np.array(row.iloc[0]["maes"])

    pred_dir = ROOT / "reports" / "evidence" / "predictions"
    config_underscored = config.replace("+", "_")
    maes = []
    for fold in range(16):
        csv_path = pred_dir / f"{model}_{config_underscored}_fold{fold}_repeat0.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing predictions for {model}/{config} fold {fold}")
        df = pd.read_csv(csv_path)
        maes.append(np.abs(df["y_true"] - df["y_pred"]).mean())
    return np.array(maes)


def bootstrap_median_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05):
    rng = np.random.default_rng(42)
    boot_medians = []
    n = len(diff)
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_medians.append(np.median(sample))
    boot_medians = np.sort(boot_medians)
    lo = boot_medians[int(alpha / 2 * n_boot)]
    hi = boot_medians[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def wrap_comparison_label(label: str) -> str:
    """Render pairwise labels consistently as [A] / vs / [B]."""
    model_a, model_b = label.split(" vs ", maxsplit=1)
    return PublicationPlotter.comparison_label(model_a, model_b)


def plot_statistical_comparison_forest():
    """Forest plot of paired median MAE differences with bootstrap 95% CIs."""
    tests = pd.read_csv(ROOT / "reports" / "evidence" / "tables" / "statistical_tests_updated.csv")

    comparisons = [
        ("Best RF dB-z vs best dB-CNN", "RandomForestModel", "ae_logspec+vib_logspec", "ResNetVibCNN", "vib_spec"),
        ("Best RF vs best WST", "RandomForestModel", "ae_logspec+vib_logspec", "LightGBMModel", "vib_wst"),
        ("RF dB vs Ridge dB", "RandomForestModel", "ae_spec+vib_spec", "RidgeRegressionModel", "ae_spec+vib_spec"),
        ("RF dB vs shallow MLP features", "RandomForestModel", "ae_spec+vib_spec", "ShallowMLPModel", "ae_features+vib_features+pp"),
        ("RF Vib-dB vs ResNetVibCNN Vib-dB", "RandomForestModel", "vib_spec", "ResNetVibCNN", "vib_spec"),
        ("RF dB-z vs RF dB", "RandomForestModel", "ae_logspec+vib_logspec", "RandomForestModel", "ae_spec+vib_spec"),
        ("RF AE-dB vs ResNetAECNN AE-dB", "RandomForestModel", "ae_spec", "ResNetAECNN", "ae_spec"),
        ("ResNetVibCNN vs ResNetAECNN", "ResNetVibCNN", "vib_spec", "ResNetAECNN", "ae_spec"),
        ("ResNetAECNN vs ChannelAttentionCNN", "ResNetAECNN", "ae_spec", "ChannelAttentionCNN", "ae_spec"),
    ]

    rows = []
    for label, m_a, c_a, m_b, c_b in comparisons:
        maes_a = get_fold_maes(m_a, c_a)
        maes_b = get_fold_maes(m_b, c_b)
        diff = maes_a - maes_b
        median_diff = np.median(diff)
        ci_lo, ci_hi = bootstrap_median_ci(diff)
        match = tests[(tests["model_a"] == m_a) & (tests["config_a"] == c_a) &
                      (tests["model_b"] == m_b) & (tests["config_b"] == c_b)]
        p_holm = match.iloc[-1]["pvalue_holm"] if len(match) else 1.0
        rows.append({
            "label": label,
            "median_diff": median_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_holm": p_holm,
            "significant": p_holm < 0.05,
        })

    df = pd.DataFrame(rows)

    managed, fig, ax = make_figure("results_statistical_comparison_forest.png", FigureProfiles.DOUBLE_TALL)
    y = np.arange(len(df))
    ci_color = PublicationPalette.OBSERVED
    point_color = PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"]

    for i, row in df.iterrows():
        alpha = 1.0 if row["significant"] else 0.78
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=ci_color, lw=2.5,
                solid_capstyle="round", alpha=alpha)
        ax.plot(row["median_diff"], i, "o", color=point_color, markersize=8,
                markeredgecolor="k", markeredgewidth=0.6, alpha=alpha)

    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_comparison_label(label) for label in df["label"]], fontsize=7)
    ax.set_xlabel("Median paired difference in LOGO MAE (Model A − Model B) (µm)")
    ax.set_title("Historical canonical comparisons: median difference and 95% bootstrap CI")
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.invert_yaxis()

    # Reserve an in-axis annotation column so the plot remains full-width.
    x_min, x_max = ax.get_xlim()
    data_span = x_max - x_min
    ax.set_xlim(x_min, x_max + 0.42 * data_span)
    annotation_x = x_max + 0.03 * data_span
    for i, row in df.iterrows():
        text = f"{row['median_diff']:.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
        if row["significant"]:
            text += " *"
        ax.text(annotation_x, i, text, va="center", ha="left", fontsize=7,
                color=PublicationPalette.NEUTRAL, clip_on=True)

    managed.save()


def plot_statistical_comparison_pairs():
    """Grouped bar chart of mean MAE for the six pairwise comparisons in Table tbl:statistical."""
    full = load_full_results(ROOT / "reports" / "evidence" / "tables" / "full_results_logo_only.csv")
    wst = pd.read_csv(ROOT / "reports" / "evidence" / "tables" / "wst_results.csv")
    tests = pd.read_csv(ROOT / "reports" / "evidence" / "tables" / "statistical_tests_updated.csv")

    def lookup(model, config):
        # full_results uses '+' separators; wst_results uses underscores
        row = full[(full["model"] == model) & (full["config"] == config)]
        if len(row) == 1:
            return row.iloc[0]
        row = wst[(wst["model"] == model) & (wst["config"] == config)]
        if len(row) == 1:
            r = row.iloc[0]
            return {"mae_mean": r["mae_mean"], "mae_std": r["mae_std"]}
        raise ValueError(f"Not found: {model} / {config}")

    comparisons = [
        ("Best RF dB-z vs best dB-CNN", "RandomForestModel", "ae_logspec+vib_logspec", "ResNetVibCNN", "vib_spec"),
        ("Best RF vs best WST", "RandomForestModel", "ae_logspec+vib_logspec", "LightGBMModel", "vib_wst"),
        ("RF dB vs Ridge dB", "RandomForestModel", "ae_spec+vib_spec", "RidgeRegressionModel", "ae_spec+vib_spec"),
        ("RF dB vs shallow MLP features", "RandomForestModel", "ae_spec+vib_spec", "ShallowMLPModel", "ae_features+vib_features+pp"),
        ("RF Vib-dB vs ResNetVibCNN Vib-dB", "RandomForestModel", "vib_spec", "ResNetVibCNN", "vib_spec"),
        ("RF dB-z vs RF dB", "RandomForestModel", "ae_logspec+vib_logspec", "RandomForestModel", "ae_spec+vib_spec"),
        ("RF AE-dB vs ResNetAECNN AE-dB", "RandomForestModel", "ae_spec", "ResNetAECNN", "ae_spec"),
        ("ResNetVibCNN vs ResNetAECNN", "ResNetVibCNN", "vib_spec", "ResNetAECNN", "ae_spec"),
        ("ResNetAECNN vs ChannelAttentionCNN", "ResNetAECNN", "ae_spec", "ChannelAttentionCNN", "ae_spec"),
    ]

    rows = []
    for label, m_a, c_a, m_b, c_b in comparisons:
        a = lookup(m_a, c_a)
        b = lookup(m_b, c_b)
        # significance from the updated test table by matching configs
        match = tests[(tests["model_a"] == m_a) & (tests["config_a"] == c_a) &
                      (tests["model_b"] == m_b) & (tests["config_b"] == c_b)]
        sig = ""
        if len(match):
            p_holm = match.iloc[-1]["pvalue_holm"]
            if p_holm < 0.001:
                sig = "***"
            elif p_holm < 0.01:
                sig = "**"
            elif p_holm < 0.05:
                sig = "*"
        rows.append({"label": label, "side": "A", "mean": a["mae_mean"], "std": a["mae_std"], "sig": sig})
        rows.append({"label": label, "side": "B", "mean": b["mae_mean"], "std": b["mae_std"], "sig": sig})

    df = pd.DataFrame(rows)
    comparison_ids = [
        wrap_comparison_label(row["label"])
        for row in rows
        if row["side"] == "A"
    ]
    means_a = [r["mean"] for r in rows if r["side"] == "A"]
    stds_a = [r["std"] for r in rows if r["side"] == "A"]
    means_b = [r["mean"] for r in rows if r["side"] == "B"]
    stds_b = [r["std"] for r in rows if r["side"] == "B"]
    sigs = [r["sig"] for r in rows if r["side"] == "A"]

    y = np.arange(len(comparison_ids))
    width = 0.35
    managed, fig, ax = make_figure("results_statistical_comparison_pairs.png", FigureProfiles.DOUBLE_TALL)
    ax.barh(y - width/2, means_a, width, xerr=stds_a, label="Model A", color=PublicationPalette.OBSERVED, capsize=3)
    ax.barh(y + width/2, means_b, width, xerr=stds_b, label="Model B", color=PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"], capsize=3)

    ax.set_xlabel("Mean LOGO MAE (µm)")
    ax.set_title("Historical canonical paired LOGO MAE")
    ax.set_yticks(y)
    ax.set_yticklabels(comparison_ids, fontsize=7)
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=PublicationPlotter.LEGEND_SIZE)
    top_vals = [max(a + sa, b + sb) for a, sa, b, sb in zip(means_a, stds_a, means_b, stds_b)]
    ax.set_xlim(0, max(top_vals) * 1.2)

    # annotate significance above the taller bar
    for i, (a_mean, b_mean, s) in enumerate(zip(means_a, means_b, sigs)):
        if not s:
            continue
        top = top_vals[i]
        ax.annotate(s, xy=(top, y[i]), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7, color="black", fontweight="bold")

    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.14, top=0.90)
    managed.save()


def plot_interpretability_bandmass():
    """Stacked bar chart of TreeSHAP/Grad-CAM importance mass by frequency band."""
    data = {
        "Vibration Grad-CAM": [0.09, 2.26, 44.55, 53.10],
        "LightGBM global TreeSHAP": [3.55, 4.97, 13.33, 78.15],
        "RF OOF TreeSHAP": [0.21, 0.26, 10.89, 88.63],
    }
    bands = ["<500 Hz", "500 Hz–2 kHz", "2–15 kHz", ">15 kHz"]
    df = pd.DataFrame(data, index=bands).T

    managed, fig, ax = make_figure("results_interpretability_bandmass.png", FigureProfiles.WIDE)
    colors = [PublicationPalette.CONDITION_7, PublicationPalette.VALIDATION, PublicationPalette.MODEL_FAMILY["LightGBMModel"], PublicationPalette.OBSERVED]
    df.plot(kind="barh", stacked=True, ax=ax, color=colors, width=0.7)
    ax.set_xlabel("Importance mass (%)")
    ax.set_title("Vibration importance mass by physical frequency band")
    ax.legend(title="Frequency band", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    managed.save()


def plot_uncertainty_calibration():
    """Scatter of MC-dropout predicted standard deviation vs absolute error."""
    full_path = ROOT / "reports" / "evidence" / "uncertainty" / "mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv"
    subset_path = ROOT / "reports" / "evidence" / "uncertainty" / "mc_dropout_ResNetVibCNN_vib_spec.csv"
    csv_path = full_path if full_path.exists() else subset_path
    df = pd.read_csv(csv_path)
    if "y_pred" in df.columns and "y_mean" not in df.columns:
        df["y_mean"] = df["y_pred"]
    if "covered" not in df.columns:
        df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["abs_error"] = (df["y_true"] - df["y_mean"]).abs()

    managed, fig, ax = make_figure("results_uncertainty_calibration.png", FigureProfiles.SINGLE)
    colors = np.where(df["covered"], PublicationPalette.MODEL_FAMILY["LightGBMModel"], PublicationPalette.CONDITION_7)
    ax.scatter(df["y_std"], df["abs_error"], c=colors, alpha=0.7, s=50, edgecolors="k", linewidths=0.3)
    max_val = max(df["y_std"].max(), df["abs_error"].max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="error = std")
    ax.plot([0, max_val], [0, 1.96 * max_val], "k:", lw=1, label="error = 1.96 std")
    ax.set_xlabel("MC-dropout standard deviation (µm)")
    ax.set_ylabel("Absolute error (µm)")
    ax.set_title("Uncertainty calibration: predicted std vs observed error")
    corr = df["y_std"].corr(df["abs_error"])
    ax.text(0.05, 0.95, f"Pearson $r$ = {corr:.2f}\nCoverage = {df['covered'].mean()*100:.1f}%",
            transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    coverage_handles = [
        Line2D([0], [0], marker="o", color="w", label="covered", markerfacecolor=PublicationPalette.MODEL_FAMILY["LightGBMModel"], markeredgecolor="k", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="uncovered", markerfacecolor=PublicationPalette.CONDITION_7, markeredgecolor="k", markersize=7),
    ]
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *coverage_handles], loc="lower right")
    fig.tight_layout()
    managed.save()


def plot_condition_error_bars():
    """Bar chart of mean MAE per grinding condition, sorted, with Condition 7 highlighted."""
    df = pd.read_csv(ROOT / "reports" / "evidence" / "tables" / "condition_error_ranking.csv")
    df = df.sort_values("mae", ascending=True)

    managed, fig, ax = make_figure("results_condition_error_bars.png", FigureProfiles.WIDE)
    colors = [PublicationPalette.CONDITION_7 if cid == 7 else PublicationPalette.OBSERVED for cid in df["condition_id"]]
    bars = ax.barh(df["condition_id"].astype(str), df["mae"], color=colors)
    ax.set_xlabel("Mean MAE across all model-configuration pairs (µm)")
    ax.set_ylabel("Condition ID")
    ax.set_title("Condition-level failure analysis: mean prediction error")
    ax.axvline(df["mae"].mean(), color="black", linestyle="--", lw=1, label=f"overall mean = {df['mae'].mean():.3f}")
    ax.legend(loc="lower right")
    # annotate condition 7
    for bar, cid in zip(bars, df["condition_id"]):
        if cid == 7:
            ax.text(
                bar.get_width() * 0.97,
                bar.get_y() + bar.get_height() / 2,
                "Condition 7: dominant outlier",
                ha="right",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
    fig.tight_layout()
    managed.save()


def plot_deployment_tradeoffs():
    """Latency-accuracy-size trade-off for representative models."""
    df = pd.read_csv(ROOT / "reports" / "evidence" / "tables" / "edge_latency_benchmark.csv")
    df = df.sort_values(["median_ms", "mae_mean"]).reset_index(drop=True)
    df["plot_id"] = np.arange(1, len(df) + 1)
    managed, fig, ax = make_figure("results_deployment_tradeoffs.png", FigureProfiles.DOUBLE)

    # bubble size proportional to checkpoint size
    sizes = (df["checkpoint_size_mb"] / df["checkpoint_size_mb"].max()) * 800 + 80
    colors = [PublicationPalette.model(model, i) for i, model in enumerate(df["model"])]
    ax.scatter(df["median_ms"], df["mae_mean"], s=sizes,
               c=colors, alpha=0.8, edgecolors="k", linewidths=0.5)

    for _, row in df.iterrows():
        ax.text(row["median_ms"], row["mae_mean"], str(int(row["plot_id"])),
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    key_labels = []
    for _, row in df.iterrows():
        label = row["model"].replace("Model", "").replace("Network", "")
        key_labels.append(
            f"{int(row['plot_id'])}: {label} / "
            f"{CONFIG_LABELS.get(row['config'], row['config'])}"
        )

    ax.set_xlabel("Median CPU latency (ms)")
    ax.set_ylabel("Mean LOGO MAE (µm)")
    ax.set_title("Deployment trade-offs: accuracy, latency, and checkpoint size")
    x_max = float(df["median_ms"].max())
    ax.set_xlim(left=0, right=x_max * 1.15)
    ax.invert_yaxis()
    midpoint = (len(key_labels) + 1) // 2
    fig.text(0.15, 0.025, "\n".join(key_labels[:midpoint]), ha="left", va="bottom", fontsize=PublicationPlotter.LEGEND_SIZE)
    fig.text(0.55, 0.025, "\n".join(key_labels[midpoint:]), ha="left", va="bottom", fontsize=PublicationPlotter.LEGEND_SIZE)
    fig.subplots_adjust(right=0.98, bottom=0.29)
    managed.save()


def plot_discussion_complexity_accuracy():
    """Model stored size versus LOGO accuracy."""
    df = pd.read_csv(ROOT / "reports" / "evidence" / "tables" / "edge_latency_benchmark.csv")
    managed, fig, ax = make_figure("discussion_complexity_accuracy.png", FigureProfiles.WIDE)

    x = df["checkpoint_size_mb"]
    y = df["mae_mean"]

    family_map = {
        "RandomForestModel": "Tree ensemble",
        "LightGBMModel": "Tree ensemble",
        "ResNetVibCNN": "Deep CNN",
        "ResNetAECNN": "Deep CNN",
        "BilinearFusionNetwork": "Fusion network",
    }
    df = df.copy()
    df["family"] = df["model"].map(family_map).fillna("Other")
    palette = {
        "Tree ensemble": PublicationPalette.OBSERVED,
        "Deep CNN": PublicationPalette.MODEL_FAMILY["ResNetVibCNN"],
        "Fusion network": PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"],
    }
    colors = [palette.get(f, PublicationPalette.NEUTRAL) for f in df["family"]]

    df = df.reset_index(drop=True)
    df["plot_id"] = np.arange(1, len(df) + 1)
    ax.scatter(x, y, s=120, c=colors, alpha=0.85, edgecolors="k", linewidths=0.5, zorder=3)
    for _, row in df.iterrows():
        ax.text(
            row["checkpoint_size_mb"],
            row["mae_mean"],
            str(int(row["plot_id"])),
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Checkpoint size (MB, log scale)")
    ax.set_ylabel("Mean LOGO MAE (µm)")
    ax.set_title("Stored model size versus LOGO accuracy")

    ax.invert_yaxis()
    family_labels = {
        "Tree ensemble": "tree",
        "Deep CNN": "CNN",
        "Fusion network": "fusion",
    }
    key_labels = [
        f"{int(row['plot_id'])}: {row['model'].replace('Model', '').replace('Network', '')} "
        f"[{family_labels.get(row['family'], row['family'])}]"
        for _, row in df.iterrows()
    ]
    midpoint = (len(key_labels) + 1) // 2
    fig.text(0.14, 0.02, "\n".join(key_labels[:midpoint]), ha="left", va="bottom", fontsize=6)
    fig.text(0.56, 0.02, "\n".join(key_labels[midpoint:]), ha="left", va="bottom", fontsize=6)
    fig.subplots_adjust(bottom=0.20)
    managed.save()


def main():
    plot_statistical_comparison_pairs()
    plot_statistical_comparison_forest()
    plot_interpretability_bandmass()
    plot_uncertainty_calibration()
    plot_condition_error_bars()
    plot_deployment_tradeoffs()
    plot_discussion_complexity_accuracy()


if __name__ == "__main__":
    main()
