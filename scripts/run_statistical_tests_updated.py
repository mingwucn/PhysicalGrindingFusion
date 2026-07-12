#!/usr/bin/env python3
"""
Updated pairwise statistical comparison for the transparent-ML + signal-processing
submission pipeline.

Uses Wilcoxon signed-rank tests on per-fold MAEs (LOGO 16-fold) with
Holm-Bonferroni correction and bootstrap CIs.

Outputs:
    reports/evidence/tables/statistical_tests_updated.csv
    reports/evidence/tables/statistical_tests_updated.md
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "reports" / "evidence" / "metrics"
TABLES_DIR = ROOT / "reports" / "evidence" / "tables"


def config_stem(config: str) -> str:
    """Turn 'ae_spec+vib_spec' into 'ae_spec_vib_spec'."""
    return config.replace("+", "_")


def load_fold_maes_from_full_results(model_name: str, config: str) -> List[float] | None:
    """Load per-fold MAEs from the LOGO-only summary table if available."""
    path = TABLES_DIR / "full_results_logo_only.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[(df["model"] == model_name) & (df["config"] == config)]
    if row.empty:
        return None
    folds = ast.literal_eval(row.iloc[0]["folds"])
    return [float(f["mae"]) for f in sorted(folds, key=lambda x: x["fold"])]


def load_fold_maes(path: Path) -> List[float]:
    with open(path) as f:
        data = json.load(f)
    if "fold_records" in data:
        return [float(r["mae"]) for r in data["fold_records"]]
    if "folds" in data:
        return [float(r["mae"]) for r in data["folds"]]
    source = data.get("source_artifacts", {})
    preds = source.get("predictions", [])
    if preds:
        maes: List[float] = []
        for csv_path in preds:
            df = pd.read_csv(csv_path)
            mae = float(np.mean(np.abs(df["y_pred"].to_numpy() - df["y_true"].to_numpy())))
            maes.append(mae)
        return maes
    raise ValueError(f"No fold data in {path}")


def find_result(model_name: str, config: str) -> Tuple[Path, List[float]]:
    # Prefer the LOGO-only summary table so the reported means match the
    # manuscript exactly; fall back to the JSON metrics files only when needed.
    maes = load_fold_maes_from_full_results(model_name, config)
    if maes is not None:
        return Path("full_results_logo_only.csv"), maes

    stem = config_stem(config)
    pattern = f"cv_results_{model_name}_{stem}.json"
    candidates = list(METRICS_DIR.glob(pattern))
    if not candidates:
        # tolerate repeats
        candidates = list(METRICS_DIR.glob(f"cv_results_{model_name}_{stem}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No metrics file for {model_name} / {config} ({pattern})")
    # prefer exact match if multiple
    candidates.sort(key=lambda p: len(p.name))
    path = candidates[0]
    return path, load_fold_maes(path)


def hodges_lehmann(diffs: np.ndarray) -> float:
    """One-sample Hodges-Lehmann estimator for paired differences."""
    walsh = []
    for i in range(len(diffs)):
        for j in range(i, len(diffs)):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    return float(np.median(walsh))


def bootstrap_hl_ci(diffs: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot.append(hodges_lehmann(sample))
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def rank_biserial(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    total = float(np.sum(ranks))
    return float(np.sum(np.sign(nonzero) * ranks) / total)


def wilcoxon_report(a: List[float], b: List[float]) -> Dict[str, Any]:
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0):
        return {
            "statistic": np.nan,
            "pvalue_raw": 1.0,
            "median_diff": 0.0,
            "hl_diff": 0.0,
            "hl_ci_lower": 0.0,
            "hl_ci_upper": 0.0,
            "rank_biserial": 0.0,
        }
    # use exact=False to avoid warnings with small ties
    stat, p = wilcoxon(diff, zero_method="zsplit")
    hl_ci = bootstrap_hl_ci(diff)
    return {
        "statistic": float(stat),
        "pvalue_raw": float(p),
        "median_diff": float(np.median(diff)),
        "hl_diff": hodges_lehmann(diff),
        "hl_ci_lower": hl_ci[0],
        "hl_ci_upper": hl_ci[1],
        "rank_biserial": rank_biserial(diff),
    }


def bootstrap_ci(values: List[float], n_boot: int = 5000, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.asarray(values)
    boot_means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def effect_size(a: List[float], b: List[float]) -> float:
    """Median-difference effect size normalised by the pooled MAD."""
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    mad = np.median(np.abs(diff - np.median(diff)))
    if mad == 0:
        return 0.0
    return float(np.median(diff) / mad)


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # (label_a, model_a, config_a, label_b, model_b, config_b)
    comparisons: List[Tuple[str, str, str, str, str, str]] = [
        # 1. central claim: best transparent ML matches/exceeds best DL
        ("Best RF (auditable ML)", "RandomForestModel", "ae_logspec+vib_logspec",
         "Best DL", "ResNetVibCNN", "vib_spec"),
        # 2. transparent ML vs physics-aware signal representation
        ("Best RF (spectrogram)", "RandomForestModel", "ae_logspec+vib_logspec",
         "Best WST", "LightGBMModel", "vib_wst"),
        # 3. tree vs linear on the same spectrogram input
        ("RF dB", "RandomForestModel", "ae_spec+vib_spec",
         "Ridge dB", "RidgeRegressionModel", "ae_spec+vib_spec"),
        # 4. tree vs linear on WST input
        ("RF WST", "RandomForestModel", "vib_wst",
         "Ridge WST", "RidgeRegressionModel", "vib_wst"),
        # 5. shallow MLP sanity baseline vs transparent tree
        ("RF dB", "RandomForestModel", "ae_spec+vib_spec",
         "Shallow MLP features", "ShallowMLPModel", "ae_features+vib_features+pp"),
        # 6. same input: classical tree vs deep CNN
        ("RF Vib-dB", "RandomForestModel", "vib_spec",
         "ResNetVibCNN Vib-dB", "ResNetVibCNN", "vib_spec"),
        # 7. modality ablation under the best transparent model
        ("RF AE/Vib dB-z", "RandomForestModel", "ae_logspec+vib_logspec",
         "RF Vib-dB-z", "RandomForestModel", "vib_logspec"),
        # 8. representation comparison under the same transparent model
        ("RF dB-z", "RandomForestModel", "ae_logspec+vib_logspec",
         "RF dB", "RandomForestModel", "ae_spec+vib_spec"),
        # 9. best tree vs best gradient boosting
        ("Best RF", "RandomForestModel", "ae_logspec+vib_logspec",
         "Best LightGBM", "LightGBMModel", "vib_logspec"),
        # 10. DL CNN vs DL fusion
        ("ResNetVibCNN", "ResNetVibCNN", "vib_spec",
         "BilinearFusion", "BilinearFusionNetwork", "ae_spec+vib_spec+physics+pp"),
        # 11. WST vs DL on vibration-only input
        ("LightGBM Vib-WST", "LightGBMModel", "vib_wst",
         "ResNetVibCNN Vib-dB", "ResNetVibCNN", "vib_spec"),
        # 12. AE spectrogram: classical tree vs deep CNN
        ("RF AE-dB", "RandomForestModel", "ae_spec",
         "ResNetAECNN", "ResNetAECNN", "ae_spec"),
        # 13. AE CNN vs channel-attention CNN on the same AE-spec input
        ("ResNetAECNN", "ResNetAECNN", "ae_spec",
         "ChannelAttentionCNN", "ChannelAttentionCNN", "ae_spec"),
        # 14. vibration CNN vs AE CNN
        ("ResNetVibCNN", "ResNetVibCNN", "vib_spec",
         "ResNetAECNN", "ResNetAECNN", "ae_spec"),
    ]

    rows = []
    for label_a, model_a, cfg_a, label_b, model_b, cfg_b in comparisons:
        try:
            path_a, maes_a = find_result(model_a, cfg_a)
            path_b, maes_b = find_result(model_b, cfg_b)
        except FileNotFoundError as exc:
            print(f"Skipping {label_a} vs {label_b}: {exc}")
            continue
        report = wilcoxon_report(maes_a, maes_b)
        ci_a = bootstrap_ci(maes_a)
        ci_b = bootstrap_ci(maes_b)
        rows.append({
            "comparison": f"{label_a} vs {label_b}",
            "model_a": model_a,
            "config_a": cfg_a,
            "mean_mae_a": float(np.mean(maes_a)),
            "ci_lower_a": ci_a[0],
            "ci_upper_a": ci_a[1],
            "model_b": model_b,
            "config_b": cfg_b,
            "mean_mae_b": float(np.mean(maes_b)),
            "ci_lower_b": ci_b[0],
            "ci_upper_b": ci_b[1],
            "median_diff": report["median_diff"],
            "hl_diff": report["hl_diff"],
            "hl_ci_lower": report["hl_ci_lower"],
            "hl_ci_upper": report["hl_ci_upper"],
            "rank_biserial": report["rank_biserial"],
            "effect_size_mad": effect_size(maes_a, maes_b),
            "wilcoxon_statistic": report["statistic"],
            "pvalue_raw": report["pvalue_raw"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparisons could be made; empty table saved")
        df.to_csv(TABLES_DIR / "statistical_tests_updated.csv", index=False)
        return 1

    # Holm-Bonferroni correction
    pvals = df["pvalue_raw"].to_numpy()
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    corrected = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(sorted_idx):
        corrected[idx] = max(prev, pvals[idx] * (n - rank))
        prev = corrected[idx]
    corrected = np.minimum(corrected, 1.0)
    df["pvalue_holm"] = corrected
    df["significant_05"] = df["pvalue_holm"] < 0.05

    out_csv = TABLES_DIR / "statistical_tests_updated.csv"
    out_md = TABLES_DIR / "statistical_tests_updated.md"
    df.to_csv(out_csv, index=False)

    # Markdown summary
    display = df[[
        "comparison", "mean_mae_a", "mean_mae_b", "hl_diff",
        "hl_ci_lower", "hl_ci_upper", "rank_biserial",
        "pvalue_raw", "pvalue_holm", "significant_05"
    ]].copy()
    display.columns = [
        "Comparison", "MAE A", "MAE B", "HL diff (A-B)",
        "HL CI lo", "HL CI hi", "Rank-biserial",
        "p raw", "p Holm", "sig (α=.05)"
    ]
    with open(out_md, "w") as f:
        f.write("# Updated pairwise statistical tests (LOGO 16-fold)\n\n")
        f.write("Wilcoxon signed-rank test on per-fold MAE, Holm-Bonferroni corrected.\n\n")
        f.write(display.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"Saved {out_csv} and {out_md}")
    print(display.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
