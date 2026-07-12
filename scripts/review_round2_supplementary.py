#!/usr/bin/env python3
"""Supplementary analyses for the second scientific-review round.

Outputs:
  reports/evidence/tables/supp_holm_full_family.csv
  reports/evidence/tables/supp_condition7_sensitivity.csv
  reports/evidence/tables/supp_per_condition_mae.csv
  reports/evidence/tables/supp_measurement_uncertainty.csv
  reports/evidence/tables/supp_run_order.csv
  reports/evidence/plots/supp/run_order.png
  reports/evidence/plots/supp/per_condition_mae.png
  overleaf/images/supp_run_order.png
  overleaf/images/supp_per_condition_mae.png
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grinding_physic_fusion.data.dataset import load_all_data
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter
from train_and_evaluate import scale_data_dict

PublicationPlotter.set_style()

OUT_DIR = ROOT / "reports" / "evidence" / "tables"
PLOTS_DIR = ROOT / "reports" / "evidence" / "plots" / "supp"
OVERLEAF_DIR = ROOT / "overleaf" / "images"
for d in (OUT_DIR, PLOTS_DIR, OVERLEAF_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_fold_arrays(path: Path, model: str, config: str) -> np.ndarray | None:
    df = pd.read_csv(path)
    row = df[(df["model"] == model) & (df["config"] == config)]
    if row.empty:
        return None
    folds = ast.literal_eval(row.iloc[0]["folds"])
    return np.array([f["mae"] for f in sorted(folds, key=lambda x: x["fold"])])


def bootstrap_paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
    """Bootstrap 95% CI for median paired difference (a - b)."""
    diffs = a - b
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot.append(np.median(diffs[idx]))
    boot = np.sort(boot)
    return float(boot[250]), float(boot[9750])


def holm_full_family() -> pd.DataFrame:
    """Load the pre-computed 14-comparison family and export it."""
    df = pd.read_csv(OUT_DIR / "statistical_tests_updated.csv")
    df = df.rename(columns={"pvalue_raw": "p_raw", "pvalue_holm": "p_Holm"})
    df["significant_05"] = df["p_Holm"] < 0.05
    df.to_csv(OUT_DIR / "supp_holm_full_family.csv", index=False)
    return df


def condition7_sensitivity() -> pd.DataFrame:
    """Paired comparisons with and without condition 7."""
    path = OUT_DIR / "full_results_logo_only.csv"
    comparisons = [
        ("Best RF vs RF AE-spec + Vib-spec + PP", "RandomForestModel", "ae_logspec+vib_logspec", "RandomForestModel", "ae_spec+vib_spec+pp"),
        ("Best RF vs ResNetVibCNN (Vib-spec)", "RandomForestModel", "ae_logspec+vib_logspec", "ResNetVibCNN", "vib_spec"),
        ("Best RF vs LightGBM (Vib-logspec)", "RandomForestModel", "ae_logspec+vib_logspec", "LightGBMModel", "vib_logspec"),
    ]
    rows = []
    for label, ma, ca, mb, cb in comparisons:
        a = load_fold_arrays(path, ma, ca)
        b = load_fold_arrays(path, mb, cb)
        if a is None or b is None:
            continue
        # All folds
        med_all = float(np.median(a - b))
        mean_all = float(np.mean(a - b))
        _, p_all = stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
        ci_lo_all, ci_hi_all = bootstrap_paired_ci(a, b)
        # Exclude condition 7 (fold index 6)
        a_ex = np.delete(a, 6)
        b_ex = np.delete(b, 6)
        med_ex = float(np.median(a_ex - b_ex))
        mean_ex = float(np.mean(a_ex - b_ex))
        _, p_ex = stats.wilcoxon(a_ex, b_ex, alternative="two-sided", zero_method="zsplit")
        ci_lo_ex, ci_hi_ex = bootstrap_paired_ci(a_ex, b_ex)
        rows.append({
            "comparison": label,
            "median_diff_all": med_all,
            "mean_diff_all": mean_all,
            "p_wilcoxon_all": p_all,
            "ci_lo_all": ci_lo_all,
            "ci_hi_all": ci_hi_all,
            "median_diff_excl_c7": med_ex,
            "mean_diff_excl_c7": mean_ex,
            "p_wilcoxon_excl_c7": p_ex,
            "ci_lo_excl_c7": ci_lo_ex,
            "ci_hi_excl_c7": ci_hi_ex,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "supp_condition7_sensitivity.csv", index=False)
    return df


def per_condition_mae() -> pd.DataFrame:
    """Table of per-condition MAE for top models."""
    path = OUT_DIR / "full_results_logo_only.csv"
    models = [
        ("Best RF", "RandomForestModel", "ae_logspec+vib_logspec"),
        ("ResNetVibCNN", "ResNetVibCNN", "vib_spec"),
        ("LightGBM WST", "LightGBMModel", "vib_wst"),
        ("LightGBM (Vib-dB-z)", "LightGBMModel", "vib_logspec"),
    ]
    arrays = {}
    for label, m, c in models:
        arr = load_fold_arrays(path, m, c)
        if arr is not None:
            arrays[label] = arr
    df = pd.DataFrame({"condition_id": np.arange(1, 17)})
    for label, arr in arrays.items():
        df[label] = arr
    df.to_csv(OUT_DIR / "supp_per_condition_mae.csv", index=False)

    # Figure
    managed = MutableFigure("supp_per_condition_mae.png", profile=FigureProfiles.DOUBLE, out_dir=PLOTS_DIR, overleaf_dir=OVERLEAF_DIR, metadata={"generator": "scripts/review_round2_supplementary.py"})
    fig, ax = managed.create()
    x = np.arange(1, 17)
    for label, arr in arrays.items():
        ax.plot(x, arr, marker="o", label=label, lw=1)
    ax.axvline(7, color="gray", ls="--", lw=0.8, label="Condition 7")
    ax.set_xlabel("Condition ID")
    ax.set_ylabel("LOGO MAE (µm)")
    ax.set_title("Per-condition LOGO MAE for top models")
    ax.legend()
    fig.tight_layout()
    managed.save()
    return df


def measurement_uncertainty() -> pd.DataFrame:
    """Summary of measured roughness variability."""
    sr = pd.read_csv(ROOT / "data" / "surface roughness.csv")
    sr.columns = ["Ra"]
    sr["condition_id"] = (sr.index // 20) + 1
    sr["sample_id"] = (sr.index % 20) + 1
    # Condition 1 sample 1 is missing sensor data; flag but keep for measurement stats
    sr["missing_sensor"] = (sr["condition_id"] == 1) & (sr["sample_id"] == 1)
    cond = sr.groupby("condition_id")["Ra"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    cond.to_csv(OUT_DIR / "supp_measurement_uncertainty.csv", index=False)
    return cond


def run_order() -> pd.DataFrame:
    """Use CSV row order as experimental run order and plot measured Ra."""
    sr = pd.read_csv(ROOT / "data" / "surface roughness.csv")
    sr.columns = ["Ra"]
    sr["pass_order"] = sr.index + 1
    sr["condition_id"] = (sr.index // 20) + 1
    sr["sample_id"] = (sr.index % 20) + 1
    # Drop the pass with missing sensor data for the plot note
    sr.to_csv(OUT_DIR / "supp_run_order.csv", index=False)

    managed = MutableFigure("supp_run_order.png", profile=FigureProfiles.DOUBLE, out_dir=PLOTS_DIR, overleaf_dir=OVERLEAF_DIR, metadata={"generator": "scripts/review_round2_supplementary.py"})
    fig, ax = managed.create()
    cmap = plt.cm.tab20
    sc = ax.scatter(sr["pass_order"], sr["Ra"], c=sr["condition_id"], cmap=cmap, s=12)
    # Highlight condition 7
    c7 = sr[sr["condition_id"] == 7]
    ax.scatter(c7["pass_order"], c7["Ra"], c=PublicationPalette.CONDITION_7, s=35, marker="x", label="Condition 7")
    ax.set_xlabel("Pass order")
    ax.set_ylabel("Measured $R_a$ (µm)")
    ax.set_title("Run-order plot of measured surface roughness")
    ax.legend()
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Condition ID")
    fig.tight_layout()
    managed.save()
    return sr


def rf_cross_calibrated_intervals() -> pd.DataFrame:
    """RF 95% intervals calibrated on an inner held-out condition.

    For each outer LOGO fold, one of the 15 training conditions is reserved
    as an internal calibration set. A model is trained on the remaining 14
    conditions, its predictions on the calibration condition are used to
    compute a conformal scaling factor, and that factor is applied to the
    tree-standard-deviation scores produced by the model trained on all 15
    training conditions. Test residuals therefore do not influence interval
    width.
    """
    CONFIG = "ae_logspec+vib_logspec"
    ALPHA = 0.05
    full = load_all_data(config=CONFIG)
    scaled, _ = scale_data_dict(full, np.arange(len(full["targets"])), scale_specs=True, scale_target=False)
    X = np.concatenate([scaled[k].reshape(scaled[k].shape[0], -1) for k in sorted(scaled.keys()) if k not in {"targets", "condition_ids", "sample_ids"} and scaled[k] is not None], axis=1)
    y = scaled["targets"]
    groups = scaled["condition_ids"]

    preds_all, y_true_all, cond_all, std_all, lower_all, upper_all = [], [], [], [], [], []
    for g in np.unique(groups):
        test_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_groups = groups[train_idx]

        # Inner calibration: hold out the first training condition
        calib_cond = train_groups[0]
        inner_train_idx = np.where(train_groups != calib_cond)[0]
        inner_calib_idx = np.where(train_groups == calib_cond)[0]
        X_inner_train, y_inner_train = X_train[inner_train_idx], y_train[inner_train_idx]
        X_calib, y_calib = X_train[inner_calib_idx], y_train[inner_calib_idx]

        inner_model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
        inner_model.fit(X_inner_train, y_inner_train)
        calib_pred = inner_model.predict(X_calib)
        calib_tree = np.array([tree.predict(X_calib) for tree in inner_model.estimators_])
        calib_std = calib_tree.std(axis=0)
        calib_resid = np.abs(y_calib - calib_pred)
        ratios = calib_resid / np.where(calib_std > 1e-9, calib_std, 1e-9)
        k = int(np.ceil((1 - ALPHA) * (len(ratios) + 1)))
        scale = float(np.sort(ratios)[k - 1])

        # Final outer model trained on all 15 training conditions
        outer_model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
        outer_model.fit(X_train, y_train)
        pred_test = outer_model.predict(X_test)
        test_tree = np.array([tree.predict(X_test) for tree in outer_model.estimators_])
        test_std = test_tree.std(axis=0)

        preds_all.extend(pred_test)
        y_true_all.extend(y_test)
        cond_all.extend([g] * len(test_idx))
        std_all.extend(test_std)
        lower_all.extend(pred_test - scale * test_std)
        upper_all.extend(pred_test + scale * test_std)

    df = pd.DataFrame({
        "condition_id": cond_all,
        "y_true": y_true_all,
        "y_pred": preds_all,
        "y_std": std_all,
        "lower": lower_all,
        "upper": upper_all,
    })
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["interval_width"] = df["upper"] - df["lower"]
    coverage = df["covered"].mean()
    mae = np.abs(df["y_true"] - df["y_pred"]).mean()
    mean_width = df["interval_width"].mean()
    print(f"Cross-calibrated RF intervals: coverage={coverage:.3f}, MAE={mae:.5f}, mean width={mean_width:.5f}")
    df.to_csv(OUT_DIR / "supp_rf_cross_intervals.csv", index=False)
    return df


def main() -> int:
    print("Holm full family ...")
    holm_full_family()
    print("Condition 7 sensitivity ...")
    condition7_sensitivity()
    print("Per-condition MAE ...")
    per_condition_mae()
    print("Measurement uncertainty ...")
    measurement_uncertainty()
    print("Run order ...")
    run_order()
    print("RF cross-calibrated intervals ...")
    rf_cross_calibrated_intervals()
    return 0


if __name__ == "__main__":
    sys.exit(main())
