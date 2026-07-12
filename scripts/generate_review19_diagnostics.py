#!/usr/bin/env python3
"""Generate review-19 fold diagnostics from archived CSV artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "reports" / "evidence"
TABLES = EVIDENCE / "tables"
PRED = EVIDENCE / "predictions"
UNC = EVIDENCE / "uncertainty"
TEX = ROOT / "overleaf" / "main"


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def generate_nested_table() -> None:
    frames = []
    for label, filename in (
        ("dB-z", "nested_logo_dbz_matched_folds.csv"),
        ("log-mel", "nested_logo_logmel_matched_folds.csv"),
    ):
        frame = pd.read_csv(TABLES / filename)
        frame["representation"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for _, row in data.iterrows():
        hp = json.loads(row["best_hparams"])
        depth = "None" if hp["max_depth"] is None else str(hp["max_depth"])
        rows.append(
            f"{row['representation']} & {int(row['test_condition'])} & {int(hp['n_estimators'])} & "
            f"{depth} & {row['inner_val_mae']:.4f} & {row['outer_test_mae']:.4f} \\\\"
        )
    content = r"""\begin{longtable}{@{}lrrrrr@{}}
\caption{Per-outer-fold selections for the current nested RF sensitivity.
The five inner folds are grouped by grinding condition. Selection minimises
mean inner-fold MAE; the selected RF is refitted on the same 14 outer-training
conditions used by the fixed RF, while the archived validation condition
remains excluded, and is evaluated once on the held-out outer condition.}
\label{tab:supp-nested-rf-selections}\\
\toprule
Representation & Test condition & Trees & Maximum depth & Inner MAE & Outer MAE \\
\midrule
\endfirsthead
\toprule
Representation & Test condition & Trees & Maximum depth & Inner MAE & Outer MAE \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
"""
    (TEX / "supp_nested_rf_selections.tex").write_text(content)


def generate_mc_table() -> None:
    mc = pd.read_csv(UNC / "mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv")
    rows = []
    records = []
    if "y_pred_deterministic" not in mc.columns:
        raise ValueError("Regenerate MC-dropout CSV with same-checkpoint deterministic predictions")
    for fold in range(16):
        stochastic = mc.loc[mc["fold"] == fold].reset_index(drop=True)
        deterministic_mae = np.abs(stochastic["y_true"] - stochastic["y_pred_deterministic"]).mean()
        stochastic_mae = np.abs(stochastic["y_true"] - stochastic["y_pred"]).mean()
        prediction_shift = np.abs(stochastic["y_pred_deterministic"] - stochastic["y_pred"]).mean()
        condition = int(stochastic["condition_id"].iloc[0])
        coverage = stochastic["covered"].mean()
        records.append({
            "condition": condition,
            "deterministic_mae": deterministic_mae,
            "stochastic_mean_mae": stochastic_mae,
            "mean_abs_prediction_shift": prediction_shift,
            "coverage": coverage,
        })
        rows.append(
            f"{condition} & {deterministic_mae:.4f} & {stochastic_mae:.4f} & "
            f"{prediction_shift:.4f} & {100 * coverage:.1f} \\\\"
        )
    pd.DataFrame(records).to_csv(TABLES / "mc_dropout_deterministic_diagnostic.csv", index=False)
    content = r"""\begin{table}[htbp]
\centering
\caption{Fold-specific diagnostic comparing deterministic checkpoint
predictions with the mean of 50 stochastic MC-dropout forwards for
ResNetVibCNN on Vib-dB. Prediction shift is the mean absolute difference
between deterministic and stochastic-mean predictions.}
\label{tab:supp-mcdropout-deterministic}
\begin{tabular}{@{}rrrrr@{}}
\toprule
Condition & Deterministic MAE & Stochastic MAE & Prediction shift & Coverage (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TEX / "supp_mcdropout_deterministic.tex").write_text(content)


def main() -> int:
    generate_nested_table()
    generate_mc_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
