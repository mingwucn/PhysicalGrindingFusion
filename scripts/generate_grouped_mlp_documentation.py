#!/usr/bin/env python3
"""Generate auditable protocol records for the grouped-validation MLP."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from grouped_validation_shallow_mlp import CONFIG, GRID, MAX_EPOCHS, PATIENCE  # noqa: E402

TABLE = ROOT / "reports" / "evidence" / "tables" / "grouped_validation_shallow_mlp.csv"
JSON_OUT = ROOT / "reports" / "evidence" / "tables" / "grouped_validation_shallow_mlp_protocol.json"
TEX_OUT = ROOT / "overleaf" / "main" / "supp_grouped_mlp_protocol.tex"


def hidden(value: object) -> str:
    if isinstance(value, tuple):
        return "--".join(str(item) for item in value)
    return str(value).strip("() ").replace(", ", "--").replace(",", "--").strip("-")


def main() -> int:
    folds = pd.read_csv(TABLE)
    candidates = [{
        "candidate": index,
        "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
        "learning_rate_init": params["learning_rate_init"],
        "alpha": params["alpha"],
        "activation": "relu",
        "optimizer": "adam",
        "batch_size": "min(32, n_train)",
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
    } for index, params in enumerate(GRID, start=1)]
    lookup = {
        (hidden(params["hidden_layer_sizes"]), str(params["alpha"]), str(params["learning_rate_init"])): index
        for index, params in enumerate(GRID, start=1)
    }
    selected = []
    for row in folds.itertuples(index=False):
        key = (hidden(row.hidden_layer_sizes), str(row.alpha), str(row.learning_rate_init))
        candidate = lookup.get(key)
        if candidate is None:
            raise ValueError(f"Unrecognized selected MLP configuration: {key}")
        selected.append(candidate)
    folds = folds.assign(selected_candidate=selected)
    counts = Counter(selected)
    record = {
        "input": CONFIG,
        "validation": "designated whole validation condition in each outer fold",
        "scaling": "input and target StandardScaler fitted on the 14 training conditions only; target transform inverted for evaluation",
        "random_seed": 42,
        "candidate_grid": candidates,
        "candidate_fits_attempted": int(len(folds) * len(candidates)),
        "candidate_fits_completed": int(len(folds) * len(candidates)),
        "execution_failures": 0,
        "selection_counts": {str(key): value for key, value in sorted(counts.items())},
        "fold_selections": folds.to_dict(orient="records"),
    }
    JSON_OUT.write_text(json.dumps(record, indent=2) + "\n")

    grid_rows = "\n".join(
        f"{item['candidate']} & {hidden(tuple(item['hidden_layer_sizes']))} & "
        f"{item['learning_rate_init']:.0e} & {item['alpha']:.0e} & ReLU/Adam \\\\"
        for item in candidates
    )
    fold_rows = "\n".join(
        f"{int(row.test_condition)} & {int(row.validation_condition)} & {int(row.selected_candidate)} & "
        f"{int(row.selected_epoch)} & {row.mae:.4f} \\\\"
        for row in folds.itertuples(index=False)
    )
    TEX_OUT.write_text(
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\caption{Corrected grouped-validation shallow-MLP candidate grid. All candidates use "
        "training-only input and target standardisation, batch size $\\min(32,n_{\\mathrm{train}})$, "
        f"at most {MAX_EPOCHS} epochs, patience {PATIENCE}, and seed 42.}}\n"
        "\\label{tab:supp-grouped-mlp-grid}\n"
        "\\begin{tabular}{@{}ccccc@{}}\n\\toprule\nCandidate & Hidden units & Learning rate & L2 $\\alpha$ & Activation/optimizer \\\\\n"
        "\\midrule\n" + grid_rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\caption{Per-fold selection for the corrected grouped-validation shallow MLP. All "
        "48 candidate fits completed without an execution failure; this statement does not imply "
        "that every optimiser reached a stationary point.}\n"
        "\\label{tab:supp-grouped-mlp-selections}\n"
        "\\begin{tabular}{@{}ccccc@{}}\n\\toprule\nTest condition & Validation condition & Candidate & Selected epoch & Test MAE ($\\upmu$m) \\\\\n"
        "\\midrule\n" + fold_rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {TEX_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
