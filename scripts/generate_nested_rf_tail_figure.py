#!/usr/bin/env python3
"""Plot paired outer-fold errors for nested dB-z and log-mel RFs."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

PublicationPlotter.set_style()
TABLES = ROOT / "reports" / "evidence" / "tables"


def main() -> int:
    dbz = pd.read_csv(TABLES / "nested_logo_dbz_matched_folds.csv").sort_values("test_condition")
    mel = pd.read_csv(TABLES / "nested_logo_logmel_matched_folds.csv").sort_values("test_condition")
    managed = MutableFigure(
        "nested_rf_dbz_logmel_folds.png",
        profile=FigureProfiles.DOUBLE,
        out_dir=ROOT / "reports" / "evidence" / "plots" / "results",
        overleaf_dir=ROOT / "overleaf" / "images",
        metadata={"generator": "scripts/generate_nested_rf_tail_figure.py"},
    )
    fig, ax = managed.create()
    ax.plot(dbz.test_condition, dbz.outer_test_mae, marker="o", color=PublicationPalette.OBSERVED, label="Nested dB-z RF")
    ax.plot(mel.test_condition, mel.outer_test_mae, marker="s", color=PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"], label="Nested log-mel RF")
    ax.axvline(7, color=PublicationPalette.CONDITION_7, linestyle="--", linewidth=1, label="Condition 7")
    ax.set_xlabel("Held-out condition")
    ax.set_ylabel("Outer-fold MAE (µm)")
    ax.set_xticks(range(1, 17))
    ax.set_ylim(bottom=0)
    ax.set_title("Nested RF mean similarity masks different tail error")
    ax.legend()
    managed.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
