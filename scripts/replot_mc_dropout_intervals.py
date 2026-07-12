#!/usr/bin/env python3
"""Replot archived full-LOGO MC-dropout intervals with publication labels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPlotter

PublicationPlotter.set_style()


def main() -> int:
    source = ROOT / "reports/evidence/uncertainty/mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv"
    frame = pd.read_csv(source).sort_values("y_pred").reset_index(drop=True)
    managed = MutableFigure(
        "mc_dropout_intervals_ResNetVibCNN_vib_spec.png",
        profile=FigureProfiles.DOUBLE,
        out_dir=ROOT / "reports/evidence/uncertainty",
        overleaf_dir=ROOT / "overleaf/images",
        metadata={"generator": "scripts/replot_mc_dropout_intervals.py"},
    )
    fig, ax = managed.create()
    index = np.arange(len(frame))
    ax.fill_between(
        index, frame["lower"], frame["upper"], alpha=0.3,
        label="Nominal 95% MC-dropout epistemic band",
    )
    ax.scatter(index, frame["y_true"], s=13, label=r"Observed $R_a$")
    ax.set_xlabel("Sample index (sorted by stochastic-mean prediction)")
    ax.set_ylabel(r"$R_a$ ($\mu$m)")
    ax.set_title("MC-dropout epistemic intervals: full LOGO (ResNetVibCNN / Vib-dB)")
    ax.legend(loc="upper left")
    managed.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
