#!/usr/bin/env python3
"""Compare the near-matched Conditions 7 and 10 using archived caches."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import load_process_parameters
from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

CACHE = ROOT / "data" / "intermediate" / "cached_specs" / "mean_specs.npz"
TABLE = ROOT / "reports" / "evidence" / "tables" / "condition7_condition10_diagnostic.csv"
PLOTS = ROOT / "reports" / "evidence" / "plots" / "results"


def main() -> int:
    PublicationPlotter.set_style()
    with np.load(CACHE, allow_pickle=True) as data:
        conditions = np.asarray(data["condition_ids"], dtype=int)
        targets = np.asarray(data["targets"], dtype=float)
        ae = np.asarray(data["ae_spec"], dtype=float)
        vib = np.asarray(data["vib_spec"], dtype=float)
    params = np.asarray(load_process_parameters(), dtype=float)

    rows = []
    spectra = {}
    for condition in (7, 10):
        mask = conditions == condition
        ae_mean = ae[mask].mean(axis=(0, 1, 3))
        vib_mean = vib[mask].mean(axis=(0, 1, 3))
        spectra[condition] = (ae_mean, vib_mean)
        rows.append({
            "condition": condition,
            "wheel_speed_m_s": params[condition - 1, 0],
            "workpiece_speed_r_min": params[condition - 1, 1],
            "depth_um": params[condition - 1, 2],
            "n_passes": int(mask.sum()),
            "mean_measured_ra_um": float(targets[mask].mean()),
            "std_measured_ra_um": float(targets[mask].std(ddof=0)),
        })
    pd.DataFrame(rows).to_csv(TABLE, index=False)

    ae_f = np.arange(ae.shape[2]) * 4_000_000 / 598 / 1e3
    vib_f = np.arange(vib.shape[2]) * 51_200 / 512 / 1e3
    managed = MutableFigure("condition7_condition10_spectra.png", profile=FigureProfiles.TWO_PANEL_ROW, out_dir=PLOTS, overleaf_dir=ROOT / "overleaf" / "images", metadata={"generator": "scripts/condition7_condition10_diagnostic.py"})
    fig, axes = managed.create()
    for condition, colour in ((7, PublicationPalette.CONDITION_7), (10, PublicationPalette.OBSERVED)):
        axes[0].plot(ae_f, spectra[condition][0], label=f"Condition {condition}", color=colour)
        axes[1].plot(vib_f, spectra[condition][1], label=f"Condition {condition}", color=colour)
    axes[0].set(xlabel="AE frequency (kHz)", ylabel="Mean dB level")
    axes[1].set(xlabel="Vibration frequency (kHz)", ylabel="Mean dB level")
    handles, labels = axes[0].get_legend_handles_labels()
    PublicationPlotter.figure_legend_below(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.94, wspace=0.30)
    managed.save()
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
