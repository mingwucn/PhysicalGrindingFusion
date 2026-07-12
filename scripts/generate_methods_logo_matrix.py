#!/usr/bin/env python3
"""Generate the 16x16 LOGO fold-assignment matrix used in Methods.tex.

Outputs:
    reports/evidence/plots/methods/methods_logo_matrix.png
    overleaf/images/methods_logo_matrix.png
"""

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

from grinding_physic_fusion.visualization import FigureProfiles, MutableFigure, PublicationPalette, PublicationPlotter

OUT_DIR = ROOT / "reports" / "evidence" / "plots" / "methods"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PublicationPlotter.set_style()

N = 16
grid = np.zeros((N, N), dtype=int)  # 0=train, 1=val, 2=test
rng = np.random.RandomState(42)
assignments = []
for i in range(N):
    test = i
    val = int(rng.choice(np.delete(np.arange(N), test)))
    grid[i, test] = 2
    grid[i, val] = 1
    assignments.append({
        "fold": i + 1,
        "test_condition": test + 1,
        "validation_condition": val + 1,
        "training_conditions": ",".join(
            str(c + 1) for c in range(N) if c not in (test, val)
        ),
        "assignment": "canonical_seed42",
    })

pd.DataFrame(assignments).to_csv(
    ROOT / "reports" / "evidence" / "tables" / "canonical_logo_assignments.csv",
    index=False,
)

colors = [PublicationPalette.TRAIN, PublicationPalette.VALIDATION, PublicationPalette.TEST]
cmap = plt.matplotlib.colors.ListedColormap(colors)

managed = MutableFigure(
    "methods_logo_matrix.png",
    profile=FigureProfiles.SQUARE,
    out_dir=OUT_DIR,
    metadata={"generator": "scripts/generate_methods_logo_matrix.py"},
)
fig, ax = managed.create()
im = ax.imshow(grid, cmap=cmap, aspect="equal", vmin=0, vmax=2)
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))
ax.set_xticklabels(np.arange(1, N + 1), fontsize=6)
ax.set_yticklabels(np.arange(1, N + 1), fontsize=6)
ax.set_xlabel("Condition")
ax.set_ylabel("LOGO fold (test condition)")
ax.set_title("Leave-one-condition-out fold assignment")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], edgecolor="k", label="Train"),
    Patch(facecolor=colors[1], edgecolor="k", label="Validation"),
    Patch(facecolor=colors[2], edgecolor="k", label="Test"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=6)

fig.tight_layout()
managed.save()
