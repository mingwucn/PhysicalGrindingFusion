#!/usr/bin/env python3
"""Export the exact evaluation target vector and condition summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from grinding_physic_fusion.data.dataset import load_surface_roughness


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "evidence" / "tables"
MISSING_SAMPLE = (1, 1)


def main() -> int:
    values = load_surface_roughness(ROOT / "data" / "surface roughness.csv")
    rows = []
    for condition_id in range(1, 17):
        for sample_id in range(1, 21):
            if (condition_id, sample_id) == MISSING_SAMPLE:
                continue
            source_index = (condition_id - 1) * 20 + sample_id - 1
            rows.append(
                {
                    "evaluation_index": len(rows),
                    "source_index": source_index,
                    "condition_id": condition_id,
                    "sample_id": sample_id,
                    "measured_ra_um": float(values[source_index]),
                }
            )

    TABLES.mkdir(parents=True, exist_ok=True)
    targets = pd.DataFrame(rows)
    target_path = TABLES / "final_evaluation_targets.csv"
    targets.to_csv(target_path, index=False, float_format="%.8f")

    summary = targets.groupby("condition_id", as_index=False).agg(
        n=("measured_ra_um", "count"),
        mean_measured_ra_um=("measured_ra_um", "mean"),
        sd_measured_ra_um=("measured_ra_um", lambda values: values.std(ddof=0)),
    )
    summary_path = TABLES / "condition_target_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8f")

    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    metadata = {
        "target_file": str(target_path.relative_to(ROOT)),
        "sha256": digest,
        "n_evaluation_samples": len(targets),
        "excluded_sample": {"condition_id": 1, "sample_id": 1},
        "condition_7_mean_measured_ra_um": float(summary.loc[summary.condition_id == 7, "mean_measured_ra_um"].iloc[0]),
        "condition_7_sd_measured_ra_um": float(summary.loc[summary.condition_id == 7, "sd_measured_ra_um"].iloc[0]),
    }
    (TABLES / "final_evaluation_targets_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
