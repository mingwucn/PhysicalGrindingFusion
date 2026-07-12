#!/usr/bin/env python3
"""Compute RF LOGO grouped by wheel speed for the recommended representation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from comprehensive_sensitivity_suite import flatten_sample, load_data, rf_fit_predict


def main() -> int:
    condition_ids, _sample_ids, y, ae, vib, params = load_data()
    X = np.array([flatten_sample(ae[i], vib[i]) for i in range(len(y))])
    preds = np.zeros_like(y)
    rows = []
    for level in np.unique(params[:, 0]):
        test_conditions = np.where(params[:, 0] == level)[0] + 1
        train = ~np.isin(condition_ids, test_conditions)
        test = np.isin(condition_ids, test_conditions)
        _, _, y_pred = rf_fit_predict(X[train], y[train], X[test])
        preds[test] = y_pred
        group_mae = float(np.abs(y_pred - y[test]).mean())
        rows.append(
            {
                "wheel_speed_m_s": float(level),
                "test_conditions": ";".join(map(str, test_conditions.tolist())),
                "n_test": int(test.sum()),
                "mae": group_mae,
            }
        )
        print(
            f"wheel_speed={level:g}, test_conditions={test_conditions.tolist()}, "
            f"n_test={int(test.sum())}, mae={group_mae:.12f}",
            flush=True,
        )
    mae = float(np.abs(preds - y).mean())
    out_path = Path(__file__).resolve().parents[1] / "reports" / "evidence" / "tables" / "wheel_speed_group_logo.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)
    print(f"LOGO by wheel speed,4,{mae:.12f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
