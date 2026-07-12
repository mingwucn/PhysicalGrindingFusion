#!/usr/bin/env python3
"""Merge disjoint nested-LOGO worker shards into auditable summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, dest="pattern")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = sorted(
        path for path in Path().glob(args.pattern) if not path.stem.endswith("_folds")
    )
    if not paths:
        raise FileNotFoundError(f"No shard CSV files matched {args.pattern}")

    fold_paths = [path.with_name(f"{path.stem}_folds.csv") for path in paths]
    if missing := [path for path in fold_paths if not path.exists()]:
        raise FileNotFoundError(f"Missing fold-selection shards: {missing}")

    folds = pd.concat((pd.read_csv(path) for path in fold_paths), ignore_index=True)
    folds.sort_values(["model", "config", "fold"], inplace=True)
    if folds.duplicated(["model", "config", "fold"]).any():
        raise ValueError("Nested worker shards contain duplicate outer folds")
    counts = folds.groupby(["model", "config"])["fold"].nunique()
    if not (counts == 16).all():
        raise ValueError(f"Expected 16 outer folds per model/config, got {counts.to_dict()}")

    summary = (
        folds.groupby(["model", "config"], as_index=False)["outer_test_mae"]
        .agg(mean_mae="mean", std_mae=lambda values: values.std(ddof=0), n_folds="count")
        .sort_values(["model", "config"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    folds.to_csv(args.output.with_name(f"{args.output.stem}_folds.csv"), index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
