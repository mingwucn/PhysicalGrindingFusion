#!/usr/bin/env python3
"""Regenerate only the RF validation-scheme comparison table."""
from __future__ import annotations

import numpy as np

from comprehensive_sensitivity_suite import cv_scheme_comparison, flatten_sample, load_data


def main() -> int:
    condition_ids, _sample_ids, y, ae, vib, _params = load_data()
    X = np.array([flatten_sample(ae[i], vib[i]) for i in range(len(y))])
    df = cv_scheme_comparison(condition_ids, y, X)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
