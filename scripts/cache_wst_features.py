#!/usr/bin/env python3
"""
Cache 2D Wavelet Scattering Transform (WST) features for AE and vibration spectrograms.

Loads mean spectrograms from data/intermediate/cached_specs/mean_specs.npz,
computes translation-invariant scattering coefficients with kymatio, averages over
spatial dimensions, and writes a compact feature cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from kymatio import Scattering2D

ROOT = Path(__file__).resolve().parent.parent
SPEC_CACHE = ROOT / "data" / "intermediate" / "cached_specs" / "mean_specs.npz"
OUT_CACHE = ROOT / "data" / "intermediate" / "cached_specs" / "wst_features.npz"

J = 3
L = 8
MAX_ORDER = 2


def compute_wst_features(spec_array: np.ndarray) -> np.ndarray:
    """
    Compute 2D scattering features for a batch of spectrograms.

    Parameters
    ----------
    spec_array : np.ndarray
        Shape (N, C, H, W).

    Returns
    -------
    np.ndarray
        Shape (N, C, K) where K is the number of scattering coefficients.
    """
    N, C, H, W = spec_array.shape
    # Kymatio's 2D API uses L for the number of angular orientations.
    S = Scattering2D(J=J, shape=(H, W), L=L, max_order=MAX_ORDER)

    # First sample to determine K
    first = S(spec_array[0, 0])
    K = first.shape[0]
    features = np.empty((N, C, K), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            sc = S(spec_array[i, c])
            # Average over spatial dims to obtain translation-invariant descriptor.
            features[i, c] = sc.mean(axis=(-2, -1))
        if i % 50 == 0 or i == N - 1:
            print(f"Processed {i + 1}/{N} samples")
    return features


def main() -> None:
    if not SPEC_CACHE.exists():
        raise FileNotFoundError(f"Spectrogram cache not found: {SPEC_CACHE}")

    data = np.load(SPEC_CACHE)
    ae_spec = data["ae_spec"]
    vib_spec = data["vib_spec"]
    sample_ids = data["sample_ids"]
    condition_ids = data["condition_ids"]
    targets = data["targets"]

    print(f"AE spec shape: {ae_spec.shape}")
    print(f"Vib spec shape: {vib_spec.shape}")

    print("Computing AE WST features...")
    ae_wst = compute_wst_features(ae_spec)
    print(f"AE WST shape: {ae_wst.shape}")

    print("Computing vibration WST features...")
    vib_wst = compute_wst_features(vib_spec)
    print(f"Vib WST shape: {vib_wst.shape}")

    OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_CACHE,
        ae_wst=ae_wst,
        vib_wst=vib_wst,
        sample_ids=sample_ids,
        condition_ids=condition_ids,
        targets=targets,
    )
    print(f"Saved WST cache to {OUT_CACHE}")


if __name__ == "__main__":
    main()
