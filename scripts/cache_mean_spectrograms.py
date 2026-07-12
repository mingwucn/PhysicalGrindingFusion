#!/usr/bin/env python3
"""
Pre-compute and cache mean spectrograms for all valid samples.

This avoids repeatedly reading the large per-sample *_spec.npz files during
training.  Cached arrays are aligned with condition_ids/sample_ids and can be
loaded in O(seconds) instead of O(many minutes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.data.dataset import (
    INTERMEDIATE_DIR,
    MISSING_SAMPLE,
    compute_mean_spectrogram,
    discover_samples,
    load_process_parameters,
    load_surface_roughness,
)

CACHE_DIR = INTERMEDIATE_DIR / "cached_specs"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pairs = [p for p in discover_samples(config=None) if p != MISSING_SAMPLE]
    params = load_process_parameters()
    roughness = load_surface_roughness()

    ae_specs, vib_specs = [], []
    condition_ids, sample_ids, targets = [], [], []

    print(f"Caching mean spectrograms for {len(pairs)} samples ...")
    for cid, sid in pairs:
        spec_path = INTERMEDIATE_DIR / f"{cid}-{sid:02d}-0_spec.npz"
        spec_data = np.load(spec_path, allow_pickle=True)
        ae_specs.append(compute_mean_spectrogram(spec_data["spec_ae"]))
        vib_specs.append(compute_mean_spectrogram(spec_data["spec_vib"]))
        spec_data.close()

        condition_ids.append(cid)
        sample_ids.append(sid)
        targets.append(roughness[(cid - 1) * 20 + (sid - 1)])

    np.savez(
        CACHE_DIR / "mean_specs.npz",
        ae_spec=np.stack(ae_specs).astype(np.float32),
        vib_spec=np.stack(vib_specs).astype(np.float32),
        condition_ids=np.array(condition_ids, dtype=np.int64),
        sample_ids=np.array(sample_ids, dtype=np.int64),
        targets=np.array(targets, dtype=np.float32),
    )
    print(f"Saved cache -> {CACHE_DIR / 'mean_specs.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
