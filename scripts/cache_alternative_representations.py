#!/usr/bin/env python3
"""
Cache alternative time-frequency representations from the existing mean spectrograms.

Inputs:
    data/intermediate/cached_specs/mean_specs.npz

Outputs:
    data/intermediate/cached_specs/alternative_reps.npz
    containing:
        - ae_logspec  : per-sample z-scored AE dB spectrogram
        - vib_logspec : per-sample z-scored vibration dB spectrogram
        - ae_mel      : AE log-mel spectrogram (64 mel bins, dB)
        - vib_mel     : vibration log-mel spectrogram (64 mel bins, dB)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "intermediate" / "cached_specs"
IN_PATH = CACHE_DIR / "mean_specs.npz"
OUT_PATH = CACHE_DIR / "alternative_reps.npz"


def compute_per_sample_zscore(spec: np.ndarray) -> np.ndarray:
    """
    Standardise an already dB-scaled spectrogram to zero mean / unit std.

    The historical ``*_logspec`` cache name is retained for compatibility,
    but this operation does not apply another logarithm.
    """
    spec = np.asarray(spec, dtype=np.float64)
    # Per-sample, per-channel standardisation over freq-time plane
    mean = spec.mean(axis=(2, 3), keepdims=True)
    std = spec.std(axis=(2, 3), keepdims=True) + 1e-8
    return ((spec - mean) / std).astype(np.float32)


def compute_mel_spec(spec: np.ndarray, sr: float, n_fft: int, n_mels: int = 64) -> np.ndarray:
    """
    Compute a log-mel spectrogram from a dB-scaled spectrogram.

    The input is assumed to be in dB (as stored in mean_specs.npz). It is
    converted to linear power, projected onto a mel filterbank, and converted
    back to dB.

    spec: (n_samples, n_channels, n_freq, n_time)
    returns: (n_samples, n_channels, n_mels, n_time)
    """
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for mel-spectrogram computation") from exc

    n_samples, n_channels, n_freq, n_time = spec.shape
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (n_mels, n_freq)

    # Convert dB -> linear power
    power = np.maximum(10.0 ** (spec / 10.0), 1e-10)

    # Reshape for batch matrix multiply
    power_t = power.transpose(0, 1, 3, 2)  # (n_samples, n_channels, n_time, n_freq)
    flat = power_t.reshape(-1, n_time, n_freq)  # (n_samples*n_channels, n_time, n_freq)
    mel_flat = np.matmul(flat, mel_filter.T)  # (..., n_time, n_mels)
    mel = mel_flat.reshape(n_samples, n_channels, n_time, n_mels)
    mel = mel.transpose(0, 1, 3, 2)  # (n_samples, n_channels, n_mels, n_time)

    # Convert back to dB
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return mel_db.astype(np.float32)


def main() -> int:
    if not IN_PATH.exists():
        print(f"Input cache not found: {IN_PATH}")
        print("Run scripts/cache_mean_spectrograms.py first.")
        return 1

    print(f"Loading {IN_PATH} ...")
    data = np.load(IN_PATH, allow_pickle=True)
    ae_spec = data["ae_spec"]
    vib_spec = data["vib_spec"]

    print("Computing per-sample z-scored dB spectrograms ...")
    ae_logspec = compute_per_sample_zscore(ae_spec)
    vib_logspec = compute_per_sample_zscore(vib_spec)

    print("Computing mel-compressed spectrograms ...")
    # AE: 300 freq bins -> n_fft = 598 (598//2+1 = 300)
    ae_mel = compute_mel_spec(ae_spec, sr=4_000_000.0, n_fft=598, n_mels=64)
    # Vibration: 257 freq bins -> n_fft = 512 (512//2+1 = 257)
    vib_mel = compute_mel_spec(vib_spec, sr=51_200.0, n_fft=512, n_mels=64)

    print(f"Saving to {OUT_PATH} ...")
    np.savez_compressed(
        OUT_PATH,
        ae_logspec=ae_logspec,
        vib_logspec=vib_logspec,
        ae_mel=ae_mel,
        vib_mel=vib_mel,
    )

    print("Done.")
    print(f"  ae_logspec : {ae_logspec.shape}")
    print(f"  vib_logspec: {vib_logspec.shape}")
    print(f"  ae_mel     : {ae_mel.shape}")
    print(f"  vib_mel    : {vib_mel.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
