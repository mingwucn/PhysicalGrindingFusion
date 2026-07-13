#!/usr/bin/env python3
"""Generate a cache provenance table for the principal time-frequency caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "intermediate" / "cached_specs"
TABLE_DIR = ROOT / "reports" / "evidence" / "tables"
OVERLEAF_MAIN = ROOT / "overleaf" / "main"


def checksum(arr: np.ndarray) -> str:
    data = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(data).hexdigest()


def latex_escape(value: object) -> str:
    """Escape generated plain text for use in LaTeX table cells."""
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def main() -> int:
    mean = np.load(CACHE_DIR / "mean_specs.npz", allow_pickle=True)
    alt = np.load(CACHE_DIR / "alternative_reps.npz", allow_pickle=True)

    rows = [
        {
            "cache": "ae_spec",
            "source": "Arithmetic mean of 2,910 local AE dB maps",
            "freq_transform": "Linear STFT bins (NFFT 598, 300 bins)",
            "amp_transform": "Magnitude -> dB, clipped at -80 dB, mean over local-map axis",
            "normalization": "None in cache; no fold-wise scaling in RF spectral runs",
            "shape": str(tuple(mean["ae_spec"].shape)),
            "sha256": checksum(mean["ae_spec"]),
        },
        {
            "cache": "vib_spec",
            "source": "Arithmetic mean of 2,910 local vibration dB maps",
            "freq_transform": "Linear STFT bins (NFFT 512, 257 bins)",
            "amp_transform": "Magnitude -> dB, clipped at -80 dB, mean over local-map axis",
            "normalization": "None in cache; no fold-wise scaling in RF spectral runs",
            "shape": str(tuple(mean["vib_spec"].shape)),
            "sha256": checksum(mean["vib_spec"]),
        },
        {
            "cache": "ae_logspec",
            "source": "Pass-mean AE dB cache",
            "freq_transform": "None",
            "amp_transform": "Per-sample z-score of dB spectrogram",
            "normalization": "Per-sample/channel standardization after dB transform",
            "shape": str(tuple(alt["ae_logspec"].shape)),
            "sha256": checksum(alt["ae_logspec"]),
        },
        {
            "cache": "vib_logspec",
            "source": "Pass-mean vibration dB cache",
            "freq_transform": "None",
            "amp_transform": "Per-sample z-score of dB spectrogram",
            "normalization": "Per-sample/channel standardization after dB transform",
            "shape": str(tuple(alt["vib_logspec"].shape)),
            "sha256": checksum(alt["vib_logspec"]),
        },
        {
            "cache": "ae_mel",
            "source": "Pass-mean AE dB cache, inverse-converted to power",
            "freq_transform": "64-bin mel filter bank",
            "amp_transform": "Power -> mel sum -> dB",
            "normalization": "None in cache; no per-sample z-score",
            "shape": str(tuple(alt["ae_mel"].shape)),
            "sha256": checksum(alt["ae_mel"]),
        },
        {
            "cache": "vib_mel",
            "source": "Pass-mean vibration dB cache, inverse-converted to power",
            "freq_transform": "64-bin mel filter bank",
            "amp_transform": "Power -> mel sum -> dB",
            "normalization": "None in cache; no per-sample z-score",
            "shape": str(tuple(alt["vib_mel"].shape)),
            "sha256": checksum(alt["vib_mel"]),
        },
    ]

    df = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "cache_provenance.csv"
    df.to_csv(csv_path, index=False)

    definition_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Exact transformation definitions for the principal time--frequency representations. Internal cache identifiers are retained here only for provenance.}\n\\label{tab:supp-cache-definitions}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}l p{2.8cm} p{2.7cm} p{3.0cm} p{2.8cm} l@{}}",
        "\\toprule",
        "Cache & Source quantity & Frequency transform & Amplitude transform & Normalization & Shape \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [
            latex_escape(row["cache"]),
            latex_escape(row["source"]),
            latex_escape(row["freq_transform"]),
            latex_escape(row["amp_transform"]),
            latex_escape(row["normalization"]),
            latex_escape(row["shape"]),
        ]
        definition_lines.append(
            " & ".join(cells) + " \\\\"
        )
    definition_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ])

    checksum_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Compact cache checksum index. Prefixes are the first 12 hexadecimal characters of SHA-256 digests computed from contiguous cached-array bytes. Complete 64-character digests are retained in \\texttt{reports/evidence/tables/cache\\_provenance.csv} and the public artifact manifest.}",
        "\\label{tab:supp-cache-provenance}",
        "\\small",
        "\\begin{tabular}{@{}lll@{}}",
        "\\toprule",
        "Cache & Shape & SHA-256 prefix \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cache = latex_escape(row["cache"])
        checksum_lines.append(
            f"\\texttt{{{cache}}} & \\texttt{{{row['shape']}}} & "
            f"\\texttt{{{row['sha256'][:12]}}} \\\\"
        )
    checksum_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    tex_path = OVERLEAF_MAIN / "supp_cache_provenance.tex"
    tex_path.write_text("\n".join(definition_lines + checksum_lines))

    mean.close()
    alt.close()
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
