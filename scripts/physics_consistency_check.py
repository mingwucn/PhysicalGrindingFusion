"""Compare model-importance frequency bins to physically expected grinding bands."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("reports/evidence/xai")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling parameters inferred from cache_alternative_representations.py
AE_SR = 4_000_000.0
AE_NFFT = 598
VIB_SR = 51_200.0
VIB_NFFT = 512


def bin_to_hz(bin_idx: int, sr: float, n_fft: int) -> float:
    return bin_idx * sr / n_fft


def load_shap_freq_importance(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Aggregate importance over channels and time per (modality, freq_bin)
    agg = df.groupby(["modality", "freq_bin"])["importance"].sum().reset_index()
    return agg


def summarise(df: pd.DataFrame, modality: str, sr: float, n_fft: int) -> pd.DataFrame:
    sub = df[df["modality"] == modality].copy()
    sub["hz"] = sub["freq_bin"].apply(lambda b: bin_to_hz(b, sr, n_fft))
    total = sub["importance"].sum()
    sub["mass_fraction"] = sub["importance"] / total if total > 0 else 0
    return sub.sort_values("importance", ascending=False)


def expected_bands():
    """Return heuristic grinding frequency bands (Hz)."""
    return {
        "forced vibration (< 500 Hz)": (0, 500),
        "low-frequency chatter (500–2 kHz)": (500, 2000),
        "chatter / structural (2–15 kHz)": (2000, 15000),
        "high-frequency / AE (> 15 kHz)": (15000, np.inf),
    }


def band_mass(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = df["importance"].sum()
    for name, (lo, hi) in expected_bands().items():
        mask = (df["hz"] >= lo) & (df["hz"] < hi)
        mass = df.loc[mask, "importance"].sum()
        rows.append({
            "band": name,
            "mass": float(mass),
            "fraction": float(mass / total) if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    # Vibration Grad-CAM
    gradcam_csv = OUT_DIR / "gradcam_resnetvib_freq_importance.csv"
    gradcam = pd.read_csv(gradcam_csv).rename(columns={"mean_importance": "importance"})
    gradcam["hz"] = gradcam["freq_bin"].apply(lambda b: bin_to_hz(b, VIB_SR, VIB_NFFT))
    gradcam_bands = band_mass(gradcam)

    # SHAP for tree/linear baseline on ae_spec+vib_spec
    shap_csv = OUT_DIR / "shap_importance_LightGBMModel_ae_spec+vib_spec.csv"
    shap_df = load_shap_freq_importance(shap_csv)
    shap_vib = summarise(shap_df, "vib_spec", VIB_SR, VIB_NFFT)
    shap_ae = summarise(shap_df, "ae_spec", AE_SR, AE_NFFT)
    shap_vib_bands = band_mass(shap_vib)
    shap_ae_bands = band_mass(shap_ae)

    # Save per-modality top-bin tables
    top_n = 20
    shap_vib.head(top_n).to_csv(OUT_DIR / "physics_shap_vib_top_bins.csv", index=False)
    shap_ae.head(top_n).to_csv(OUT_DIR / "physics_shap_ae_top_bins.csv", index=False)
    gradcam.sort_values("importance", ascending=False).head(top_n).to_csv(
        OUT_DIR / "physics_gradcam_vib_top_bins.csv", index=False
    )

    # Plot Grad-CAM importance vs Hz
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gradcam["hz"], gradcam["importance"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean |Grad-CAM|")
    ax.set_title("ResNetVibCNN frequency importance")
    ax.set_xlim(0, 20000)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "physics_gradcam_vib_hz.png", dpi=150)
    plt.close(fig)

    # Write report
    report_path = OUT_DIR / "physics_consistency_report.md"
    with open(report_path, "w") as f:
        f.write("# Physics-consistency check of important frequency bins\n\n")
        f.write("## Assumptions\n\n")
        f.write(f"- Vibration sampling rate: {VIB_SR/1e3:.1f} kHz, n_fft={VIB_NFFT} → resolution {VIB_SR/VIB_NFFT:.1f} Hz/bin\n")
        f.write(f"- AE sampling rate: {AE_SR/1e6:.1f} MHz, n_fft={AE_NFFT} → resolution {AE_SR/AE_NFFT:.1f} Hz/bin\n\n")

        f.write("## Grad-CAM (ResNetVibCNN, vib_spec)\n\n")
        f.write("### Importance mass by frequency band\n\n")
        f.write(gradcam_bands.to_markdown(index=False))
        f.write("\n\n### Top 10 frequency bins\n\n")
        top_gc = gradcam.sort_values("importance", ascending=False).head(10)
        f.write(top_gc[["freq_bin", "hz", "importance"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## SHAP (LightGBM, vib_spec)\n\n")
        f.write("### Importance mass by frequency band\n\n")
        f.write(shap_vib_bands.to_markdown(index=False))
        f.write("\n\n### Top 10 frequency bins\n\n")
        f.write(shap_vib.head(10)[["freq_bin", "hz", "importance", "mass_fraction"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## SHAP (LightGBM, ae_spec)\n\n")
        f.write("### Importance mass by frequency band\n\n")
        f.write(shap_ae_bands.to_markdown(index=False))
        f.write("\n\n### Top 10 frequency bins\n\n")
        f.write(shap_ae.head(10)[["freq_bin", "hz", "importance", "mass_fraction"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write("If the dominant importance concentrates in the chatter/structural band (2–15 kHz) for vibration, ")
        f.write("the model is relying on physically plausible dynamic-response frequencies rather than random noise. ")
        f.write("AE importance should be concentrated at high frequencies (>20 kHz), consistent with acoustic emission.\n")
    print(f"Saved {report_path}")

if __name__ == "__main__":
    main()
