"""Generate the self-contained 27-model, 38-input inventory for the supplement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reports/evidence/tables/full_results_logo_only.csv"
OUTPUT = ROOT / "overleaf/main/supp_model_inventory.tex"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("+", r" + ")


DISPLAY_TOKENS = {
    "ae_logspec": "AE-dB-z",
    "vib_logspec": "Vib-dB-z",
    "ae_mel": "AE-log-mel",
    "vib_mel": "Vib-log-mel",
    "ae_spec": "AE-dB",
    "vib_spec": "Vib-dB",
    "ae_features": "AE features",
    "vib_features": "Vib features",
    "ae_embed": "AE embedding",
    "vib_embed": "Vib embedding",
    "ae_trajectory": "AE trajectory",
    "vib_trajectory": "Vib trajectory",
    "physics": "physics features",
    "pp": "PP",
    "all": "all inputs",
}


def display_config(config: str) -> str:
    return " + ".join(DISPLAY_TOKENS.get(token, token) for token in config.split("+"))


def main() -> None:
    df = pd.read_csv(SOURCE)
    all_configs = sorted(df["config"].unique())
    config_ids = {config: f"C{index:02d}" for index, config in enumerate(all_configs, start=1)}
    rows = []
    for model, group in df.groupby("model", sort=True):
        configs = sorted(group["config"].unique())
        rows.append((model, len(configs), ", ".join(config_ids[config] for config in configs)))

    assert len(rows) == 27, f"Expected 27 model variants, found {len(rows)}"
    assert len(all_configs) == 38, f"Expected 38 inputs, found {len(all_configs)}"

    lines = [
        r"\begin{longtable}{@{}p{0.42\linewidth}cp{0.43\linewidth}@{}}",
        r"\caption{Complete LOGO-only model inventory. The archived canonical benchmark contains 27 model variants evaluated across 38 distinct input configurations.}\label{tab:supp-model-inventory}\\",
        r"\toprule",
        r"Model variant & Inputs ($n$) & Configuration IDs \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model variant & Inputs ($n$) & Configuration IDs \\",
        r"\midrule",
        r"\endhead",
    ]
    for model, count, configs in rows:
        lines.append(f"\\texttt{{{latex_escape(model)}}} & {count} & \\texttt{{{latex_escape(configs)}}} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", "", r"\begin{longtable}{@{}cp{0.82\linewidth}@{}}", r"\caption{Configuration key for Table~\ref{tab:supp-model-inventory}.}\\", r"\toprule", r"ID & Input configuration \\", r"\midrule", r"\endfirsthead", r"\toprule", r"ID & Input configuration \\", r"\midrule", r"\endhead"])
    for config in all_configs:
        lines.append(f"{config_ids[config]} & {display_config(config)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
