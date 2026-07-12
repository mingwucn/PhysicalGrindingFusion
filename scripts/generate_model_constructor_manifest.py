#!/usr/bin/env python3
"""Generate the complete canonical model-constructor manifest and LaTeX table."""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grinding_physic_fusion.models.architectures import MODEL_REGISTRY

INVENTORY = ROOT / "overleaf" / "main" / "supp_model_inventory.tex"
JSON_OUT = ROOT / "reports" / "evidence" / "tables" / "canonical_model_constructor_defaults.json"
RESOLVED_JSON_OUT = ROOT / "reports" / "evidence" / "tables" / "resolved_model_configuration_pairs.json"
TEX_OUT = ROOT / "overleaf" / "main" / "supp_model_constructor_defaults.tex"

ALIASES = {
    "SSLResNetVibCNN": "SelfSupervisedPretrainedCNN",
}


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def main() -> int:
    text = INVENTORY.read_text()
    names = re.findall(r"\\texttt\{([^}]+)\}\s*&\s*\d+\s*&", text)
    inventory_rows = {}
    for line in text.splitlines():
        match = re.match(r"\\texttt\{([^}]+)\}\s*&\s*\d+\s*&", line)
        if match:
            inventory_rows[match.group(1)] = re.findall(r"C\d+", line)
    config_labels = {
        key: label.strip()
        for key, label in re.findall(r"^(C\d+)\s*&\s*(.*?)\s*\\\\$", text, flags=re.MULTILINE)
    }
    records = []
    for published_name in names:
        registry_name = ALIASES.get(published_name, published_name)
        constructor = MODEL_REGISTRY.get(registry_name)
        if constructor is None:
            raise KeyError(f"No model registry entry for {published_name}")
        signature = str(inspect.signature(constructor))
        records.append(
            {
                "published_model": published_name,
                "registry_model": registry_name,
                "constructor_defaults": signature,
                "configuration_ids": inventory_rows.get(published_name, []),
            }
        )

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "common_neural_training": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "loss": "MSE",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience_epochs": 5,
            "early_stopping_patience_epochs": 20,
            "maximum_epochs": 200,
            "batch_size_resnet": 16,
            "batch_size_other": 32,
            "canonical_seed": 42,
        },
        "models": records,
    }
    JSON_OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    resolved_pairs = []
    for record in records:
        model = record["published_model"]
        for config_id in inventory_rows.get(model, []):
            label = config_labels.get(config_id, "unresolved input ID")
            note = "constructor defaults used by the archived runner"
            if model == "MultiscaleSpectrogramCNN" and config_id == "C35":
                note = "Vib-dB requires effective in_channels=3; the archived override was not retained, so this artifact is not claimed as exactly reconstructable"
            elif model == "SSLResNetVibCNN" and config_id == "C35":
                note = "Vib-dB requires effective modality='vib'; the archived override was not retained, so this artifact is not claimed as exactly reconstructable"
            elif model == "ShallowMLPModel":
                note = "historical baseline used internal validation_fraction=0.1; corrected grouped-validation sensitivity uses the designated condition, train-only input/target scaling, and no internal split"
            resolved_pairs.append({
                "model": model,
                "configuration_id": config_id,
                "effective_input": label,
                "constructor_defaults": record["constructor_defaults"],
                "resolution_note": note,
            })
    RESOLVED_JSON_OUT.write_text(json.dumps(resolved_pairs, indent=2) + "\n")

    def resolution_summary(record: dict) -> str:
        model = record["published_model"]
        ids = ", ".join(record["configuration_ids"])
        if model == "MultiscaleSpectrogramCNN":
            return f"{ids}; Vib-dB requires 3 channels; archived override unavailable"
        if model == "SSLResNetVibCNN":
            return f"{ids}; Vib-dB requires modality=vib; archived override unavailable"
        if model == "ShallowMLPModel":
            return f"{ids}; historical internal 10% validation; corrected run uses external condition and target scaling"
        return f"{ids}; constructor defaults used"

    rows = "\n".join(
        f"\\texttt{{{latex_escape(r['published_model'])}}} & "
        f"\\nolinkurl{{{r['constructor_defaults']}}} & "
        f"{latex_escape(resolution_summary(r))} \\\\"
        for r in records
    )
    TEX_OUT.write_text(
        "\\begingroup\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{longtable}{@{}p{0.18\\linewidth}p{0.47\\linewidth}p{0.29\\linewidth}@{}}\n"
        "\\caption{Constructor defaults and resolved benchmark-use caveats for all 27 model variants. "
        "Input configuration IDs are given in Table~\\ref{tab:supp-model-inventory}; "
        "the machine-readable pair-level record is "
        "\\texttt{resolved\\_model\\_configuration\\_pairs.json}. The archived "
        "MultiscaleSpectrogramCNN and SSLResNetVibCNN entries require Vib-dB channel/modality "
        "overrides that were not retained and are therefore not claimed as exactly reconstructable. "
        "The historical shallow MLP used an internal random 10\\% validation split; the corrected "
        "grouped-validation sensitivity is reported separately.}\n"
        "\\label{tab:supp-model-constructors}\\\\\n"
        "\\toprule\nModel & Constructor arguments and defaults & Resolved benchmark use \\\\\n\\midrule\n"
        "\\endfirsthead\n\\toprule\nModel & Constructor arguments and defaults & Resolved benchmark use \\\\\n\\midrule\n\\endhead\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{longtable}\n\\endgroup\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {RESOLVED_JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {TEX_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
