#!/usr/bin/env python3
"""Generate machine-readable model manifests and a concise LaTeX inventory."""
from __future__ import annotations

import hashlib
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
ALIASES = {"SSLResNetVibCNN": "SelfSupervisedPretrainedCNN"}
ARCHIVAL = {"MultiscaleSpectrogramCNN", "SSLResNetVibCNN"}
PUBLIC_ARTIFACT_COMMIT = "7446310571510fd615da7a86a0fb31ebe0ffd31d"


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


def key_defaults(registry_name: str) -> str:
    parts = []
    for name, parameter in inspect.signature(MODEL_REGISTRY[registry_name]).parameters.items():
        if name in {"config", "kwargs"}:
            continue
        value = name if parameter.default is inspect.Parameter.empty else f"{name}={parameter.default!r}"
        parts.append(value)
        if len(parts) == 4:
            break
    return ", ".join(parts) if parts else "No exposed architecture arguments"


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
        records.append({
            "published_model": published_name,
            "registry_model": registry_name,
            "constructor_defaults": str(inspect.signature(constructor)),
            "configuration_ids": inventory_rows.get(published_name, []),
        })

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "common_neural_training": {
            "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
            "loss": "MSE", "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5,
            "scheduler_patience_epochs": 5, "early_stopping_patience_epochs": 20,
            "maximum_epochs": 200, "batch_size_resnet": 16, "batch_size_other": 32,
            "canonical_seed": 42,
        },
        "models": records,
    }
    JSON_OUT.write_text(json.dumps(manifest, indent=2) + "\n")

    resolved_pairs = []
    for record in records:
        model = record["published_model"]
        for config_id in inventory_rows.get(model, []):
            note = "constructor defaults used by the archived runner"
            if model == "MultiscaleSpectrogramCNN" and config_id == "C35":
                note = "Vib-dB requires effective in_channels=3; archived override unavailable; not exactly reconstructable"
            elif model == "SSLResNetVibCNN" and config_id == "C35":
                note = "Vib-dB requires effective modality='vib'; archived override unavailable; not exactly reconstructable"
            elif model == "ShallowMLPModel":
                note = "historical baseline used internal validation_fraction=0.1; corrected sensitivity uses designated condition, train-only input/target scaling, and no internal split"
            resolved_pairs.append({
                "model": model,
                "configuration_id": config_id,
                "effective_input": config_labels.get(config_id, "unresolved input ID"),
                "constructor_defaults": record["constructor_defaults"],
                "resolution_note": note,
            })
    RESOLVED_JSON_OUT.write_text(json.dumps(resolved_pairs, indent=2) + "\n")
    resolved_sha256 = hashlib.sha256(RESOLVED_JSON_OUT.read_bytes()).hexdigest()

    def input_summary(record: dict) -> str:
        ids = record["configuration_ids"]
        if len(ids) == 1:
            return config_labels.get(ids[0], ids[0])
        return f"{len(ids)} configurations"

    rows = "\n".join(
        f"\\texttt{{{latex_escape(record['published_model'])}}} & "
        f"{latex_escape(input_summary(record))} & "
        f"{latex_escape(key_defaults(record['registry_model']))} & "
        f"{'Archival' if record['published_model'] in ARCHIVAL else 'Reconstructable'} \\\\"
        for record in records
    )
    TEX_OUT.write_text(
        "\\begingroup\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{longtable}{@{}p{0.23\\linewidth}p{0.20\\linewidth}p{0.39\\linewidth}p{0.12\\linewidth}@{}}\n"
        "\\caption{Concise resolved inventory of 25 reconstructable variants and two archival records. "
        "Input IDs are defined in Table~\\ref{tab:supp-model-inventory}. Complete signatures and pair-level notes are in "
        "\\texttt{canonical\\_model\\_constructor\\_defaults.json} and "
        "\\texttt{resolved\\_model\\_configuration\\_pairs.json} (SHA-256 prefix "
        f"\\texttt{{{resolved_sha256[:12]}}}) at public artifact commit "
        f"\\texttt{{{PUBLIC_ARTIFACT_COMMIT[:12]}}}. Complete identifiers are retained in the public manifest. The two archival entries require Vib-dB channel or modality "
        "overrides that were not retained. The historical shallow MLP used an internal random 10\\% "
        "validation split; the corrected grouped-validation sensitivity is reported separately.}\n"
        "\\label{tab:supp-model-constructors}\\\\\n"
        "\\toprule\nModel & Configured input & Key exposed defaults & Status \\\\\n\\midrule\n"
        "\\endfirsthead\n\\toprule\nModel & Configured input & Key exposed defaults & Status \\\\\n\\midrule\n\\endhead\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{longtable}\n\\endgroup\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {RESOLVED_JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {TEX_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
