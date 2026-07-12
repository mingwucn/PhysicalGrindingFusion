#!/usr/bin/env python3
"""Enforce the publication-figure OOP and sizing contract.

The registry covers every Python-generated artifact currently included by the
main manuscript or supplementary material.  It deliberately checks source,
render metadata, and LaTeX inclusion together: checking only generated PNGs
would miss a later nonstandard source edit or a manuscript scale-down.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = ROOT / "reports" / "evidence" / "figure_metadata"

ARTIFACTS = {
    "submission_ranking_top20": ("scripts/generate_submission_report.py", "dense_ranking"),
    "submission_prediction_scatter_top3": ("scripts/generate_submission_report.py", "three_panel_row_shared"),
    "methods_signal_representations": ("scripts/generate_methods_signal_representations.py", "two_by_two"),
    "methods_logo_matrix": ("scripts/generate_methods_logo_matrix.py", "square"),
    "methods_xai_comparison": ("scripts/generate_methods_xai_figure.py", "vertical_triptych"),
    "methods_uncertainty_overview": ("scripts/generate_methods_uncertainty_figure.py", "top_span_two_bottom"),
    "methods_signal_representation_comparison": ("scripts/generate_signal_representation_bar_chart.py", "double"),
    "representation_controlled_comparison": ("scripts/generate_fixed_input_representation_figure.py", "double_tall"),
    "results_statistical_comparison_pairs": ("scripts/generate_results_figures.py", "double_tall"),
    "results_statistical_comparison_forest": ("scripts/generate_results_figures.py", "double_tall"),
    "results_interpretability_bandmass": ("scripts/generate_results_figures.py", "wide"),
    "results_uncertainty_calibration": ("scripts/generate_results_figures.py", "single"),
    "results_condition_error_bars": ("scripts/generate_results_figures.py", "wide"),
    "results_deployment_tradeoffs": ("scripts/generate_results_figures.py", "double"),
    "discussion_complexity_accuracy": ("scripts/generate_results_figures.py", "wide"),
    "top_models_fold_mae": ("scripts/review_required_analyses.py", "double"),
    "top_models_per_condition_mae": ("scripts/review_required_analyses.py", "double"),
    "supp_full_residual_diagnostics": ("scripts/generate_full_residual_diagnostics.py", "diagnostic_grid"),
    "supp_run_order": ("scripts/review_round2_supplementary.py", "double"),
    "supp_per_condition_mae": ("scripts/review_round2_supplementary.py", "double"),
    "supp_runorder_diagnostics": ("scripts/generate_runorder_diagnostics_figure.py", "vertical_quad"),
    "shap_rf_ae_logspec_vib_logspec": ("scripts/shap_rf_logspec.py", "two_panel_row"),
    "rf_dbz_oof_treeshap": ("scripts/generate_rf_oof_shap_figure.py", "two_panel_row"),
    "nested_rf_dbz_logmel_folds": ("scripts/generate_nested_rf_tail_figure.py", "double"),
    "shap_fold_stability": ("scripts/generate_shap_fold_stability_figure.py", "vertical_duo"),
    "shap_importance_LightGBMModel_ae_spec+vib_spec": ("scripts/generate_xai_composite_figure.py", "two_panel_row"),
    "mc_dropout_intervals_ResNetVibCNN_vib_spec": ("scripts/replot_mc_dropout_intervals.py", "double"),
    "mc_dropout_reliability": ("scripts/reliability_diagram.py", "two_panel_row"),
    "rf_conformal_intervals": ("scripts/review_rf_oob_figure.py", "double"),
    "latency_vs_accuracy": ("scripts/benchmark_latency_cached.py", "double"),
}


def tex_includes() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in (ROOT / "overleaf" / "main").glob("*.tex"):
        text = path.read_text(encoding="utf-8")
        for options, image in re.findall(r"\\includegraphics\[([^]]*)\]\{([^}]+)\}", text):
            result.setdefault(Path(image).stem, []).append(f"{path.relative_to(ROOT)}: [{options}]")
    return result


def main() -> int:
    errors: list[str] = []
    includes = tex_includes()
    checked_sources: set[str] = set()

    for artifact, (source_rel, profile_name) in ARTIFACTS.items():
        metadata_path = METADATA_DIR / f"{artifact}.json"
        if not metadata_path.exists():
            errors.append(f"Missing render metadata: {metadata_path.relative_to(ROOT)}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["profile"]["name"] != profile_name:
            errors.append(f"{artifact}: expected profile {profile_name}, found {metadata['profile']['name']}")
        if metadata.get("palette") != "PublicationPalette":
            errors.append(f"{artifact}: missing shared PublicationPalette provenance")

        source = ROOT / source_rel
        if source_rel not in checked_sources:
            checked_sources.add(source_rel)
            text = source.read_text(encoding="utf-8")
            if "MutableFigure" not in text and "PublicationFigure" not in text:
                errors.append(f"{source_rel}: final figure generator does not use the OOP framework")
            if "plt.subplots" in text:
                errors.append(f"{source_rel}: direct plt.subplots is forbidden in registered final figures")
            if "PublicationPlotter.savefig" in text:
                errors.append(f"{source_rel}: direct savefig is forbidden; use figure.save()")

        for include in includes.get(artifact, []):
            if re.search(r"width=0\.", include):
                errors.append(f"{artifact}: nonstandard LaTeX scale factor in {include}")

    if errors:
        print("Publication figure style validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Publication figure style validation passed for {len(ARTIFACTS)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
