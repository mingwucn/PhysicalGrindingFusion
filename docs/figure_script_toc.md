# Figure / Table / Script Traceability

This document maps each analysis script to the figures, tables, and cached artefacts it produces, and notes where the output is referenced in the manuscript.

## Publication Figure Profiles

All final Python-generated figures use `PublicationFigure` or
`MutableFigure` from `src/grinding_physic_fusion/visualization/` and record
their resolved profile under `reports/evidence/figure_metadata/`.

| Profile | Physical export size | Intended use |
|---|---:|---|
| `single` | 89 mm wide | One compact axis or simple diagnostic |
| `wide` | 183 mm wide | Label-dense single panel using the Nature double-column width |
| `double` | 183 mm wide | Standard manuscript-wide panel |
| `dense_ranking` | 183 mm wide, up to 170 mm high | One readable 15--25 item ranking |
| `three_panel_row` | 183 mm wide | Three related panels, one shared legend/colourbar maximum |
| `four_panel_row` | 183 mm wide | Compact shared-axis small multiples only |
| `two_by_two` | 183 mm wide | Four independent panels when a 1×4 row is not readable |

The Top-20 LOGO ranking is produced by
`scripts/generate_submission_report.py` using the `dense_ranking` profile.

## Native TikZ Rules

All manuscript TikZ figures must load the shared
`overleaf/tikz/publication_styles.tex` layer through `overleaf/Manuscript.tex`.
Each diagram must live in its own `overleaf/tikz/<figure_name>.tex` source and
be included from the manuscript with `\\input{tikz/<figure_name>}`; inline
`tikzpicture` environments in `overleaf/main/*.tex` are forbidden.
Use `pub/input`, `pub/representation`, `pub/process`, `pub/physics`,
`pub/model`, `pub/explanation`, and `pub/target` for semantic nodes; use
`pub/arrow`, `pub/flow`, and `pub/dashed` for connectors. Do not introduce
figure-local palette definitions, arbitrary rounded-corner radii, or nested
scaling. The four canonical full-width schematics may use one outer
`\resizebox{\linewidth}{!}` because their geometry and type are designed and
visually verified as a unit; arbitrary local scaling remains forbidden.
Geometry must target the final 89 mm or 183 mm width, and a compact horizontal legend is permitted only when colour
carries meaning. Semantic nodes and headings use the shared 7 pt-equivalent
sans-serif style; legends and notes use the shared 5--6 pt-equivalent style.
Do not apply figure-local `font=` overrides to semantic nodes or headings.
Avoid scaling a small diagram up to `\linewidth`: use its natural compact
width instead. Diagram geometry may vary by task, but typography, strokes,
arrows, and semantic colours must remain shared across the paper.
Architecture pipelines should wrap into two rows before nodes are compressed
below the shared type size. Taxonomies should use the available line width
with balanced outer margins and at least one node-height of clearance around
`pub/note` text. Wide schematic legends should use compact swatches in two
rows instead of forcing the diagram beyond `\linewidth`.
Multi-panel TikZ figures use `pub/panel label` for 8 pt bold upright lowercase
letters at the top left. No prose subcaption belongs inside a panel; the main
LaTeX caption defines `\textbf{a}`, `\textbf{b}`, and subsequent panels.

These are hard constraints from the official Nature figure guidance:

- https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
- https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
- https://www.nature.com/nature/for-authors/final-submission
- https://www.nature.com/documents/natrev-figure-guidelines-v1.pdf

### TikZ Visual Verification

Native TikZ figures must be visually inspected after geometry changes. Run
`python scripts/render_tikz_previews.py` to compile the compact native
diagrams using the shared style layer and write local PDF/PNG previews to
`/tmp/vibegrinding-tikz-previews`. The command sends only the selected TikZ
source and shared style layer to the renderer; it does not send the manuscript,
raw data, or cached features. Large schematics may exceed the public renderer's
request-size limit and should be checked in the manuscript build instead.

### Python Figure Visual Verification

Before submission, run `python scripts/validate_publication_figures.py` and
`python scripts/render_publication_figure_audit.py`. The latter resolves the
PNG figures actually included by the manuscript and writes labelled contact
sheets to `/tmp/vibegrinding-python-figure-audit`; inspect the contact sheets
and any label-dense source figure at its native resolution. This is a visual
readability check, not a replacement for data validation. Keep long labels out
of plotting areas: use stable marker IDs with a compact key beneath a wide
plot when necessary. For non-negative metrics such as MAE, do not use
symmetric error bars that extend below zero.

## Data caches

| Script | Output | Used by |
|--------|--------|---------|
| `scripts/cache_mean_spectrograms.py` | `data/intermediate/cached_specs/mean_specs.npz` | Most downstream scripts |
| `scripts/cache_alternative_representations.py` | `data/intermediate/cached_specs/alternative_reps.npz` | RF / DL training, ablations |
| `scripts/generate_cache_provenance.py` | `reports/evidence/tables/cache_provenance.csv`; `overleaf/main/supp_cache_provenance.tex` | Cache-definition and checksum provenance table |

## Main results figures and tables

| Script | Output | Manuscript reference |
|--------|--------|----------------------|
| `scripts/review_required_analyses.py` | `reports/evidence/tables/full_results_logo_only.csv` | Tables 5, 6, 7 and statistical tests |
| `scripts/review_required_analyses.py` | `reports/evidence/tables/top_models_summary.csv` | Table 6 |
| `scripts/review_required_analyses.py` | `reports/evidence/tables/top_models_per_condition_mae.csv` | Figure 10, Supplementary tables |
| `scripts/review_required_analyses.py` | `reports/evidence/tables/representation_comparison_controlled.csv` | Table `tbl:controlled-representations` |
| `scripts/review_required_analyses.py` | `reports/evidence/tables/condition7_sensitivity_top_pairs.csv` | Section 4.6 |
| `scripts/review_required_analyses.py` | `reports/evidence/plots/results/top_models_fold_mae.png/.pdf` | `fig:top-models-fold-mae`; publication labels replace internal cache keys |
| `scripts/review_required_analyses.py` | `reports/evidence/plots/results/top_models_per_condition_mae.png/.pdf` | `fig:top-models-per-condition`; publication labels replace internal cache keys |
| `scripts/fixed_input_representation_comparison.py`; `scripts/generate_fixed_input_representation_figure.py` | `reports/evidence/plots/results/representation_controlled_comparison.png/.pdf` | `fig:representation-controlled` |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/results_statistical_comparison_pairs.png/.pdf` | `fig:statistical-pairs`; horizontal bars with model A / `vs` / model B wrapped labels |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/results_statistical_comparison_forest.png/.pdf` | `fig:statistical-forest`; blue CI/orange median palette and A / `vs` / B wrapped labels match `fig:statistical-pairs` |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/results_interpretability_bandmass.png/.pdf` | `fig:bandmass`; RF OOF TreeSHAP, LightGBM global TreeSHAP, and ResNetVibCNN Grad-CAM are separate rows |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/results_uncertainty_calibration.png/.pdf` | `fig:uncertainty-calibration`; Nature single-column profile, included at 89 mm |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/results_condition_error_bars.png/.pdf` | `fig:condition-error-bars` |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/results_deployment_tradeoffs.png/.pdf` | `fig:deployment-tradeoffs` |
| `scripts/generate_results_figures.py` | `reports/evidence/plots/results/discussion_complexity_accuracy.png/.pdf` | Discussion |
| `scripts/condition_error_analysis.py` | `reports/evidence/plots/results/results_condition_error_bars.png` | Figure 13 (legacy path) |

## Uncertainty quantification figures and tables

| Script | Output | Manuscript reference |
|--------|--------|----------------------|
| `scripts/mc_dropout_generic.py` | `reports/evidence/uncertainty/mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv` | Section 4.5, same-checkpoint deterministic/MC predictions, Supplementary Tables `tab:supp-mcdropout-full-logo` and `tab:supp-mcdropout-deterministic` |
| `scripts/generate_xai_composite_figure.py` | `reports/evidence/xai/shap_importance_LightGBMModel_ae_spec+vib_spec.png` | Figure `fig:xai` |
| `scripts/mc_dropout_generic.py` | `reports/evidence/uncertainty/mc_dropout_summary_ResNetVibCNN_vib_spec_logo_all.csv` | Section 4.5; 50 stochastic passes, fold seed `42 + fold`, strict checkpoint loading |
| `scripts/mc_dropout_generic.py` | `reports/evidence/uncertainty/mc_dropout_per_condition_ResNetVibCNN_vib_spec_logo_all.csv` | Supplementary Table `tab:supp-mcdropout-full-logo` |
| `scripts/mc_dropout_generic.py`; `scripts/replot_mc_dropout_intervals.py` | `reports/evidence/uncertainty/mc_dropout_intervals_ResNetVibCNN_vib_spec.png/.pdf` | Figure 12, Methods Figure 5 panels; the first script creates the seeded same-checkpoint data and the second applies publication labels without rerunning inference |
| `scripts/generate_review19_diagnostics.py` | `mc_dropout_deterministic_diagnostic.csv`; `supp_mcdropout_deterministic.tex`; `supp_nested_rf_selections.tex` | Same-checkpoint MC diagnostic and per-fold nested RF selections |
| `scripts/reliability_diagram.py` | `reports/evidence/uncertainty/mc_dropout_reliability.csv` | Figure 12, Methods Figure 5 |
| `scripts/reliability_diagram.py` | `reports/evidence/uncertainty/mc_dropout_reliability.png/.pdf` | Figure 12, Methods Figure 5; bold `a`/`b` labels and one shared legend below both panels |
| `scripts/condition7_condition10_diagnostic.py` | `reports/evidence/plots/results/condition7_condition10_spectra.png/.pdf` | Condition 7/10 diagnostic; bold `a`/`b` labels and one shared legend below both panels |
| `scripts/generate_methods_uncertainty_figure.py` | `reports/evidence/plots/methods/methods_uncertainty_overview.png/.pdf` | Figure 5; one full-width panel above two diagnostic panels |
| `scripts/generate_rf_conformal_intervals.py` | `reports/evidence/uncertainty/rf_conformal_intervals.png/.pdf` | Figure 11 |

## Ablations and sensitivity analyses

| Script | Output | Manuscript reference |
|--------|--------|----------------------|
| `scripts/ablation_no_psnorm.py` | `reports/evidence/tables/ablation_per_sample_normalisation.csv` | Section 3.2.1, Supplementary Table `tab:supp-psnorm-ablation` |
| `scripts/ablation_no_psnorm.py` | `reports/evidence/tables/ablation_per_sample_normalisation_per_condition.csv` | Supplementary |
| `scripts/run_cv_scheme_comparison_only.py` | `reports/evidence/tables/cv_scheme_comparison.csv` | Table `tab:cv-schemes` |
| `scripts/run_wheel_speed_logo_only.py` | wheel-speed row in `reports/evidence/tables/cv_scheme_comparison.csv` | Table `tab:cv-schemes` |
| `scripts/comprehensive_sensitivity_suite.py` | `reports/evidence/tables/rf_modality_ablation.csv` | Table 11 (`tab:modality-ablation`) |
| `scripts/comprehensive_sensitivity_suite.py` | `reports/evidence/tables/rf_frequency_ablation.csv` | Table 12 (`tab:frequency-ablation`) |
| `scripts/comprehensive_sensitivity_suite.py` | `reports/evidence/tables/rf_calibration_strategies.csv` | Table 13 (`tab:calibration-strategies`) |
| `scripts/comprehensive_sensitivity_suite.py` | `reports/evidence/tables/sensitivity_summary.md` | This TOC / project record |
| `scripts/nested_logo_cv.py` | `reports/evidence/tables/nested_logo_results.csv`; `nested_logo_results_folds.csv` | Supplementary nested grouped sensitivity table and per-fold selected hyperparameters |
| `scripts/nested_logo_cv.py`; `scripts/merge_nested_logo_shards.py` | `reports/evidence/tables/nested_logo_dbz_matched.csv`; `nested_logo_dbz_matched_folds.csv`; `nested_logo_logmel_matched.csv`; `nested_logo_logmel_matched_folds.csv` | Cache-matched nested dB-z/log-mel reruns reported beside the legacy nested log-mel artifact |
| `scripts/pass_position_sensitivity.py` | `reports/evidence/tables/pass_position_sensitivity.csv`; `pass_position_sensitivity_folds.csv` | Supplementary pass-position and ordered multi-window sensitivity table |
| `scripts/generate_full_residual_diagnostics.py` | `reports/evidence/tables/rf_full_residual_diagnostics.csv` | Supplementary Figure `fig:supp-full-residual-diagnostics` |
| `scripts/generate_full_residual_diagnostics.py` | `reports/evidence/plots/supp/supp_full_residual_diagnostics.png/.pdf` | Supplementary Figure `fig:supp-full-residual-diagnostics` |
| `scripts/generate_shap_fold_stability_figure.py` | `reports/evidence/plots/results/shap_fold_stability.png/.pdf` | Supplementary Figure `fig:supp-shap-stability` |
| `scripts/rf_oof_treeshap.py`; `scripts/generate_rf_oof_shap_figure.py` | `reports/evidence/plots/results/rf_dbz_oof_treeshap.png/.pdf`; pooled, condition-balanced, Condition-7-excluded OOF profiles, predictions, and fold summary CSVs | Main Figure `fig:rf-oof-shap`; primary explanation of the current RF; 16 refits explain held-out conditions only |
| `scripts/generate_nested_rf_tail_figure.py` | `reports/evidence/plots/results/nested_rf_dbz_logmel_folds.png/.pdf` | Figure `fig:nested-rf-tail`; paired current nested dB-z/log-mel outer-fold errors and Condition 7 tail difference |
| `scripts/shap_rf_logspec.py` | `reports/evidence/plots/results/shap_rf_ae_logspec_vib_logspec.png/.pdf`; per-frequency CSVs | Secondary global RF explanatory refit retained for provenance |
| `scripts/rf_condition_aware_conformal.py` | `reports/evidence/tables/rf_conformal_global.csv` | Table 13 / Section 4.5.4 |
| `scripts/rf_condition_aware_conformal.py` | `reports/evidence/tables/rf_conformal_condition_aware.csv` | Table 13 / Section 4.5.4 |
| `scripts/rf_parameter_group_logo.py` | `reports/evidence/tables/rf_parameter_group_logo.csv` | Table 10 (`tab:cv-schemes`) |
| `scripts/generate_methods_logo_matrix.py` | `methods_logo_matrix.png`; `canonical_logo_assignments.csv` | Canonical seed-42 16-by-16 fold-assignment matrix and exact split table |
| `scripts/ablation_no_psnorm.py` | `reports/evidence/tables/ablation_per_sample_normalisation*.csv` | Supplementary Tables `tab:supp-psnorm-ablation` and `tab:supp-rf-provenance` |
| `scripts/nested_logo_cv.py` + `scripts/merge_nested_logo_shards.py` | `reports/evidence/tables/nested_logo_{dbz,logmel}_matched*.csv` | Supplementary Tables `tab:supp-nested-logo` and `tab:supp-rf-provenance` |

## Latency and deployment

| Script | Output | Manuscript reference |
|--------|--------|----------------------|
| `scripts/benchmark_latency_cached.py` | `reports/evidence/tables/edge_latency_benchmark.csv`; `reports/evidence/tables/edge_latency_benchmark_per_fold.csv` | Table `tbl:latency`; pooled 16-checkpoint timing and fold-level checkpoint/hash manifest |
| `scripts/benchmark_latency_cached.py` | `reports/evidence/plots/latency_vs_accuracy.png/.pdf` | Figure `fig:pareto` |

## Training and evaluation

| Script | Output | Notes |
|--------|--------|-------|
| `scripts/train_and_evaluate.py` | `checkpoints/*.pt` and per-fold CSVs | Trains all model/configuration checkpoints used by downstream analyses |
| `scripts/run_statistical_tests.py` | Holm-corrected comparison tables | Section 4.2 |

## How to regenerate a figure/table

Most scripts are self-contained and use the cached spec files. A typical regeneration order is:

1. `scripts/cache_mean_spectrograms.py` (if raw data change)
2. `scripts/cache_alternative_representations.py` (if preprocessing changes)
3. `scripts/train_and_evaluate.py` (if models or hyperparameters change)
4. `scripts/review_required_analyses.py` (main results tables/figures)
5. `scripts/generate_results_figures.py` (main results figures)
6. `scripts/mc_dropout_full_logo.py` + `scripts/reliability_diagram.py` + `scripts/generate_methods_uncertainty_figure.py` (uncertainty figures)
7. `scripts/comprehensive_sensitivity_suite.py` (ablation/sensitivity tables)
8. `scripts/generate_full_residual_diagnostics.py` (supplementary residual diagnostic grid)
9. `scripts/benchmark_latency.py` (latency table/figure)
