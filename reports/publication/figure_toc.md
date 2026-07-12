# Figure TOC — AEI Manuscript

This table maps every figure in `reports/publication/manuscript_AEI_sectional.md` to the script, notebook, or command that produced it.  It is intended to make human fine-tuning and re-generation straightforward.

> **Overleaf deployment:** every methods illustration script now writes the canonical PNG to `reports/evidence/plots/methods/` and copies it to `overleaf/images/`.  The LaTeX source references figures by filename only (e.g., `\includegraphics{width=\textwidth}{methods_xai_comparison.png}`), so the files must live in `overleaf/images/` for Overleaf compilation.

| Fig. | Label | File path | Generator | Command / key call | Key parameters | Inputs / dependencies | Reproducibility notes |
|------|-------|-----------|-----------|--------------------|----------------|----------------------|-----------------------|
| 1 | `fig:workflow` | LaTeX/TikZ workflow schematic | `overleaf/tikz/workflow_schematic.tex` | Compile through the manuscript build | Shared `publication_styles.tex`; TikZ `\resizebox{\linewidth}{!}{...}` | Raw AE columns map plane 1 to narrowband and plane 2 to broadband | The schematic shows representative families; `scripts/generate_model_inventory.py` produces the complete 27-model, 38-input supplementary inventory. |
| 1a | `fig:preprocessing-pipeline` | Native TikZ in `overleaf/main/Methods.tex` | Compile through the manuscript build | dB cache, optional per-sample z-score, and implemented inverse-dB-to-power mel branch | `scripts/cache_alternative_representations.py` | The diagram follows the actual cache hierarchy: pass-mean dB is inverse-converted to power before mel filtering. |
| 2 | `fig:representations` | `reports/evidence/plots/methods/methods_signal_representations.png` | `scripts/generate_methods_signal_representations.py` | `python scripts/generate_methods_signal_representations.py` | Condition 10, sample 1; archived AE plane 2 and vibration Z-axis; STFT: AE `n_fft=598, hop=426`, vib `n_fft=512, hop=426`; mel panels use mel-filter-index coordinates | `data/intermediate/cached_specs/mean_specs.npz`, `data/intermediate/cached_specs/alternative_reps.npz` | AE plane physical/filter provenance is unresolved; output is copied to `overleaf/images/`. |
| 3 | `fig:taxonomy` | LaTeX/TikZ model taxonomy | `overleaf/tikz/model_taxonomy.tex` | Compile through the manuscript build | Top-down alternative learner-family bands; shared `publication_styles.tex` typography | None | Native TikZ; semantic nodes use the shared `\footnotesize` typography and palette. |
| 4 | `fig:logo` | LaTeX/TikZ LOGO schematic | `overleaf/tikz/logo_scheme.tex` | Compile through the manuscript build | Compact 2×8 condition grid; one representative test/validation split | None | Native TikZ at natural compact width; captions provide the full title. |
| 4a | `fig:resnetvibcnn-arch` | LaTeX/TikZ ResNetVibCNN architecture | `overleaf/tikz/resnetvibcnn_architecture.tex` | Compile through the manuscript build | Exact input and intermediate tensor dimensions; shared `publication_styles.tex` typography | `src/grinding_physic_fusion/models/architectures.py:ResNetVibCNN` | Native TikZ; the architecture schematic records the evaluated vibration-only forward path. |
| 5 | `fig:ranking` | `reports/evidence/plots/submission_ranking_top20.png` | `scripts/generate_submission_report.py` | `python scripts/generate_submission_report.py` | Historical canonical top-20 by `mae_mean`; OOP dense-ranking profile | `reports/evidence/tables/full_results_logo_only.csv`; per-fold predictions in `reports/evidence/predictions/` | Explicitly labelled historical canonical benchmark; current reproducible RF results are reported separately in the manuscript. |
| 6 | `fig:xai` | `reports/evidence/xai/shap_importance_LightGBMModel_ae_spec+vib_spec.png` | `scripts/generate_xai_composite_figure.py` | `python scripts/generate_xai_composite_figure.py` | Left panel uses LightGBM TreeSHAP CSV from `scripts/shap_spectrogram_baselines.py`; right panel uses Grad-CAM CSV from `scripts/gradcam_resnet_vib.py`; AE labels converted to MHz and vibration axis to kHz | `reports/evidence/xai/shap_importance_LightGBMModel_ae_spec+vib_spec.csv`; `reports/evidence/xai/gradcam_resnetvib_freq_importance.csv` | Composite manuscript Figure 19: LightGBM AE TreeSHAP plus ResNetVibCNN vibration Grad-CAM. |
| S35 | `fig:supp-shap-rf` | `reports/evidence/plots/results/rf_dbz_oof_treeshap.png` | `scripts/rf_oof_treeshap.py`; `scripts/generate_rf_oof_shap_figure.py` | Run the OOF analysis in tmux, then `python scripts/generate_rf_oof_shap_figure.py` | 16 current RF/dB-z refits; 200 trees, depth 8, seed 42; SHAP only on each held-out condition | `alternative_reps.npz`; sample-pooled, condition-balanced, Condition-7-excluded profiles, prediction, and fold-summary CSVs | Reproduces current OOF MAE 0.021009 and displays weighting sensitivity explicitly; historical cache identifiers appear only in source keys. |
| Current RF tail | `fig:nested-rf-tail` | `reports/evidence/plots/results/nested_rf_dbz_logmel_folds.png` | `scripts/generate_nested_rf_tail_figure.py` | `python scripts/generate_nested_rf_tail_figure.py` | Paired outer-fold MAE from matched current nested RF runs | `nested_logo_dbz_matched_folds.csv`; `nested_logo_logmel_matched_folds.csv` | Shows near-equal means but larger log-mel Condition 7 and maximum-fold error. |
| 7 | `fig:uncertainty` | `reports/evidence/uncertainty/mc_dropout_intervals_ResNetVibCNN_vib_spec.png` | `scripts/mc_dropout_generic.py`; `scripts/replot_mc_dropout_intervals.py` | Run `python scripts/mc_dropout_generic.py --model ResNetVibCNN --config vib_spec` for predictions, then `python scripts/replot_mc_dropout_intervals.py` for the publication-labelled figure | 50 MC-dropout passes; fold seed `42 + fold`; strict loading of all 16 repeat-0 checkpoints; deterministic prediction computed in the same run | Fold checkpoints `checkpoints/ResNetVibCNN_vib_spec_fold*_repeat0.pt`; Vib-dB cache | No retraining or checkpoint fallback; plot-only renderer avoids repeating inference and copies the final image to `overleaf/images/`. |
| 8 | `fig:reliability` | `reports/evidence/uncertainty/mc_dropout_reliability.png` | `scripts/reliability_diagram.py` | `python scripts/reliability_diagram.py` | `N_BINS = 5`; right panel compares MAE with nominal 95% interval half-width | `reports/evidence/uncertainty/mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv` | Full-LOGO canonical source; output is copied to `overleaf/images/`. |
| 9 | `fig:pareto` | `reports/evidence/plots/latency_vs_accuracy.png` | `scripts/benchmark_latency_cached.py` | `python scripts/benchmark_latency_cached.py` | Deployment candidates; pooled CPU feature-precomputed inference over all 16 canonical repeat-0 LOGO checkpoints; p50 point and p95 whisker | `reports/evidence/tables/full_results_logo_only.csv`; `reports/evidence/tables/edge_latency_benchmark_per_fold.csv`; checkpoints in `checkpoints/` | Aggregate latency and MAE refer to the same 16-fold fitted-model family. Output copied to `overleaf/images/`. |
| 10 | `fig:scatter-top3` | `reports/evidence/plots/submission_prediction_scatter_top3.png` | `scripts/generate_submission_report.py` | `python scripts/generate_submission_report.py` | Top 3 model configs; predicted vs observed $R_a$; `figsize=(8, 8)`; `dpi=300` | `reports/evidence/tables/full_results_logo_only.csv`; per-fold predictions in `reports/evidence/predictions/` | Produced together with `fig:ranking`. |
| 11 | `fig:statistical-pairs` | `reports/evidence/plots/results/results_statistical_comparison_pairs.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Nine horizontal pairwise comparisons; labels wrap as model A / `vs` / model B; mean MAE ± std; Holm significance stars | `reports/evidence/tables/full_results_logo_only.csv`, `wst_results.csv`, `statistical_tests_updated.csv` | Output copied to `overleaf/images/`. |
| 12 | `fig:statistical-forest` | `reports/evidence/plots/results/results_statistical_comparison_forest.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Median paired LOGO MAE differences with 95% bootstrap CIs; blue CI/orange median palette and three-line A / `vs` / B labels match Figure 11; significance from Holm-Bonferroni | `reports/evidence/tables/full_results_logo_only.csv`, `statistical_tests_updated.csv`, per-fold prediction CSVs for WST | Output copied to `overleaf/images/`. |
| 13 | `fig:representations-bar` | `reports/evidence/plots/methods/methods_signal_representation_comparison.png` | `scripts/generate_signal_representation_bar_chart.py` | `python scripts/generate_signal_representation_bar_chart.py` | Historical canonical best-observed mean LOGO MAE per representation family; error bars = std across 16 folds | `reports/evidence/tables/signal_representation_comparison.csv` | Output copied to `overleaf/images/`; current dB-z/log-mel reruns are reported separately in the manuscript. |
| 13b | `fig:representation-controlled` | `reports/evidence/plots/results/representation_controlled_comparison.png` | `scripts/generate_fixed_input_representation_figure.py` | `python scripts/fixed_input_representation_comparison.py && python scripts/generate_fixed_input_representation_figure.py` | Exact fixed-input matrix for RF/Ridge/LightGBM across six common input configurations; cells show mean ± fold SD across 16 seed-42 LOGO folds | `reports/evidence/tables/fixed_input_representation_comparison.csv` | The raw-target shallow MLP fits are excluded because they fail numerically; trajectory is absent because no common trajectory input exists for the three fixed learners. Output copied to `overleaf/images/`. |
| 14 | `fig:bandmass` | `reports/evidence/plots/results/results_interpretability_bandmass.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Three-row stacked chart: ResNetVibCNN Grad-CAM, LightGBM global TreeSHAP, and current RF OOF TreeSHAP | Table `tbl:bandmass` plus the separately labelled LightGBM global profile | Output copied to `overleaf/images/`; model dependence is explicit. |
| 15 | `fig:uncertainty-calibration` | `reports/evidence/plots/results/results_uncertainty_calibration.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Predicted standard deviation versus absolute error; covered/uncovered colour and 1σ/1.96σ reference lines | `reports/evidence/uncertainty/mc_dropout_ResNetVibCNN_vib_spec_logo_all.csv` | Cluster-aware correlation and LOCO sensitivity are generated by `scripts/clustered_uncertainty_diagnostics.py`; output copied to `overleaf/images/`. |
| 16 | `fig:condition-bars` | `reports/evidence/plots/results/results_condition_error_bars.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Mean MAE per grinding condition; Condition 7 highlighted | `reports/evidence/tables/condition_error_ranking.csv` | Output copied to `overleaf/images/`. |
| 17 | `fig:deployment-tradeoffs` | `reports/evidence/plots/results/results_deployment_tradeoffs.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Accuracy vs median latency bubble chart using exact Table `tbl:latency` values; bubble size = checkpoint size; colour = parameter count; numbered legend below plot | `reports/evidence/tables/edge_latency_benchmark.csv` | Output copied to `overleaf/images/`. |
| 18 | `fig:discussion-complexity` | `reports/evidence/plots/results/discussion_complexity_accuracy.png` | `scripts/generate_results_figures.py` | `python scripts/generate_results_figures.py` | Serialized checkpoint size (log scale) versus mean LOGO MAE; tree ensembles are not assigned a synthetic parameter count | `reports/evidence/tables/edge_latency_benchmark.csv` | Output copied to `overleaf/images/`. |
| 19 | `fig:condition7-condition10` | `reports/evidence/plots/results/condition7_condition10_spectra.png` | `scripts/condition7_condition10_diagnostic.py` | `python scripts/condition7_condition10_diagnostic.py` | Conditions 7 and 10; archived mean AE/vibration dB spectra | `data/intermediate/cached_specs/mean_specs.npz`; process parameters | Descriptive diagnostic only; does not identify a causal condition mechanism. |
| 25--26 | `fig:top-models-fold-mae`, `fig:top-models-per-condition` | `reports/evidence/plots/results/top_models_fold_mae.png`, `top_models_per_condition_mae.png` | `scripts/review_required_analyses.py` | `python scripts/review_required_analyses.py` | Historical top-five condition-level errors with publication-facing representation labels | Historical canonical fold arrays | Internal cache keys are translated to dB-z/log-mel labels in the generator. |
| 34 | `fig:supp-per-condition-mae` | `reports/evidence/plots/supp/supp_per_condition_mae.png` | `scripts/review_round2_supplementary.py` | `python scripts/review_round2_supplementary.py` | Three available per-condition historical series with publication-facing labels | Historical fold arrays for RF dB-z, ResNetVibCNN Vib-dB, and LightGBM Vib-dB-z | The unavailable Vib-WST fold curve is not claimed or drawn. |
| S | `fig:supp-full-residual-diagnostics` | `reports/evidence/plots/supp/supp_full_residual_diagnostics.png` | `scripts/generate_full_residual_diagnostics.py` | `python scripts/generate_full_residual_diagnostics.py` | Residuals versus measured/predicted \(R_a\), process parameters, run order, AE/Vib RMS, and target-support distance | `reports/evidence/predictions/RandomForestModel_ae_logspec_vib_logspec_fold*_repeat0.csv`, `supp_run_order.csv`, `supp_runorder_diagnostics.csv` | Output copied to `overleaf/images/`. |

## Script-to-figure quick reference

All figure generation now goes through the hard-coded `PublicationPlotter`
style in `src/grinding_physic_fusion/visualization/publication_plotter.py`.
Each script below produces both a publication PDF (editable master) and a
PNG copy in `overleaf/images/` for LaTeX compilation.

| Script | Figures produced | Notes |
|---|---|---|
| `scripts/generate_methods_signal_representations.py` | `methods_signal_representations.png` | 2×2 AE/vib spectrograms |
| `scripts/generate_methods_model_taxonomy.py` | `methods_model_taxonomy.png` | Transparent vs deep-learning taxonomy |
| `scripts/generate_methods_logo_scheme.py` | `methods_logo_scheme.png` | LOGO schematic |
| `scripts/generate_methods_logo_matrix.py` | `methods_logo_matrix.png`; `canonical_logo_assignments.csv` | Canonical seed-42 16×16 fold-assignment matrix and exact split table |
| `scripts/generate_methods_xai_figure.py` | `methods_xai_comparison.png` | Ridge/TreeSHAP/Grad-CAM comparison |
| `scripts/generate_methods_uncertainty_figure.py` | `methods_uncertainty_overview.png` | One full-width interval panel above two diagnostic panels |
| `scripts/generate_signal_representation_bar_chart.py` | `methods_signal_representation_comparison.png` | Best MAE per representation family |
| `scripts/generate_fixed_input_representation_figure.py` | `representation_controlled_comparison.png` | Fixed-input RF/Ridge/LightGBM representation heatmap |
| `scripts/generate_submission_report.py` | `submission_ranking_top20.png`, `submission_prediction_scatter_top3.png`, ... | Ranking and diagnostic plots |
| `scripts/generate_results_figures.py` | `results_statistical_comparison_*.png`, `results_interpretability_bandmass.png`, `results_uncertainty_calibration.png`, `results_condition_error_bars.png`, `results_deployment_tradeoffs.png`, `discussion_complexity_accuracy.png` | Results illustrations using shared OOP profiles and legend typography |
| `scripts/generate_full_residual_diagnostics.py` | `supp_full_residual_diagnostics.png` | Supplementary residual diagnostics |
| `scripts/benchmark_latency.py` | `latency_vs_accuracy.png` | Latency/accuracy trade-off |
| `scripts/shap_spectrogram_baselines.py` | `shap_importance_*_ae_spec+vib_spec.png` | Top SHAP features |
| `scripts/reliability_diagram.py` | `mc_dropout_reliability.png` | Reliability diagram and coverage bins |
| `scripts/uncertainty_best_model.py` | `mc_dropout_intervals_*.png`, `mc_dropout_calibration_*.png` | MC-dropout intervals and calibration |
| `scripts/mc_dropout_uncertainty.py` | `mc_dropout_calibration_ResNetVibCNN_vib_spec_fold0.png` | Per-fold calibration scatter |
| `scripts/gradcam_resnet_vib.py` | `gradcam_resnetvib_samples.png`, `gradcam_resnetvib_aggregate.png` | Grad-CAM overlays and aggregate importance |

## Quick regeneration checklist

Run the following commands from the repository root (`/cw/dtaiexp/2024-Ming/SmartGrinding/VibeGrinding`) in the `ai` conda environment:

```bash
# Methods illustrations
python scripts/generate_methods_signal_representations.py
python scripts/generate_methods_model_taxonomy.py
python scripts/generate_methods_logo_scheme.py
python scripts/generate_methods_xai_figure.py
python scripts/generate_methods_uncertainty_figure.py

# Ranking + calibration + residual + modality plots
python scripts/generate_submission_report.py

# SHAP / global explanations
python scripts/shap_spectrogram_baselines.py

# MC-dropout uncertainty (same checkpoints, deterministic and 50 seeded stochastic passes)
python scripts/mc_dropout_generic.py --model ResNetVibCNN --config vib_spec

# Latency vs. accuracy
python scripts/benchmark_latency.py

# Results-section illustrations
python scripts/generate_signal_representation_bar_chart.py
python scripts/generate_results_figures.py
python scripts/generate_full_residual_diagnostics.py

# Workflow schematic (run the notebook cell)
jupyter notebook notebooks/06_baselines/01_baseline_models.ipynb
```

## Methods illustrations — now scripted

The three methods illustrations are now generated by committed scripts:

```bash
python scripts/generate_methods_signal_representations.py
python scripts/generate_methods_model_taxonomy.py
python scripts/generate_methods_logo_scheme.py
```

When fine-tuning:

- Edit the corresponding script for layout, colours, or labels.
- Keep the same **output file names** so `manuscript_AEI_sectional.md` and `overleaf/main/Methods.tex` continue to reference them correctly.
- After regenerating, overwrite the existing files and commit:
  ```bash
  git add reports/evidence/plots/methods/ scripts/generate_methods_*.py
  git commit -m "Regenerate methods illustrations"
  ```

## Software environment

All scripts are tested in the `ai` conda environment:

```text
Python 3.11, NumPy 1.26, SciPy 1.14, scikit-learn 1.5,
LightGBM 4.5, XGBoost 2.1, PyTorch 2.4, Kymatio 0.3.0, SHAP 0.51.0
```
