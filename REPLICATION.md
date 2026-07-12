# Replication Guide - PhysicalGrindingFusion

The immutable public review snapshot is the Git tag `review-v1`. The exact
319-sample evaluation target is stored in
`reports/evidence/tables/final_evaluation_targets.csv`; its SHA-256 digest and
the excluded sample identifier are recorded in
`reports/evidence/tables/final_evaluation_targets_manifest.json`.

This document describes how to reproduce the grouped benchmark, XAI analysis,
uncertainty diagnostics, robustness sensitivities, and submission figures.

## Software environment

```bash
conda env create -f environment.yml
conda activate ai
```

Key dependencies include PyTorch, scikit-learn, LightGBM, XGBoost,
Kymatio 0.3.0, SHAP, SciPy, pandas, and Matplotlib.

## Data

Raw data are expected under `data/`:

* `data/AE/`: acoustic-emission recordings.
* `data/Vibration/`: vibration recordings.
* `data/parameters.xlsx`: process parameters.
* `data/surface roughness.csv`: averaged measured Ra targets.

Preprocessed features are cached in `data/intermediate/cached_specs/`. These
large artifacts are not mirrored in Git. Research-use access to available raw
recordings and caches may be requested from the corresponding author.

## Reproduction steps

1. **Cache signal representations**
   ```bash
   python scripts/cache_alternative_representations.py
   python scripts/cache_wst_features.py
   ```

2. **Train models with 16-fold LOGO**
   ```bash
   python scripts/train_and_evaluate.py --models RandomForestModel --configs ae_logspec+vib_logspec --cv_folds 16
   python scripts/train_and_evaluate.py --models LightGBMModel --configs vib_wst --cv_folds 16
   python scripts/train_and_evaluate.py --models ResNetVibCNN --configs vib_spec --cv_folds 16
   ```

3. **Run matched current nested RF sensitivities**
   ```bash
   python scripts/nested_logo_cv.py
   ```

4. **Regenerate the statistical family**
   ```bash
   python scripts/run_statistical_tests_updated.py
   ```

5. **Run corrected methodological sensitivities**
   ```bash
   python scripts/grouped_validation_shallow_mlp.py
   python scripts/pre_normalization_band_ablation.py
   python scripts/retrained_frequency_band_ablation.py
   python scripts/rf_oof_treeshap.py
   ```

6. **Generate XAI and physics-consistency evidence**
   ```bash
   python scripts/shap_spectrogram_baselines.py
   python scripts/gradcam_resnet_vib.py
   python scripts/physics_consistency_check.py
   python scripts/generate_rf_oof_shap_figure.py
   ```

7. **Generate uncertainty diagnostics**
   ```bash
   python scripts/mc_dropout_uncertainty.py
   python scripts/reliability_diagram.py
   ```

8. **Run condition-level failure analysis**
   ```bash
   python scripts/condition_error_analysis.py
   ```

9. **Run the latency benchmark**
   ```bash
   python scripts/benchmark_latency_cached.py
   ```

10. **Generate publication figures and reports**
    ```bash
    python scripts/generate_submission_report.py
    python scripts/generate_findings_summary.py
    python scripts/validate_publication_figures.py
    ```

## Key outputs

| File | Description |
|---|---|
| `reports/evidence/tables/full_results_logo_only.csv` | Historical canonical per-fold LOGO results |
| `reports/evidence/tables/nested_logo_dbz_matched.csv` | Current matched nested dB-z RF summary |
| `reports/evidence/tables/nested_logo_logmel_matched.csv` | Current matched nested log-mel RF summary |
| `reports/evidence/tables/statistical_tests_updated.csv` | Wilcoxon tests with Holm correction |
| `reports/evidence/tables/grouped_validation_shallow_mlp.csv` | Corrected grouped-validation MLP sensitivity |
| `reports/evidence/tables/pre_normalization_band_ablation.csv` | Strict pre-normalization band-removal sensitivity |
| `reports/evidence/xai/` | OOF SHAP and other attribution artifacts |
| `reports/evidence/uncertainty/` | MC-dropout and RF interval diagnostics |

## Notes

* The manuscript distinguishes historical canonical artifacts from current
  reproducible reruns; they must not be silently mixed.
* Internal cache keys such as `ae_logspec` are retained for compatibility;
  publication labels describe their actual transformations.
* Figure generators and manuscript mappings are listed in
  `docs/figure_script_toc.md` and `reports/publication/figure_toc.md`.
