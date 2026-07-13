# Manuscript Reproducibility Ledger

## Purpose

This ledger is the authoritative map between the manuscript, executable
analyses, compact evidence artifacts, and the frozen public snapshot. It exists
to prevent historical results, current reruns, robustness variants, and
method-specific diagnostics from being described as the same experiment.

## Frozen snapshot

- Public repository: `https://github.com/mingwucn/PhysicalGrindingFusion`
- Immutable review tag: `review-v3` (created after this correction cycle)
- Tagged tree: `https://github.com/mingwucn/PhysicalGrindingFusion/tree/review-v3`
- Historical artifact commit: `7446310571510fd615da7a86a0fb31ebe0ffd31d`
- Public export builder: `scripts/sync_physical_grinding_fusion.py`
- Public export manifest: `PUBLIC_EXPORT_MANIFEST.json`

The public repository is generated from an allowlist. Raw recordings,
intermediate caches, and checkpoints are not mirrored because of their size;
available large artifacts may be requested from the corresponding author.

## Authoritative data identity

| Item | Source of truth |
|---|---|
| Final evaluation targets | `reports/evidence/tables/final_evaluation_targets.csv` |
| Target checksum and excluded sample | `reports/evidence/tables/final_evaluation_targets_manifest.json` |
| Condition target statistics | `reports/evidence/tables/condition_target_summary.csv` |
| Process parameters | `data/parameters.xlsx` |
| Target loader and indexing | `src/grinding_physic_fusion/data/dataset.py` |
| Canonical LOGO assignments | `reports/evidence/tables/canonical_logo_assignments.csv` |
| Cache definitions and hashes | `reports/evidence/tables/cache_provenance.csv` and `overleaf/main/supp_cache_provenance.tex` |

The evaluation set contains 319 passes. Condition 1, sample 1 is excluded.
Condition 7 has mean measured Ra `0.35397000` um and population SD
`0.01314995` um. Condition 10 has mean measured Ra `0.07826000` um.

Regenerate the target manifest with:

```bash
PYTHONPATH=src python scripts/generate_evaluation_target_manifest.py
```

## Representation taxonomy

Internal cache names are retained for compatibility. Publication labels must
describe the implemented transform.

| Internal name | Publication label | Implemented quantity |
|---|---|---|
| `ae_spec` | AE-dB | Full-resolution AE magnitude spectrogram converted to dB |
| `vib_spec` | Vib-dB | Full-resolution vibration magnitude spectrogram converted to dB |
| `ae_logspec` | AE-dB-z | AE-dB after per-sample, per-plane z-standardisation |
| `vib_logspec` | Vib-dB-z | Vib-dB after per-sample, per-axis z-standardisation |
| `ae_mel` | AE-log-mel | Pass-mean dB inverted to geometric-mean power, mel filtered, then converted to dB |
| `vib_mel` | Vib-log-mel | Pass-mean dB inverted to geometric-mean power, mel filtered, then converted to dB |
| `ae_wst` | AE-WST | Two-dimensional wavelet scattering descriptor |
| `vib_wst` | Vib-WST | Two-dimensional wavelet scattering descriptor |

The implemented log-mel operation is not equivalent to averaging local
power-mel maps because local dB maps are averaged before inverse conversion.
This processing order is documented in the manuscript and must not be silently
renamed as a conventional arithmetic pass-average log-mel representation.

## Pass aggregation

- Prediction unit: one completed grinding pass.
- Final cache shape: one pass-mean representation per valid pass.
- Intermediate local maps: 2,910 maps per pass in the archived intermediate
  representation.
- Temporal ordering is removed by full-pass averaging.
- The implementation supports pass-end retrospective prediction; it does not
  demonstrate within-pass streaming or closed-loop readiness.

## Validation protocols

### Canonical outer LOGO

- Sixteen outer folds, one held-out process condition per fold.
- One complete remaining condition is reserved for validation.
- Fourteen conditions are used for fitting.
- The canonical validation assignments are deterministic and recorded in
  `canonical_logo_assignments.csv`.

### Current fixed RF

- Same test and validation assignments as canonical LOGO.
- Fourteen fitting conditions.
- 200 trees, maximum depth 8, minimum leaf size 1, random state 42.
- Spectral RF pipelines do not use a fold-wise scaler.
- dB-z caches already contain per-sample standardisation.

### Current nested RF

- Same fourteen-condition fitting budget as the fixed RF.
- Five inner grouped folds select RF hyperparameters.
- Selected model is refitted on the same fourteen fitting conditions.
- Per-fold selections are in `overleaf/main/supp_nested_rf_selections.tex`.

### Historical shallow MLP

- Uses scikit-learn internal random 10% validation.
- Included only in the archived 14-comparison family.
- Its significant RF comparison is implementation-specific.

### Corrected grouped-validation MLP

- Uses the designated whole validation condition.
- Input and target scalers are fitted on the fourteen fitting conditions only.
- Internal random validation is disabled.
- Uses a three-candidate training-only configuration grid.
- Mean MAE: `0.05293295` um; maximum fold MAE: `0.26449296` um.
- Script: `scripts/grouped_validation_shallow_mlp.py`.
- Candidate grid and fold selections:
  `reports/evidence/tables/grouped_validation_shallow_mlp_protocol.json`.
- Documentation generator: `scripts/generate_grouped_mlp_documentation.py`.

### Current fixed-setting five-seed analysis

- Seeds 42 through 46.
- Same fourteen-condition fitting budget for RF and neural baselines.
- Architecture and current-protocol settings are fixed within the repeated set.
- These runs do not reproduce the historical tuned neural checkpoints.
- Interpret as current-protocol stochastic sensitivity, not seed variance
  around the historical benchmark.

## RF experiment families

The full table is `overleaf/main/supp_rf_experiment_provenance.tex`.

| Family | Training conditions | Role |
|---|---:|---|
| Historical canonical dB-z/log-mel | 14 | Archived ranking and 14-comparison family |
| Current fixed dB-z/dB | 14 | Primary reproducible fixed results |
| Current nested dB-z/log-mel | 14 | Matched inner-grouped sensitivity |
| Equal-budget RF seed repeat | 14 | Current fixed-setting repeated analysis |
| Matched modality RF | 14 | Exact current-split modality ablation |
| Robustness/masking RF | 15 | Separate robustness diagnostics |
| RF OOB uncertainty variant | 15 | Method-specific interval diagnostic |
| OOF TreeSHAP RF | 14 | Explanation of current RF held-out predictions |

Do not combine values from these families without naming the experiment ID and
training-condition budget.

## Central performance interpretation

- Current fixed dB-z RF mean MAE: `0.0210` um.
- Current nested dB-z RF mean/max fold MAE: `0.0207` / `0.1402` um.
- Current nested log-mel RF mean/max fold MAE: `0.0206` / `0.2061` um.
- dB-z and log-mel form a leading group by mean error.
- dB-z has better observed worst-condition behavior in the current nested
  analysis.
- Differences below approximately `0.005-0.010` um are not treated as
  practically meaningful given the available metrological evidence.
- The matched 14-condition modality rerun is authoritative for the AE/fusion
  summary; its generated values are in
  `reports/evidence/tables/rf_modality_ablation_14condition.csv`.
- Matched means: AE only `0.02823394` um, vibration only `0.02172285` um,
  and AE plus vibration `0.02096063` um.

## Statistical analysis

- Effective paired sample size: sixteen held-out conditions.
- Primary family: fourteen historical post-selection pairwise comparisons.
- The family is artifact-based: exact historical prediction files are resolved
  dynamically from `canonical_prediction_manifest.csv` and are not presented
  as independently regenerated current results.
- Test: paired Wilcoxon signed-rank with documented zero handling.
- Multiplicity: Holm-Bonferroni correction over the fourteen tests.
- Effect estimate: paired Hodges-Lehmann Walsh-average location shift.
- Confidence intervals are unadjusted descriptive bootstrap intervals.
- Only RF versus the historical implementation-specific shallow MLP survives
  Holm correction.
- The corrected grouped MLP sensitivity is not inserted into the historical
  family.

Primary files:

- `reports/evidence/tables/statistical_tests_updated.csv`
- `reports/evidence/tables/supp_holm_full_family.csv`
- `overleaf/main/statistical_comparison_table.tex`

## Interpretability protocols

### Current RF out-of-fold TreeSHAP

- A separate RF is fitted for every outer fold.
- Each held-out pass is explained only by the RF trained without its condition.
- Profiles are reported as sample-pooled, condition-balanced, and
  condition-balanced excluding Condition 7.
- Condition-balanced vibration mass above 15 kHz is 85.76%; excluding
  Condition 7 it is 85.74%.
- Vibration peak: 16.2 kHz. AE peak: 334 kHz.
- This is the primary interpretability analysis in the main Results.

### Secondary cross-model explanations

- Global LightGBM TreeSHAP: illustrative exact-tree explanation.
- ResNetVibCNN Grad-CAM: CNN-specific vibration explanation.
- Differences between these profiles cannot be attributed to the explainer
  alone because model, input, preprocessing, and aggregation also differ.

### Frequency-band ablations

- Evaluation-time masking tests redundancy inside a fitted RF.
- Retrained removal tests whether a model can be learned without direct bands.
- Strict pre-normalisation removal deletes vibration bands from dB before
  recomputing per-sample z-scores.
- Removing vibration above 15 kHz before normalisation increases mean MAE from
  `0.02093` to `0.02547` um; descriptive paired interval
  `[0.00063, 0.01066]` um.
- The upper band is predictive for the fitted RF, but sensor/mounting and
  acquisition-chain behavior cannot be separated from grinding dynamics.

## Uncertainty protocols

### MC-dropout

- Uses a separately regenerated ResNetVibCNN checkpoint set.
- Deterministic checkpoint-set MAE: `0.0443` um.
- The stochastic mean is a materially different predictor from the historical
  best CNN.
- Reported bands contain epistemic dropout variance only.
- Coverage is 50.2%; interpret as failure of an epistemic-only band to serve as
  a predictive interval, not as universal invalidation of MC-dropout.

### RF intervals

- OOB/tree-variance analysis is a separate fifteen-condition RF uncertainty
  variant, not the exact fourteen-condition benchmark RF.
- All interval methods fail on Condition 7.
- Calibration formulas and conventions are documented in Methods and the
  uncertainty evidence tables.

## Condition 7

- Final mean measured Ra: `0.35397000` um.
- Population SD: `0.01314995` um.
- It controls maximum-fold error, residual bias, and uncertainty failure.
- Target-support distance is a post-hoc diagnostic because the target is not
  available online.
- Input-space novelty, sensor RMS, run order, and Condition 10 comparisons do
  not establish a unique physical mechanism.
- Without force, power, temperature, wheel-state, surface-image, and calibrated
  sensor-chain evidence, describe Condition 7 as a difficult condition-specific
  distribution shift rather than a confirmed physical regime.

## Latency and deployment

- Reported timing is feature-precomputed model inference.
- Fold checkpoint hashes and per-fold timings are in
  `edge_latency_benchmark_per_fold.csv`.
- Full-pass aggregation imposes pass-end observation latency.
- Signal acquisition, representation construction, scaling, communication,
  scheduling, and target hardware are excluded.
- More than 50 model calls per second after feature availability is not a
  50 Hz process-monitoring update rate.

## Figure and table generation

- Python style contract: `docs/12_VISUALIZATION_CONTRACT.md`.
- Figure generator map: `docs/figure_script_toc.md`.
- Manuscript figure provenance: `reports/publication/figure_toc.md`.
- TikZ style source: `overleaf/tikz/publication_styles.tex`.
- Figure validation: `python scripts/validate_publication_figures.py`.
- Tables and figures must use publication labels; internal cache names are
  allowed only in provenance records.

## Known unavailable evidence

The following cannot be reconstructed from the repository and must remain
explicit limitations:

- exact historical package environment for the canonical 0.0200/0.0201 um artifacts;
- resolved overrides for `MultiscaleSpectrogramCNN` and `SSLResNetVibCNN`;
- calibrated AE and vibration acquisition-chain transfer functions;
- raw profilometer traces and gauge R&R;
- independent machine, wheel, dressing-cycle, material-batch, or sensor-remount validation;
- direct physical diagnostics for Condition 7;
- demonstrated partial-pass streaming or closed-loop control.

The model inventory is therefore described as 25 exactly reconstructable
variants plus two archival variants with incomplete configuration provenance.

## Change-control checklist

Before changing any reported number:

1. Identify the experiment family and training-condition budget.
2. Regenerate the source CSV/JSON rather than editing a rendered table.
3. Update the manuscript, supplementary table, and figure caption together.
4. Update both figure TOCs when a figure moves or changes meaning.
5. Regenerate the target manifest if target indexing changes.
6. Run figure, contract, and LaTeX-reference validation.
7. Run `graphify update .` after code changes.
8. Regenerate the allowlisted public snapshot.
9. Create a new immutable public tag for a submitted revision.
