# PhysicalGrindingFusion

Research code for pass-end surface-roughness prediction in precision grinding
from acoustic-emission, vibration, and process-parameter measurements.

The project evaluates auditable classical models and deep-learning baselines
under leave-one-condition-out validation across 16 grinding conditions. The
implemented representations include dB spectrograms, per-sample-standardised
dB descriptors, log-mel descriptors, wavelet scattering, scalar signal
features, process parameters, and heuristic physics-inspired indicators.

## Repository contents

* `src/grinding_physic_fusion/`: data, model, training, and validation code.
* `scripts/`: reproducible analysis and publication-figure entry points.
* `configs/`: experiment and runtime configuration.
* `tests/`: contract, preprocessing, runtime, and validation tests.
* `overleaf/`: manuscript and TikZ source.
* `reports/evidence/`: small result, provenance, XAI, and uncertainty manifests.
* `docs/figure_script_toc.md`: figure-to-generator mapping.
* `docs/35_MANUSCRIPT_REPRODUCIBILITY_LEDGER.md`: authoritative scientific
  provenance, experiment-family, result, and limitation ledger.
* `docs/37_SUBMITTED_MANUSCRIPT_ARCHIVE.md`: submitted manuscript revision,
  validation record, reconstruction instructions, and archive identifiers.
* `docs/README.md`: documentation index.
* `REPLICATION.md`: environment, data placement, and principal commands.

## Data availability

Raw sensor recordings, intermediate caches, trained checkpoints, and
profilometer data are not stored in Git because of their size. Research-use
access to the available large data artifacts may be requested from the
corresponding author. See `DATA_AVAILABILITY.md` in the public snapshot.

## Reproduction

Create the environment, place supplied data under the paths documented in
`src/grinding_physic_fusion/data/dataset.py`, and follow `REPLICATION.md`.
Historical canonical artifacts and current reproducible reruns are deliberately
identified separately in the evidence tables and manuscript.

## Submitted manuscript

The manuscript submitted to *Advanced Engineering Informatics* on 2026-07-13
is archived by the annotated tag `submission-aei-2026-07-13`. The tag pins the
exact Overleaf submodule revision. See
`docs/37_SUBMITTED_MANUSCRIPT_ARCHIVE.md` for checksums and reconstruction
instructions.

## License

See `LICENSE` when present. Dataset access is governed separately from source
code distribution.
