# B7 public API test audit

This is a source-level map of the package's public `src/skia` interfaces to
their direct regression tests. It records exercised behavior, not a statement
of branch or line coverage. No coverage plugin is present in the pinned
environment, and no third-party dependency was added for this audit.

| Public module/API | Direct test locations | Exercised behavior |
|---|---|---|
| `cutout`: `Cutout`, `load_cutout`, `save_cutout` | `tests/test_cutout.py`, `tests/test_fitter.py` | Immutable/validated worker input, deterministic path and file-like serialization, truth-field stripping, round-trip, malformed archives and units |
| `fitter`: parameter conversion, covariance errors, `SmoothRenderer`, `FitConfig`, `fit_cutout`, `save_fit` | `tests/test_fitter.py`, `tests/test_scanner.py`, `tests/test_lenstronomy_crosscheck.py` | Parameter round-trip, singular covariance, analytic/numeric Jacobian agreement, render parity, fit recovery/failures/priors, deterministic serialization and independent-method smoke |
| `realdata`: `NoiseEstimate`, `estimate_sky_noise`, `load_q1_fits` | `tests/test_realdata.py` | Deterministic border MAD, masks, sparse/invalid input errors, FITS WCS scale, native units, non-image primary and PSF validation |
| `scanner`: baseline/grid normalization, scalar/vector halo fields, `scan`, `save_scan` | `tests/test_scanner.py`, `tests/test_task_bundle.py` | Grid schema and numeric validation, zero/finite fields, scalar/vector agreement, empty batches, baseline refit, fixed output, checkpoint resume and malformed state |
| `simulator`: `Lens`, `Source`, `Subhalo`, `Observation`, `Simulation`, `deflection`, `gaussian_psf`, `simulate`, `save_simulation` | `tests/test_simulator.py`, `tests/test_cutout.py`, `tests/test_fitter.py` | Constructor constraints, lens/shear/profile behavior, PSF normalization, deterministic noise, flux integration, serialization and fit-render parity |
| `task_bundle`: `create_task_bundle`, `read_task_bundle` | `tests/test_task_bundle.py` | Deterministic archive, member schema and hashes, canonical cutout contents, malformed JSON, resume integration |
| `cli.main` (`skia-fit`) | `tests/test_fitter.py` | Success, prior, optimizer failure, missing input and deterministic process output |
| `scan_cli.main` (`skia-scan`) | `tests/test_scanner.py` | Valid scan and malformed baseline exit behavior |
| `bundle_cli.main` (`skia-bundle`) | `tests/test_task_bundle.py` | Successful archive creation and validation failure exit behavior |
| `app_cli.main` (`skia-app`) | `tests/test_task_bundle.py` | Result format, progress output and byte-identical repeated runs |

Benchmark helpers under `tests/bench/` are experiment drivers rather than
installed package APIs. Their important protocol and aggregation paths have
focused tests where they are used as resumable or summary-producing runners;
the long scientific populations are recorded separately under `docs/`.

Run the complete package test suite with `make test`.

## Re-audit (2026-11-03)

Re-enumerated top-level public functions/classes in `src/skia/` and checked
their call paths against the test modules above. No package-level callable was
left without direct behavioral coverage. `FitResult` and `NoiseEstimate` are
public result records exercised through `fit_cutout` and `estimate_sky_noise`;
their returned fields are asserted in the corresponding fitter and real-data
tests. The audit remains a source-to-test map, not a numerical line/branch
coverage claim. Current full-suite result: `114 passed, 7 warnings in 35.28s`.
