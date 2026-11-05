# B1 code-audit follow-up

The provisional P1 review noted that a `src/` code audit had not yet been done.
This local review focused on persisted scan inputs and the public cutout/baseline
parsers, where malformed values can cross an API boundary.

| Finding | Location | Change | Regression coverage |
|---|---|---|---|
| A checkpoint row with a finite but incomplete parameter mapping could pass resume validation. | `src/skia/scanner.py`, `_validate_checkpoint_rows` | Require the exact one-/two-Sersic fit parameter keys, validate bound names and nonnegative `nfev`, and require complete baseline parameters. | `tests/test_task_bundle.py`: malformed result rows and incomplete baseline checkpoint |
| String or complex image arrays could leak low-level NumPy type/cast behavior. | `src/skia/cutout.py`, `Cutout.__post_init__` | Require real numeric image/noise/PSF arrays and normalize them to float64; emit a stable validation error. | `tests/test_cutout.py`: constructor and NPZ loader inputs |
| Boolean, non-finite, string and overflowing JSON baseline values could be accepted or fail inconsistently; normalization also mutated the caller's mapping. | `src/skia/scanner.py`, `vector_from_fit` | Validate finite real numeric values before model construction and work on a copy of the mapping. | `tests/test_scanner.py`: malformed values and no-mutation case |
| Boolean grid and halo coordinates silently converted to numeric 0/1. | `src/skia/scanner.py`, grid and TNFW input normalization | Reject Python and NumPy booleans for scalar radii, axes and point triples. | `tests/test_scanner.py`: boolean cases for grids and scalar/vector fields |

The audit also inspected the task ZIP reader/writer and Q1 FITS/noise adapter.
Their tested schema/hash, file-size, WCS, finite-pixel and noise-estimation
checks were consistent with the documented local formats. This is a targeted
input-boundary code review, not a substitute for an independent reviewer or a
portable-build/security audit.

Latest complete suite after these changes: `make test` — `114 passed, 7 warnings
in 39.80s` (seven upstream Matplotlib pending-deprecation warnings).
