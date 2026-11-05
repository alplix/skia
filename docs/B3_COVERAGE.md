# B3 local parameter-error coverage

The original study below covers the 15-parameter single-Sersic fit. The
two-Sersic follow-up covers all 18 parameters, with held-out calibration factors
and boundary diagnostics: see [B3 two-Sersic coverage](B3_TWO_SERSIC_COVERAGE.md).
The paired source-mismatch follow-up tests one-Sersic lens errors on
two-Sersic truths: see [B3 source-mismatch coverage](B3_SOURCE_MISMATCH_COVERAGE.md).

## Method

The fitter now records approximate one-sigma parameter errors from the weighted
least-squares Jacobian. It forms `(JᵀJ)⁻¹`, scales by data `chi2/dof`, and applies
the first-order delta method to log-amplitude and log-radius parameters. The
gamma-prior curvature is included when present. If the Jacobian is rank
deficient, errors are emitted as `null`. These are local Gaussian
approximations; they omit nonlinear, bound-truncation and model-selection
effects.

Coverage uses 200 fixed-seed matched-model simulations with the same lens/source
ranges, 40x40 observation, 2-start fitter and noise seed sequence as P1. It
counts truth inside ±1 reported sigma (nominal 68%) and ±1.96 sigma (nominal
95%). All draws are retained. The raw per-fit errors and truth are in
[`b3-coverage/coverage.jsonl`](b3-coverage/coverage.jsonl), with configuration
in [`protocol.json`](b3-coverage/protocol.json) and counts in
[`summary.json`](b3-coverage/summary.json).

## Observed coverage

All 200 fits returned finite, converged, full-rank Jacobians; 10 contacted a
parameter bound. Coverage across all finite fits:

| Parameter | 1 sigma | 1.96 sigma |
|---|---:|---:|
| lens theta_E | 66.5% | 90.5% |
| lens gamma | 66.0% | 95.5% |
| lens e1 | 68.0% | 90.5% |
| lens e2 | 69.5% | 92.5% |
| lens center_x | 61.0% | 90.0% |
| lens center_y | 67.5% | 92.0% |
| lens gamma1 | 71.0% | 93.0% |
| lens gamma2 | 67.0% | 92.5% |
| source amp | 66.0% | 92.5% |
| source R_sersic | 70.5% | 95.5% |
| source n_sersic | 67.0% | 93.5% |
| source e1 | 70.0% | 95.0% |
| source e2 | 66.0% | 94.5% |
| source center_x | 67.5% | 94.0% |
| source center_y | 59.5% | 95.5% |

Among the 190 converged interior fits, 1-sigma coverage was 61.6–73.2% and
1.96-sigma coverage was 92.6–97.4%. Conditioning on interior fits excludes
boundary cases and does not replace the all-finite result. At n=200, sampling
uncertainty is material, but the 1-sigma undercoverage for lens center_x and
source center_y is visible and some 95% rates are also low.

The local Hessian errors are not calibrated well enough for scientific
confidence claims, especially for centers and boundary cases. The sample is
matched to the model with known synthetic noise/PSF; no Q1 coverage is assessed.
Run `make test` for the exact line; tests verify the covariance
calculation and serialization, not statistical coverage outside this study.
