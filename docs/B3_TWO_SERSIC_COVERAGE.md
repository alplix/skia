# B3 follow-up: two-Sersic parameter-error coverage

## Protocol

This follow-up calibrates the 18 local-Jacobian standard errors emitted by the
experimental shared-geometry two-Sersic fit. It uses 200 fixed-seed EPL+shear
simulations with two-Sersic truths, randomized lens and primary-source
parameters from the P1 ranges, and secondary-component amplitude fraction
0.15–0.60, radius 0.04–0.10 arcsec and Sersic index 0.8–3.0. The two source
components share center and ellipticity. Images are 40×40 at 0.1 arcsec/pixel,
oversampling 2. Noise is independent Gaussian with a known per-pixel standard
deviation computed from the noiseless richer source plus the simulator sky and
read-noise settings.

The fitter uses two starts, seed 731, `max_nfev=200`, no gamma prior, and
`source_components=2`. All finite outputs are retained, including optimizer
nonconvergence and bound contacts. Even-index simulations (100) set
per-parameter multiplier estimates: the empirical 68.27th percentile of
absolute standardized error for nominal 68%, and the empirical 95th percentile
divided by 1.96 for nominal 95%. The `higher` quantile method is fixed. Odd
indices (100) are held out for evaluation. Coverage is reported both for all
finite fits and for the converged, non-boundary subset. The latter contains
only 58 fits total (27 calibration, 31 evaluation).

Reproduce with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python \
  tests/bench/coverage_two_sersic.py --count 200 --max-nfev 200 \
  --output docs/b3-two-sersic-coverage
```

Re-analyze the saved population with five-fold calibration, without refitting:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python \
  tests/bench/coverage_two_sersic.py --analyze-existing \
  --output docs/b3-two-sersic-coverage
```

The exact protocol, all per-fit truth/estimates/errors, and full per-parameter
coverage counts and Wilson intervals are in
[`b3-two-sersic-coverage/`](b3-two-sersic-coverage/).

## Fit and error availability

All 200 fits returned finite parameters and covariance estimates; none had a
rank-deficient Jacobian. Seven did not converge. There were 137 fits with at
least one bound contact, leaving 58 converged interior fits. The most frequent
contacts were `source2.n_sersic` (108), `source2.R_sersic` (34), and
`source.n_sersic` (33). The unconstrained local covariance does not account for
truncation at those bounds.

## Held-out coverage

Each entry is a count out of 100 held-out finite fits. “Scaled” applies a
per-parameter factor derived only from the even-index calibration half. The
factors are shown as `k68 / k95`; their large variation is itself a warning
against treating one multiplier as a universal fix.

| Parameter | Raw 68% | Raw 95% | Scaled 68% | Scaled 95% | k68 / k95 |
|---|---:|---:|---:|---:|---:|
| lens theta_E | 70/100 | 90/100 | 78/100 | 91/100 | 1.26 / 1.05 |
| lens gamma | 63/100 | 93/100 | 69/100 | 94/100 | 1.05 / 1.16 |
| lens e1 | 65/100 | 90/100 | 61/100 | 94/100 | 0.93 / 1.28 |
| lens e2 | 61/100 | 92/100 | 61/100 | 94/100 | 1.00 / 1.36 |
| lens center_x | 65/100 | 86/100 | 71/100 | 93/100 | 1.21 / 1.53 |
| lens center_y | 66/100 | 86/100 | 67/100 | 90/100 | 1.03 / 1.09 |
| lens gamma1 | 69/100 | 91/100 | 77/100 | 94/100 | 1.14 / 1.31 |
| lens gamma2 | 66/100 | 87/100 | 70/100 | 94/100 | 1.12 / 1.44 |
| source amp | 74/100 | 92/100 | 73/100 | 96/100 | 0.95 / 1.48 |
| source R_sersic | 84/100 | 97/100 | 56/100 | 96/100 | 0.54 / 0.88 |
| source n_sersic | 80/100 | 98/100 | 68/100 | 91/100 | 0.67 / 0.70 |
| source e1 | 62/100 | 90/100 | 61/100 | 94/100 | 0.97 / 1.17 |
| source e2 | 68/100 | 91/100 | 75/100 | 90/100 | 1.19 / 0.99 |
| source center_x | 65/100 | 93/100 | 80/100 | 94/100 | 1.23 / 1.07 |
| source center_y | 66/100 | 98/100 | 73/100 | 98/100 | 1.10 / 1.00 |
| source2 amp | 73/100 | 87/100 | 68/100 | 97/100 | 0.80 / 1.63 |
| source2 R_sersic | 64/100 | 79/100 | 72/100 | 94/100 | 1.46 / 2.36 |
| source2 n_sersic | 70/100 | 84/100 | 72/100 | 97/100 | 1.11 / 1.88 |

The raw 68% coverage spans 61–84%; raw 95% spans 79–98%. The even/odd split's
held-out scaling changes these to 56–80% and 90–98%, respectively. This split
is noisy: for example, the source-radius 68% rate falls from 84/100 to 56/100
after its calibration factor shrinks the errors.

### Five-fold cross-validation

To test sensitivity to that one split without refitting, a second analysis
assigns folds by `index % 5`; each group of 40 is evaluated using scales fitted
on the other 160 rows. All 200 rows are evaluated once. For all finite fits,
the 18 parameters' cross-validated 68% rates range from 67% to 70%, and 95%
rates range from 94.5% to 96%; every parameter's Wilson 95% interval includes
its nominal rate. In the smaller converged-interior group (58 total), rates
range from 65.5% to 70.7% at 68% and 91.4% to 94.8% at 95%; all intervals
include the nominal rates. Per-parameter counts, intervals and fold-scale
ranges are in `crossval_summary.json`.

The fitted scale factors vary considerably across parameters: median fold `k68`
ranges 0.64–1.33 and median `k95` ranges 0.76–7.03 for all finite fits. The
large 95% tail factor for some parameters makes intervals broad. Cross-fold
held-out calibration therefore matches this simulation population, but the
single split shows that factors are sample-sensitive. It is an empirical
simulation result, not evidence that the raw fit JSON errors or Q1 intervals
are calibrated.

## Conclusion

The raw local-Jacobian errors are **not calibrated confidence intervals** for
this two-Sersic fit. Five-fold empirical scales achieved nominal coverage on
the matched simulated population, but the scales are not embedded in fit output,
vary strongly by parameter and sample split, and have not been validated on an
independent population or on Q1. The 137 bound-contact fits also show that the
local covariance approximation omits a common source of uncertainty. More
robust production intervals require independent calibration data, explicit
bound/model-degeneracy handling and Q1-like noise/PSF validation.
