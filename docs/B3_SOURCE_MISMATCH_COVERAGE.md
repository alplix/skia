# B3 source-mismatch coverage follow-up

## Question and paired protocol

The original B3 study measured local-Jacobian parameter-error coverage when a
single-Sersic source was both the truth and the fit model. B2 showed that the
simulated source can contain two Sersic components. This follow-up checks
whether the reported **lens** parameter errors change when that one-Sersic fit
is applied to a two-Sersic truth.

It uses 200 matched pairs with the same lens and primary-source draws, 40×40
observation, noise seed and two-start/no-prior fit configuration as the original
[single-Sersic B3 run](b3-coverage/). Only the truth image changes: a secondary
Sersic component is added with shared geometry, amplitude 15–60% of the primary,
radius 0.04–0.10 arcsec and index 0.8–3.0. The noise uses Poisson counts plus
Gaussian read noise from each truth image. The same local covariance errors
and nominal ±1σ / ±1.96σ intervals are evaluated. Source parameters are omitted
because a single fitted Sersic parameter has no one-to-one truth counterpart.

The mismatch fit population has 200/200 finite results, 198 optimizer
convergences, 12 boundary-contact fits, and no rank-deficient covariance. The
two non-converged finite fits and all boundary cases remain in the all-finite
coverage denominator.

## Raw paired coverage

| Lens parameter | Matched 1-Sersic truth 68% | 2-Sersic truth, 1-Sersic fit 68% | Matched 95% | Mismatch 95% |
|---|---:|---:|---:|---:|
| `theta_E` | 66.5% | 65.5% | 90.5% | 89.5% |
| `gamma` | 66.0% | 66.5% | 95.5% | 93.5% |
| `e1` | 68.0% | 64.0% | 90.5% | 88.0% |
| `e2` | 69.5% | 65.5% | 92.5% | 89.5% |
| `center_x` | 61.0% | 61.5% | 90.0% | 88.5% |
| `center_y` | 67.5% | 62.5% | 92.0% | 91.0% |
| `gamma1` | 71.0% | 68.0% | 93.0% | 89.0% |
| `gamma2` | 67.0% | 64.0% | 92.5% | 88.0% |

The matched values are from the existing 200-fit B3 raw study. The paired
comparison confirms the same lens and primary-source truth at every index; the
second component is the only deliberate truth-model change. Raw 68% coverage
under mismatch ranges 61.5–68.0%; raw 95% coverage ranges 88.0–93.5%. The
changes are modest in this sample, but several parameters move further below
their nominal rates. At 200 pairs, sampling intervals remain broad, so these
rates do not establish a precise universal correction.

## Within-population five-fold scaling

Scales derived from four folds and evaluated on the remaining fold give
67.5–69.5% coverage for nominal 68% intervals and 93.5–95.5% for nominal 95%
intervals across the eight lens parameters. Every per-parameter Wilson 95%
interval includes its nominal rate. This calibration is specific to the
simulated two-Sersic population. Fold-to-fold 68% scale factors range 0.96–1.28;
95% factors range 1.06–10.05, showing that tail calibration is unstable for
some parameters even with 200 cases. No scale is embedded in fit output, and no
Q1 coverage is measured.

The follow-up diagnosis identifies the high factors with center/ellipticity
tails and bound-sensitive fits, and documents a parameter-specific,
training-only provisional inflation rule in
[`B3_95_SCALE_DIAGNOSTIC.md`](B3_95_SCALE_DIAGNOSTIC.md). The rule has not been
implemented in fitter outputs.

Raw per-fit parameters, errors, truth, convergence and bound contacts are in
[`b3-source-mismatch-coverage-v2/coverage.jsonl`](b3-source-mismatch-coverage-v2/coverage.jsonl).
The exact seeds and optimizer settings are in
[`protocol.json`](b3-source-mismatch-coverage-v2/protocol.json); aggregate raw,
paired and cross-validated results are in
[`summary.json`](b3-source-mismatch-coverage-v2/summary.json).

Reproduce with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/coverage_source_mismatch.py --count 200 \
  --max-nfev 200 --folds 5 --reference-coverage docs/b3-coverage/coverage.jsonl \
  --output docs/b3-source-mismatch-coverage-v2
```

These are controlled simulated results. They do not calibrate errors for Q1,
for other source-morphology distributions, or for real-data PSF/noise mismatch.
Local covariance outputs remain approximate errors rather than validated
confidence intervals.

An earlier 40-case exploratory run is retained in
[`b3-source-mismatch-coverage/`](b3-source-mismatch-coverage/); it used a
different image size, one optimizer start and a gamma prior, so it is excluded
from the paired results above.
