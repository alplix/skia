# B3 diagnosis of the wide 95% error scales

## What drives the 1.06–10.05 range

This analysis uses the existing 200 paired one-Sersic-fit / two-Sersic-truth
simulations. It makes no new fit, changes no optimizer setting, and does not
select a scale to improve test-fold coverage. The five folds retain the
pre-existing deterministic assignment `index % 5`; each scale is computed
from the other 160 finite simulations and evaluated on the held-out 40. The
absolute standardized error is
`z=(fit-truth)/local_covariance_error`, and the fold factor is
`higher_quantile(|z|, 0.95)/1.96`.

| Lens parameter | 95% factor range over folds | Median fold factor | Raw 95% coverage | Held-out 95% coverage |
|---|---:|---:|---:|---:|
| `theta_E` | 1.097–1.457 | 1.457 | 89.5% | 93.5% |
| `gamma` | 1.063–1.210 | 1.137 | 93.5% | 95.5% |
| `e1` | 1.716–4.463 | 4.463 | 88.0% | 95.5% |
| `e2` | 1.231–4.594 | 4.225 | 89.5% | 95.0% |
| `center_x` | 2.807–10.054 | 10.054 | 88.5% | 95.5% |
| `center_y` | 1.507–4.667 | 4.123 | 91.0% | 95.5% |
| `gamma1` | 1.448–1.802 | 1.802 | 89.0% | 95.5% |
| `gamma2` | 1.236–2.169 | 2.169 | 88.0% | 95.5% |

The largest scales are specific to lens center and ellipticity (`center_x`,
`center_y`, `e1`, `e2`); `theta_E` and the slope `gamma` need much smaller
scales in these folds. This is not one global variance misspecification.

The maximum, `center_x` factor 10.054, comes from the fold-0 training set. Its
160 training cases have a higher-method 95th percentile `|z|=19.71`. The 8
training cases at or above this tail threshold are converged fits; 7 of the 8
contact at least one parameter bound. Their fitted `center_x` values are all
negative while the true centers are near zero, so multiplying the local
covariance error is compensating for substantial fit bias as well as local
error underestimation. A variance multiplier cannot remove that bias.

This tail recurs across lens parameters: training-tail cases 38 and 114 are in
the top 5% for all eight parameters; indices 4, 37 and 154 are also in several
parameter tails. For the parameter-specific maximum-scale fold tails, 6–7 of
the eight cases touch a fit bound. Their secondary-source amplitude fraction,
relative radius and index medians range from 0.446–0.570, 0.447–0.517 and
1.986–2.387, respectively; the corresponding full training-set medians are
0.379–0.391, 0.386–0.394 and 1.904. This points to a combination of strong,
extended or higher-index secondary source structure and bound-sensitive lens
fits in the extreme error tail. It is descriptive: tail sizes are eight cases,
the high-error cases overlap across parameters, and source-property
stratification was not specified before the simulation. It is not a validated
regime classifier or a basis for per-regime correction.

The fold-held-out coverage after the factors is 93.5–95.5% across parameters,
but the factor spread shows that the tails are poorly determined by 160
training cases. Boundary-contact rows are part of this analysis; the fit
convergence flag alone does not identify the largest errors.

## Proposed provisional inflation rule

Do not use a single global multiplier. For a future representative calibration
simulation set, estimate one factor for each reported parameter using only the
calibration rows:

```text
z[i, j] = (fit[i, j] - truth[i, j]) / local_covariance_error[i, j]
k95[j]  = max(1, quantile_higher(abs(z[calibration, j]), 0.95) / 1.96)
sigma_calibrated[j] = k95[j] * local_covariance_error[j]
95% interval[j] = fit[j] +/- 1.96 * sigma_calibrated[j]
```

Freeze each `k95[j]` before evaluating any held-out simulations or science
targets. Keep boundary contacts and non-converged fits explicitly flagged;
inflation does not turn a bound-limited or biased estimate into a trustworthy
measurement. Do not fit extra morphology-dependent factors from this 200-case
sample. A larger independent calibration set should be used to estimate the
factors and their uncertainty before this rule is implemented in fit output.

For the current sample this remains a documented proposal, not a production
correction: the center/shape factors vary widely and some corrections are
large enough to be uninformative. The local covariance errors remain
approximate, with source-model mismatch a demonstrated limitation. No held-out
fold was used to choose or revise the formula above.

The underlying paired outcomes and full fold summary remain in
[`b3-source-mismatch-coverage-v2/coverage.jsonl`](b3-source-mismatch-coverage-v2/coverage.jsonl)
and [`b3-source-mismatch-coverage-v2/summary.json`](b3-source-mismatch-coverage-v2/summary.json).
