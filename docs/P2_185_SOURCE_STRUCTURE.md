# Source-structure attribution check on all 185 Q1 cutouts

## Fixed comparison

The one-Sersic baseline fit and its fixed-grid TNFW scan for all 185 Q1 cutouts
were compared with one two-Sersic fit per cutout. Both source components share
ellipticity and center; the second component adds amplitude, radius and index.
It starts with amplitude 3, radius 0.06 arcsec and index 1. The primary source
starts from the fixed B6 baseline fit. The two-Sersic fit uses one start, seed
731, `gamma_prior_sigma=0.1`, and `max_nfev=80`. No cutout-specific settings
were chosen, and all finite fits are retained in the full-sample summary.

For each cutout, source gain is `chi2(one Sersic)-chi2(two Sersics)`; TNFW gain
is the maximum fixed-grid `delta_chi2`. These are raw data-chi-square
improvements. They are not corrected for three additional source parameters
or for the TNFW position/strength search, and the TNFW statistic includes any
non-converged scan points.

## Results

| Population | n | Source Δχ² median (p16–p84) | Source median % baseline χ² | TNFW Δχ² median (p16–p84) | TNFW median % | Source gain larger |
|---|---:|---:|---:|---:|---:|---:|
| All finite comparisons | 185 | 9,418 (764–71,556) | 3.50% | 8,640 (161–65,164) | 2.99% | 107/185 |
| Two-Sersic optimizer converged | 108 | 4,442 (426–44,385) | 1.66% | 14,463 | 3.63% | 50/108 |
| One-Sersic baseline χ²/dof ≤ 10 | 73 | 3,509 (355–22,535) | 6.11% | 1,786 | 2.63% | 50/73 |
| One-Sersic baseline χ²/dof > 100 | 28 | 28,906 (1,382–194,543) | 0.115% | 40,395 | 0.042% | 14/28 |

The two-Sersic optimizer converged for 108/185 cutouts, contacted one or more
parameter bounds for 79/185, and produced zero hard errors. All 77 finite
non-converged fits reached the fixed 80-function-evaluation limit. In the full
sample, the raw two-Sersic improvement exceeds the TNFW gain in 107/185 cases;
among converged two-Sersic fits this is 50/108. The full-sample median includes
very large residual cases; for the 28 cutouts with baseline chi-square/dof
above 100, the median source gain is only 0.115% of baseline chi-square.

The first 153 cutouts match the historical source-structure output exactly in
two-Sersic chi-square, source gain and TNFW gain. The 185-cutout result is
therefore the historical 153 plus the 32 new cutouts under the same fixed
configuration.

## Attribution limit

This comparison cannot identify what physically caused the residuals. A more
flexible source can absorb residual patterns from source morphology, but it
can also absorb PSF errors, correlated or misestimated noise, sky structure or
other model mismatch. The two-Sersic family is constrained and is not an
arbitrary source. Q1 has no measured PSF or RMS map; its noise is estimated
from the outer border and treated as independent. The fraction of Q1 residual
chi-square due to source structure rather than PSF/noise/mass-model mismatch
is **not identifiable from these data and fits**. The reported gains quantify
fit flexibility only, not causal attribution.

These results do not change the P2 no-go: Q1 has no non-lens controls, the
unmodified candidate trigger rate is already high, and no physical subhalo
mass threshold can be derived from the tested angular `alpha_Rs` strengths.

All per-cutout parameters, chi-square values, fit statuses, bounds and source
versus TNFW gains are in
[`b6-185/source-structure.jsonl`](b6-185/source-structure.jsonl); aggregate
statistics are in
[`b6-185/source-structure.summary.json`](b6-185/source-structure.summary.json).
Reproduce with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/q1_153_source_structure.py \
  --b5-input docs/b6-185/real-robustness.jsonl \
  --output docs/b6-185/source-structure.jsonl
```
