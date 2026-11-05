# P2.3 exploratory injection and false-positive study

This is a fixed-protocol prototype for the scan statistic, not a calibrated
subhalo search. It uses the maximum *data* `delta_chi2` over four positions
`(x,y) in {-0.3,+0.3} arcsec` and three spherical-TNFW strengths
`alpha_Rs_arcsec in {0,0.02,0.05}`; the injected position is `(0.3,0.3)` arcsec.
Other TNFW settings are `Rs=0.2 arcsec` and `r_trunc=0.8 arcsec`. For each
image, a one-start fit uses the fixed gamma prior `Normal(2,0.1)` and at most 80
function evaluations. The provisional raw cutoffs are `delta_chi2 >= 2, 5, 10`.
They are comparison points, **not formal significance levels**.

The simulated arm contains 20 seed-fixed no-subhalo lenses and two injections
per lens. Real Q1 injection preserves the observed residuals by adding the
best-fit smooth-with-TNFW image minus the smooth-only image to each of the 24
original cutouts. This controls for the observed residual structure while still
being a model-based injection. The Q1 adapter uses the documented outer-border
MAD noise estimate and unverified 0.16 arcsec Gaussian PSF. No per-image settings
changed.

## Results

| Data | Nulls | Null median / 95th percentile max delta-chi-square | False positives at cuts 2 / 5 / 10 | alpha 0.02 recoveries at cuts 2 / 5 / 10 | alpha 0.05 recoveries at cuts 2 / 5 / 10 |
|---|---:|---:|---:|---:|---:|
| Simulated | 20 | 0.957 / 3.068 | 4 / 0 / 0 | 5 / 0 / 0 | 5 / 1 / 0 |
| Real Q1 | 24 | 1700.230 / 148266.122 | 22 / 22 / 22 | 24 / 24 / 24 | 24 / 24 / 24 |

The apparent 24/24 Q1 injection recovery is not useful: the same detections occur
in 22/24 un-injected images at every tested cutoff. The large null maxima show
that source structure, residual sky/noise, PSF mismatch, and/or other model
deficiencies dominate this statistic on the supplied images. Raising these raw
cutoffs from 2 to 10 did not lower the observed Q1 false-positive count.

An integrated S/N proxy is `sum(max(image,0))/sqrt(sum(noise_sigma^2))`, with an
independent-pixel approximation. This Q1 sample's low/middle/high S/N terciles
had 7/8, 8/8, and 7/8 false positives at all three cuts, respectively; each
tercile had 8/8 recoveries at both injection strengths. Simulated-tercile
S/N was narrowly distributed (about 99–103), so it does not establish a useful
S/N trend. Full S/N stratification is in the raw result JSON.

All simulated grid fits converged (240/240 per null or injection arm). For real
Q1 the converged grid-point counts were 253/288 null, 226/288 at alpha 0.02, and
221/288 at alpha 0.05. The raw best-point statistic is still reported when some
grid fits did not converge, so those outcomes add uncertainty rather than being
silently omitted. Raw per-image outcomes and S/N values are in
[`p2-injection-results.json`](p2-injection-results.json). Reproduce with
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 .venv/bin/python tests/bench/injection_recovery.py`.

There is no conversion from `alpha_Rs_arcsec` to a subhalo mass in this study.
The Q1 input lacks the lens/source redshifts, mass-profile calibration, and
validated PSF/noise products needed to make that conversion or interpret raw
delta-chi-square as a false-alarm probability. The Q1 null sample is also small
and selected; conclusions are limited to these cutouts and this prototype.
