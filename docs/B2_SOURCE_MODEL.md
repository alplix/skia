# B2 source complexity: two Sersics versus one

## Question and model choice

The fitter supports a second Sersic component with shared center and
ellipticity. This adds source amplitude, effective radius and Sersic index while
holding source geometry common. It is an experimental alternative; the
single-Sersic model remains the default because this study does not show a
false-alarm or sensitivity improvement that would justify promoting the extra
component.

## Protocol

The primary paired study contains 40 fixed-seed simulations. Each truth is an
EPL+shear lens with a two-Sersic source; the secondary component shares the
primary center and ellipticity, with randomized amplitude fraction 0.15–0.60,
effective radius 0.04–0.10 arcsec and Sersic index 0.8–3.0. Lens and primary
source parameters are independently sampled from the P1 ranges. Images are
24×24 pixels at 0.1 arcsec/pixel with oversampling 1. The noise is independent
Gaussian with per-pixel standard deviation computed from the richer noiseless
source, fixed sky and read-noise settings. One realization is shared between
the one- and two-Sersic fits for each null/injection pair.

The common optimizer policy is one start, seed 731, `max_nfev=60`, and gamma
prior sigma 0.1. Every fit uses the same fixed four-position scan with
`alpha_Rs_arcsec` in {0, 0.01, 0.03, 0.06}. Injected model deltas are rendered
at (0.3, 0.3) arcsec for strengths 0.01, 0.03 and 0.06 arcsec. The statistic is
the maximum data Δχ² over the grid. Thresholds 2, 5 and 10 are descriptive raw
cuts; they are not calibrated significance thresholds. Wilson 95% intervals
show binomial sampling uncertainty. No samples were removed based on fit
quality.

Reproduce the formal study with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python \
  tests/bench/source_complexity_population.py --count 40 --max-nfev 60 \
  --output docs/b2-source-v3
```

Machine-readable protocol, per-lens outcomes and summary are in
[`b2-source-v3/`](b2-source-v3/). An earlier 40-lens run with only the 0/0.03
grid is preserved in `b2-source-v2/`; its false-alarm rates are not used below
because it searched fewer strengths.

## Results

| Raw Δχ² cut | Null triggers, one Sersic | Null triggers, two Sersics | α=0.01 recovery, one / two | α=0.03 recovery, one / two | α=0.06 recovery, one / two |
|---:|---:|---:|---:|---:|---:|
| 2 | 11/40 (27.5%) | 10/40 (25.0%) | 15/40 (37.5%) / 14/40 (35.0%) | 22/40 (55.0%) / 19/40 (47.5%) | 26/40 (65.0%) / 24/40 (60.0%) |
| 5 | 6/40 (15.0%) | 5/40 (12.5%) | 6/40 (15.0%) / 3/40 (7.5%) | 9/40 (22.5%) / 6/40 (15.0%) | 17/40 (42.5%) / 13/40 (32.5%) |
| 10 | 2/40 (5.0%) | 2/40 (5.0%) | 3/40 (7.5%) / 2/40 (5.0%) | 4/40 (10.0%) / 2/40 (5.0%) | 10/40 (25.0%) / 8/40 (20.0%) |

At Δχ²≥2, the more complex source model reduced null triggers by one case,
while it lost seven injection triggers and gained two in the paired sample.
At α=0.03, recovery at this cut fell from 22/40 to 19/40; at α=0.06 it fell
from 26/40 to 24/40. At cuts 5 and 10, the two-Sersic fit also recovered fewer
injections at every tested strength. False-alarm changes are small and overlap
within the wide binomial intervals; the two-Sersic model did not improve the
false-alarm/sensitivity tradeoff in these simulations.

The median null image χ²/dof changed only from 1.00696 to 1.00670. Null baseline
fits converged in 39/40 one-Sersic cases and 35/40 two-Sersic cases. Across the
640 null grid fits, 622 and 613 converged, respectively. Injection baseline fit
convergence for the one-/two-Sersic alternatives was 37/32 at α=0.01, 38/34 at
α=0.03, and 40/37 at α=0.06. Nonconverged finite grid points remain included
in trigger counts.

## Conclusion and limits

This two-Sersic extension is implemented and measured, but it is not a
demonstrated better default: it slightly changes raw null rates, reduces
injection recovery, and converges less often. The primary fitter default
therefore remains one Sersic; two components can be selected explicitly with
`source_components=2`. This is a controlled matched-Gaussian simulation, not
Q1 false-positive calibration. There are 40 lenses and broad Wilson intervals;
the scan is a small four-position grid. The source family shares center and
ellipticity and cannot model arbitrary morphology. These rates are not mass
sensitivity or calibrated detection significance. Q1 source attribution also
remains unresolved because the real images lack a measured PSF and noise map.
