# Source-complexity attribution check on 153 Q1 cutouts

This is the historical 153-cutout result. The expanded all-185 comparison is
in [`P2_185_SOURCE_STRUCTURE.md`](P2_185_SOURCE_STRUCTURE.md); its first 153
per-cutout fit and gain values match this report exactly.

## Protocol

The fixed B5 one-Sersic fit and its fixed-grid TNFW scan were compared with one
two-Sersic fit for each of all 153 Q1 cutouts. The two-Sersic components share
ellipticity and center and add three free parameters (second amplitude, radius
and index). The second component starts at amplitude 3, radius 0.06 arcsec and
index 1. The first component starts from the fixed B5 one-Sersic fit. All
two-Sersic fits use one start, seed 731, `Normal(gamma=2, sigma=0.1)` and
`max_nfev=80`; no per-cutout settings or thresholds were selected. The model
comparison is a fixed, per-image analysis; no population-level training or
threshold tuning is performed on the evaluation cutouts.

For each cutout:

- source-complexity gain is `chi2(one Sersic) - chi2(two Sersics)`;
- TNFW gain is the maximum data `delta_chi2` from the fixed B5 12-point scan;
- both gains are also reported as a percentage of the one-Sersic baseline
  image chi-square.

These are raw, unpenalized fit improvements. They do not correct for three
additional source parameters or the TNFW position/strength search. The TNFW
maximum includes the fixed-grid look-elsewhere effect and may include
non-converged grid points. No likelihood-ratio calibration or model selection
claim is made.

All per-lens model parameters, chi-square values, convergence status, bounds,
messages, TNFW best point and comparison values are retained in
[`p2-153-source-structure.jsonl`](p2-153-source-structure.jsonl); aggregate
statistics are in
[`p2-153-source-structure.summary.json`](p2-153-source-structure.summary.json).

## Results

| Comparison population | n | Source Δχ² median (p16–p84) | Source median % of baseline χ² | TNFW Δχ² median (p16–p84) | TNFW median % | Source gain > TNFW |
|---|---:|---:|---:|---:|---:|---:|
| All finite two-Sersic fits | 153 | 7,533 (687–55,024) | 3.92% | 5,841 (62–57,978) | 2.79% | 93/153 |
| Two-Sersic optimizer converged | 88 | 3,509 (411–40,701) | 1.81% | 8,379 (13–60,624) | 2.75% | 44/88 |
| One-Sersic baseline chi-square/dof ≤ 10 | 70 | 3,439 | 5.90% | 1,792 | 2.71% | 47/70 |
| One-Sersic baseline chi-square/dof > 100 | 22 | 18,473 | 0.070% | 10,547 | 0.037% | 11/22 |

The two-Sersic fit converged for 88/153 cutouts and touched at least one
parameter bound for 68/153. The other 65 finite fits all stopped at the
80-evaluation limit; no hard errors occurred. Although the two-Sersic model's
unpenalized chi-square gain exceeds the TNFW gain in 93/153 full-sample fits,
that reverses to 44/88 among its converged fits. It does not consistently
explain more residual chi-square than the TNFW scan. The table's full-sample
median is influenced by extreme residual cases: for the 22 cutouts with
one-Sersic chi-square/dof >100, median gains are only 0.070% (two Sersic) and
0.037% (TNFW) of the baseline chi-square.

## Attribution limit

This comparison cannot identify what physically caused the residuals. A more
flexible source can absorb residual patterns from source morphology, but it can
also absorb PSF errors, correlated/misestimated noise, sky structure or other
model mismatch. The two-Sersic family has shared center/ellipticity and is not
an arbitrary source model. Q1 has no measured PSF or RMS map, and the estimated
noise is an independent-pixel outer-border MAD approximation. Therefore the
fraction of real Q1 residual chi-square dominated by source structure rather
than PSF/noise/mass-model mismatch is **not identifiable from these data and
fits**. The measured χ² reductions quantify flexibility only, not causal
attribution.

These results do not change the P2 no-go: the candidate-only Q1 sample cannot
give a clean false-positive rate, the injection statistic already triggers on
most unmodified candidates, and no redshift/cosmology conversion to subhalo
mass is available.

Reproduce with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/q1_153_source_structure.py
```

