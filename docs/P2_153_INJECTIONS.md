# Tile-held-out injection study on 153 Q1 cutouts

## Protocol and split

The full 153-image Q1 sample is grade-A lens candidates with Euclid reference
fits; it contains no non-lens negative controls. A deterministic SHA-256 hash
of `tile_index` (`skia-q1-153-v1:` salt, modulo 5) assigned whole sky-tile
groups before analysis: 100 calibration tiles / 124 cutouts and 22 evaluation
tiles / 29 cutouts. Tile overlap is zero. Per-image model fitting uses the same
fixed one-start, seed-731, `Normal(gamma=2, sigma=0.1)`, `max_nfev=80`
configuration and shared fit bounds; only calibration-tile matched simulations
set the q95 threshold. The evaluation split is not used to select that threshold
or the S/N bins.

Each matched clean null is a fitted smooth model rendered with that cutout's
oversampled PSF array and geometry plus independent Gaussian noise drawn from
that cutout's constant outer-border MAD noise map. This supplies a subhalo-free
simulation with the Q1 adapter's assumed PSF and estimated noise amplitude. It
does not reproduce correlated noise, unknown PSF errors, lens light or any
other real residual structure. The baseline fit parameters from the fixed B5
pipeline define the simulated smooth model. Every evaluation Q1 cutout also
received residual-preserving TNFW injections at `(x,y)=(0.3,0.3)` arcsec with
`alpha_Rs_arcsec=0.02` and `0.05`; `Rs=0.2` and truncation radius `0.8` arcsec.
Injection adds the fitted model-with-halo minus smooth-model image to the
unmodified Q1 image. These angular strengths are not masses.

The statistic is the maximum data `delta_chi2` over the fixed four-position,
three-strength grid. An empirical 95th percentile using NumPy's `higher`
quantile method was selected on calibration matched-clean-null statistics only:
`delta_chi2 = 5.666623163`. Fixed cuts 2, 5 and 10 are also retained for
comparison and were not retuned.

The unmodified candidate scan for each image is reused from the identical B5
fixed-grid run, avoiding duplicate work. The first calibration cutout was
processed during a preliminary runner attempt and its raw row additionally
contains two injection scans; those exploratory training-row injections are
excluded from all injection recovery summaries. Evaluation injections are run
only on the 29 held-out cutouts.

Raw per-image values, all matched-null and evaluation-injection grid points,
convergence counts, and the aggregate summary are in
[`p2-153-injection-results.jsonl`](p2-153-injection-results.jsonl) and
[`p2-153-injection-results.summary.json`](p2-153-injection-results.summary.json).

## Results

| Arm | Sample | Median max `delta_chi2` | Above calibration q95 5.6666 | At fixed cuts 2 / 5 / 10 |
|---|---:|---:|---:|---:|
| Calibration matched clean null | 124 | 0.000 | 7/124 (5.6%, in-sample) | 13/124 / 7/124 / 1/124 |
| Evaluation matched clean null | 29 | 0.000 | 2/29 (6.9%) | 2/29 / 2/29 / 2/29 |
| Evaluation unmodified Q1 candidates | 29 | 7,725.1 | 26/29 (89.7%) | 26/29 / 26/29 / 26/29 |
| Evaluation Q1 + alpha 0.02 | 29 | 16,661.5 | 27/29 (93.1%) | 27/29 / 27/29 / 27/29 |
| Evaluation Q1 + alpha 0.05 | 29 | 16,998.7 | 29/29 (100%) | 29/29 / 29/29 / 29/29 |
| All unmodified Q1 candidates | 153 | 5,841.5 | 132/153 (86.3%) | 132/153 / 132/153 / 131/153 |

The holdout matched-clean-null trigger fraction is 2/29 at the threshold fixed
on the separate calibration tiles. This is a small, model-matched Gaussian-noise
test and not a calibrated real-instrument false-positive rate. In contrast,
26/29 unmodified evaluation candidates already exceed the same threshold. These
are candidate-only images and may contain real subhaloes; **26/29 is a candidate
trigger fraction, not a false-positive rate**. At the calibrated threshold,
alpha 0.02 turns 3 previously unflagged candidates on and 2 previously flagged
candidates off; alpha 0.05 turns 3 on and none off. Thus the high nominal
injection recovery counts add little discrimination beyond the unmodified
candidate triggers.

At the true injected sky-grid corner `(0.3,0.3)`, the best scan point lands at
that exact position for 9/29 alpha-0.02 injections and 10/29 alpha-0.05
injections. The best-fit grid strength is not a clean estimate: for alpha 0.02,
the winning grid alpha was 0 / 0.02 / 0.05 in 2 / 8 / 19 cases; for alpha 0.05,
it was 0 / 0.02 / 0.05 in 0 / 3 / 26 cases.

The integrated positive-flux S/N proxy uses
`sum(max(image,0))/sqrt(sum(noise_sigma**2))`. Tercile edges were derived from
the calibration split (118.95 and 174.75) and then applied to evaluation:

| Evaluation S/N bin | n | Unmodified candidate triggers | alpha 0.02 triggers | alpha 0.05 triggers |
|---|---:|---:|---:|---:|
| Low | 6 | 4/6 | 4/6 | 6/6 |
| Middle | 9 | 8/9 | 9/9 | 9/9 |
| High | 14 | 14/14 | 14/14 | 14/14 |

No useful S/N trend is established: the candidate trigger rate is already high
in every bin, and the evaluation bins are small.

## Convergence and interpretation

The same convergence limitations affect the statistic:

| Arm | Converged grid points / total |
|---|---:|
| Calibration matched clean null | 1,063/1,488 |
| Evaluation matched clean null | 235/348 |
| Evaluation unmodified Q1 candidate (reused from B5) | 286/348 |
| Evaluation alpha 0.02 | 273/348 |
| Evaluation alpha 0.05 | 274/348 |

Non-converged points remain in maximum-statistic accounting. The clean-null
threshold therefore inherits incomplete optimizer fits; it is only an
exploratory threshold for this model-matched simulation. The Q1 unmodified
candidate rates cannot estimate real-data specificity because there are no
non-lens negatives. The injection recovery rates likewise cannot distinguish
subhalo response from residual/model mismatch. No useful mass sensitivity or
real-data false-positive rate is supported.

## Reproduction

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/q1_153_injection.py
```

