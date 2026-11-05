# Injection recovery and trigger rates on all 185 Q1 cutouts

## Protocol

This expands the historical [153-cutout study](P2_153_INJECTIONS.md) to all
185 grade-A Q1 cutouts. Every row from the earlier 153 cutouts is preserved;
all 185 cutouts were processed together for a new calibration/evaluation split.
The fixed hash of `tile_index` (`skia-q1-153-v1:` salt, modulo 5) assigns whole
tiles: 147 calibration cutouts on 110 tiles and 38 evaluation cutouts on 28
tiles, with zero tile overlap. No cutout-specific threshold or optimizer
configuration was selected.

For each cutout, the fixed B6 baseline fit supplies a smooth model. A matched
clean null is that smooth model plus independent Gaussian noise from the
cutout's estimated constant noise map. The statistic is the maximum data
`delta_chi2` on the fixed 12-point TNFW grid. The empirical calibration q95
uses NumPy's `higher` method on calibration-tile clean-null maxima only. It is
`delta_chi2=4.526957404`. Fixed cuts 2, 5 and 10 are also reported unchanged.
For each of the 38 evaluation cutouts, the pipeline also scans the unmodified
candidate and residual-preserving injections at `(x,y)=(0.3,0.3)` arcsec with
`alpha_Rs=0.02` and `0.05` arcsec (`Rs=0.2`, truncation radius `0.8` arcsec).
These alpha strengths are angular lensing proxies, not subhalo masses.

The clean-null simulation reproduces the assumed PSF array, image geometry and
noise amplitude. It does not reproduce unknown PSF error, correlated noise,
source morphology mismatch or other real-data residuals. The Q1 sample contains
no non-lens controls, so a trigger on a Q1 candidate is not a false positive.

## Rates

| Arm | n | At held-out calibration q95 4.52696 | At fixed cuts 2 / 5 / 10 |
|---|---:|---:|---:|
| Calibration matched clean null (in-sample) | 147 | 8/147 (5.4%) | 13/147 / 7/147 / 1/147 |
| Evaluation matched clean null | 38 | 2/38 (5.3%) | 2/38 / 2/38 / 2/38 |
| Evaluation unmodified Q1 candidates | 38 | 33/38 (86.8%) | 33/38 / 33/38 / 33/38 |
| Evaluation Q1 + alpha 0.02 | 38 | 35/38 (92.1%) | 35/38 / 35/38 / 35/38 |
| Evaluation Q1 + alpha 0.05 | 38 | 38/38 (100%) | 38/38 / 38/38 / 38/38 |
| All unmodified Q1 candidates | 185 | 160/185 (86.5%) | 160/185 / 160/185 / 159/185 |

The 2/38 held-out clean-null rate is a modeled Gaussian-noise false alarm
fraction, not an empirically measured real-instrument false-positive rate. The
33/38 unmodified Q1 evaluation triggers are a candidate trigger fraction only;
without negative controls they cannot be interpreted as 33 false positives or
as evidence that the candidates contain subhaloes. At the q95 threshold, the
alpha 0.02 arm has 35/38 triggers and the alpha 0.05 arm 38/38. The unmodified
candidate arm already has 33/38, so these injection counts show little
separation between the fixed injections and real-candidate residuals.

The larger set changes the calibration q95 from 5.666623163 in the historical
153 run to 4.526957404, while held-out clean-null triggers remain 2 (2/29
historically, 2/38 now). Candidate triggers are 26/29 versus 33/38; alpha
0.02 recovery is 27/29 versus 35/38; alpha 0.05 recovery is 29/29 versus 38/38.
These fractions are similar within the small evaluation samples. The sample
expansion does not provide a useful real-data specificity estimate.

The 32-cutout run was also processed first, as required. Its 23 calibration
cutouts produced q95=0 because the empirical clean-null maxima were too
discrete at that sample size. That degenerate threshold is retained in
[`b6-new32/injection-results.summary.json`](b6-new32/injection-results.summary.json)
for transparency and is not used for the 185-cutout result.

## Dependence on integrated S/N

Integrated positive-flux S/N is the same proxy as in the historical report:
`sum(max(image,0))/sqrt(sum(noise_sigma**2))`. Its tercile edges (122.87,
215.38) come from calibration cutouts and are applied unchanged to evaluation.

| Evaluation S/N bin | n | Unmodified candidate triggers | alpha 0.02 | alpha 0.05 |
|---|---:|---:|---:|---:|
| Low | 8 | 6/8 | 6/8 | 8/8 |
| Middle | 10 | 7/10 | 9/10 | 10/10 |
| High | 20 | 20/20 | 20/20 | 20/20 |

The high-S/N bin is already fully triggered without an injection. The bins are
small and the unmodified candidate rate is substantial in every bin, so these
results do not establish a useful S/N-dependent sensitivity curve.

## Mass sensitivity and fit behavior

The runs inject two fixed values of `alpha_Rs_arcsec`; they do not sample
subhalo mass. This dataset and model do not supply the lens/source redshifts,
cosmology-dependent angular-diameter distances, or a calibrated mass-to-alpha
mapping needed to convert those strengths to physical mass. **There is no
defensible statement that this pipeline detects subhaloes down to a particular
mass.** The lower-mass limit is not measured here; the reported injection
levels are 0.02 and 0.05 arcsec in `alpha_Rs`, not solar masses.

At the known injected position `(0.3,0.3)`, the best scan point lands at the
true corner in 10/38 alpha-0.02 runs and 11/38 alpha-0.05 runs. The winning
scan alpha is 0 / 0.02 / 0.05 in 3 / 9 / 26 cases for alpha 0.02 and 0 / 0.02 /
0.05 in 0 / 4 / 34 cases for alpha 0.05. Thus the candidate fit does not
reliably recover even the injected grid coordinate or strength.

Non-converged grid points remain in the maximum statistic. Converged points
were: calibration clean null 1,215/1,764; evaluation clean null 305/456;
unmodified candidates 387/456; alpha 0.02 379/456; and alpha 0.05 376/456.
Every non-convergence reported the fixed 80-evaluation limit. This incomplete
optimizer behavior and the assumed noise/PSF model further limit these rates.

## Raw outputs and reproduction

Per-cutout maxima, all grid point values, convergence, boundaries and split
assignments are in [`b6-185/injection-results.jsonl`](b6-185/injection-results.jsonl);
the exact aggregate is in
[`b6-185/injection-results.summary.json`](b6-185/injection-results.summary.json).
The new32-only run is retained under [`b6-new32/`](b6-new32/).
The expanded source-complexity comparison is reported in
[`P2_185_SOURCE_STRUCTURE.md`](P2_185_SOURCE_STRUCTURE.md).

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/q1_153_injection.py \
  --b5-input docs/b6-185/real-robustness.jsonl \
  --output docs/b6-185/injection-results.jsonl
```
