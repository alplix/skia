# P2 scan feasibility report

## Decision

**This prototype supports no useful Euclid VIS subhalo-mass detection threshold.**
The tested scan statistic is the maximum data `delta_chi2` over the specified
position/strength grid. The controlled simulation study has weak recovery at an
empirical 95th-percentile null cutoff, and the real Q1 cutouts have very large
null improvements from the model's residual structure. Positive injections are
not separated reliably from those nulls. No operational detection cutoff is
selected and no candidates should be called detections from this result.

The strongest limitation is that `alpha_Rs_arcsec` is only an angular lensing
strength. The Q1 inputs do not provide the per-lens redshifts/cosmology and
validated calibration required to convert it into `Msun`; therefore the result
cannot answer a numerical mass threshold. **No mass threshold is supported by
this method on this evidence.** This is not evidence that Euclid VIS is
fundamentally insensitive to every subhalo mass.

## Statistic and empirical cutoffs

For each image the scan refits smooth nuisance parameters at four positions and
three TNFW strengths, then records the maximum data `delta_chi2`. We compared
raw cutoffs 2, 5, and 10, and the empirical 95th percentile of each study's own
null sample. Those percentiles are in-sample diagnostics, not independently
validated false-alarm probabilities.

| Sample | Nulls | Null 95th percentile | Nulls at that cutoff | alpha 0.02 injections recovered | alpha 0.05 injections recovered |
|---|---:|---:|---:|---:|---:|
| Simulated | 20 | 3.068 | 1/20 | 1/20 | 2/20 |
| Q1 real | 24 | 148266.122 | 2/24 | 1/24 | 3/24 |

At the simpler raw cutoffs 2/5/10, Q1 false positives are 22/24 at every
cutoff, while both injected strengths appear recovered in 24/24 images. This
is not useful discrimination: the no-injection images trigger almost as often
as the injected images. Applying a simulation-derived cutoff to real data would
also fail to control the Q1 null tail. The raw data and detailed S/N strata are
in [`P2_INJECTIONS.md`](P2_INJECTIONS.md) and
[`p2-injection-results.json`](p2-injection-results.json).

## Expanded 153-cutout check

The same fixed B5 scan was subsequently run on all 153 supplied grade-A Q1
candidates. A tile-hash split held out 29 cutouts across 22 `tile_index` groups;
the remaining 124 cutouts across 100 tile groups calibrated an exploratory
threshold using matched smooth-lens simulations with the cutouts' estimated
noise amplitudes and assumed PSFs. The train-only clean-null q95 was
`delta_chi2=5.666623`; 2/29 held-out clean simulations crossed it. However,
26/29 unmodified held-out Q1 candidates crossed that threshold, as did 27/29
and 29/29 residual-preserving injections at alpha 0.02 and 0.05. The selected
images are lens candidates, not non-lens controls, so their trigger fraction is
not a false-positive rate. Several grid fits also failed to converge. Full
protocol, S/N breakdown and limitations are in
[`P2_153_INJECTIONS.md`](P2_153_INJECTIONS.md).

## Current 185-cutout result

The fixed pipeline was extended to all 185 grade-A candidates, using 147
calibration cutouts across 110 tiles and 38 evaluation cutouts across 28 other
tiles (zero tile overlap). The calibration-only modeled-clean-null q95 is
`delta_chi2=4.526957`. At the q95 and at each raw cutoff 2/5/10, matched clean
null triggers were 2/38 in evaluation, unmodified Q1 candidate triggers were
33/38, alpha 0.02 injection triggers were 35/38, and alpha 0.05 triggers were
38/38. On all 185 unmodified candidates the corresponding raw counts were
160/185, 160/185 and 159/185 at cutoffs 2, 5 and 10. Candidate triggers are
not false positives: this set has no non-lens controls. The modeled-null rate
is only for the assumed PSF and independent noise model, not an instrument
false-positive calibration. Relative to 153 cutouts, held-out null triggers
remain 2 (2/29 before), and candidate and injection trigger fractions are
similar (33/38 vs 26/29; alpha 0.02 35/38 vs 27/29; alpha 0.05 38/38 vs
29/29). These data do not support a useful sensitivity or specificity claim.

The calibration S/N tercile edges were 122.87 and 215.38 in the same integrated
positive-flux proxy. Held-out unmodified candidate trigger counts by low,
middle and high S/N were 6/8, 7/10 and 20/20; alpha 0.02 counts were 6/8, 9/10
and 20/20; alpha 0.05 counts were 8/8, 10/10 and 20/20. Each bin is small and
unmodified candidate triggers are already common. The two fixed angular
injection strengths do not map to solar masses, and no calibrated alpha-to-mass
mapping, per-lens redshifts or cosmology is supplied. **Plain-language mass
answer: this evidence does not show Euclid VIS support for detection down to
any particular subhalo mass; no mass limit was measured.** This is not a claim
that Euclid VIS is intrinsically insensitive to all subhalo masses.

Expanded injection rates, clean-null trigger counts at every threshold, S/N
tables and convergence are in [`P2_185_INJECTIONS.md`](P2_185_INJECTIONS.md).
The expanded source-structure analysis is in
[`P2_185_SOURCE_STRUCTURE.md`](P2_185_SOURCE_STRUCTURE.md); its unpenalized
two-Sersic gains do not identify what physically caused Q1 residuals. B6
baseline and scan failures and exact fixed configuration are documented in
[`B6_185_REAL_CUTOUTS.md`](B6_185_REAL_CUTOUTS.md).

The source-structure comparison fit a fixed shared-geometry two-Sersic model on
all 153 cutouts. Its unpenalized median chi-square gain was 7,533 (3.92% of the
one-Sersic baseline), versus 5,841 (2.79%) for the fixed-grid TNFW scan; the
two-Sersic gain was larger in 93/153 fits. But the two-Sersic optimizer
converged in only 88/153, and among those converged fits the median source gain
was 3,509 (1.81%) versus TNFW 8,379 (2.75%), with source gain larger in 44/88.
This does not establish source-dominated residuals: PSF/noise mismatch and
other model deficiencies can also be absorbed by the extra light component.
The source fraction of the real residual chi-square is not identifiable from
these data. See [`P2_153_SOURCE_STRUCTURE.md`](P2_153_SOURCE_STRUCTURE.md) and
the per-lens JSONL.

A separate controlled simulated source-complexity study compared one- and
two-Sersic fits on 40 paired two-component truths. Across raw Δχ² cuts 2/5/10,
the two-Sersic model had null trigger counts 10/5/2 versus 11/6/2 for one
Sersic. It recovered fewer injections at every tested angular strength and
converged less often. The extra source component is therefore still
experimental and is not promoted to the default fit. Full rates, uncertainty
intervals, protocol and model limits are in
[`B2_SOURCE_MODEL.md`](B2_SOURCE_MODEL.md).

The small simulated benchmark measured 0.0324 s of baseline-subtracted work
per grid point. The full B6 pipeline on 185 Q1 cutouts took 7,213.2 s total,
with 38.99 s mean and 35.22 s median per cutout for baseline plus 12 scan
points. CPU affinity was not enforced and optimization used multiple cores, so
this is not a verified one-core rate. JAX exposed no GPU device. See
[`P2_COST.md`](P2_COST.md) and [`B4_PERFORMANCE.md`](B4_PERFORMANCE.md).

## Limits

- The original injection study used 20 simulated nulls and 24 selected Q1
  lenses; the expanded study adds 29 held-out candidate lenses and 29 matched
  clean simulations, still too few for a reliable extreme-tail false-positive
  probability.
- Simulated lenses use the same simplified source and forward model as the fit.
  Their null residuals do not represent the structured real-Q1 residuals.
- Q1 injections preserve residuals by adding a best-fit smooth-model image
  difference. They are not independent physical image simulations and inherit
  the baseline source-model and assumed-PSF limitations.
- The Q1 Gaussian PSF is an unverified 0.16 arcsec assumption; noise is a
  constant outer-border MAD estimate without a supplied RMS map. Pixel
  correlation, lens light, sky structure and the true PSF are not calibrated.
- Real Q1 grid-point convergence was 253/288 null, 226/288 at alpha 0.02 and
  221/288 at alpha 0.05. Raw maxima retain nonconverged points; this makes
  significance less secure, not more.
- The alpha levels are angular profile strengths, not masses. There are no
  redshift-based or cosmology-based mass conversions in the inputs.

## Next work

P2 is recorded as a **no-go for a calibrated science search with this prototype**.
Per autonomous-mode instructions, P3 will continue as provisional packaging
work without presenting this prototype as a validated detection pipeline. The
injection study should be revisited after improved source modelling, validated
PSF/noise products and an explicit mass calibration are available.
