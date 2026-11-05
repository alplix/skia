# B5 robustness pass on 153 Q1 cutouts

## Fixed protocol

All 153 `*_VIS.fits` files in `data/q1_cutouts/` were processed in sorted
filename order and matched to all 153 rows of `manifest.csv`. Every image used
the same outer-15%-border MAD sky/noise estimate, constant sky-only noise map,
assumed circular Gaussian PSF with FWHM 0.16 arcsec, one optimizer start,
seed 731, gamma prior `Normal(2, 0.1)`, and `max_nfev=80`. The same 12-point
grid was used for every image: x,y in {-0.3,+0.3} arcsec and
`alpha_Rs_arcsec` in {0, 0.02, 0.05}, with `Rs=0.2` arcsec and truncation
radius 0.8 arcsec. No per-image settings changed.

The complete per-cutout baseline, adapter metadata, reference comparison,
best scan result, every unconverged grid coordinate, boundary contact and
optimizer message are retained in
[`b5-153-real-robustness.jsonl`](b5-153-real-robustness.jsonl). The aggregate
failure index is
[`b5-153-real-robustness.summary.json`](b5-153-real-robustness.summary.json).

## Robustness results

| Measure | Result |
|---|---:|
| Cutouts / adapter hard errors | 153 / 0 |
| Baseline converged | 131/153 |
| Baseline stopped at `max_nfev` | 22/153 |
| Scan fit points converged | 1,569/1,836 |
| Scan fit points stopped at `max_nfev` | 267/1,836 (14.5%) |
| Cutouts with at least one failed scan point | 92/153 |
| Baseline fits with bound contacts | 54/153 |
| Grid fits with bound contacts | 572/1,836 |
| Median baseline image chi-square/dof | 13.142 |
| Baselines with chi-square/dof > 10 / > 100 | 83/153 / 22/153 |
| Maximum baseline chi-square/dof | 365,359.7 |

All 22 baseline non-convergences and all 267 grid-point non-convergences have
status 0, `nfev=80`, and the same message: “The maximum number of function
evaluations is exceeded.” No input, FITS, adapter, or grid hard errors occurred.
This identifies the immediate optimizer failure mode as evaluation-budget
exhaustion. It does not establish whether increasing the budget would improve
the fit or make its scientific interpretation reliable.

The 22 cutouts with baseline chi-square/dof above 100 are:

| Cutout | chi-square/dof |
|---|---:|
| `102019588_NEG581189290502362418_VIS.fits` | 5,561.1 |
| `102019595_NEG630246532498667881_VIS.fits` | 13,388.8 |
| `102020061_NEG607087127495633316_VIS.fits` | 7,728.7 |
| `102021980_NEG576667803474825283_VIS.fits` | 107.3 |
| `102021981_NEG586493481476452509_VIS.fits` | 165.1 |
| `102021988_NEG640289253474412348_VIS.fits` | 226.2 |
| `102043549_NEG530257443283430270_VIS.fits` | 110,390.6 |
| `102044186_NEG535621262280129063_VIS.fits` | 114.4 |
| `102157952_2658211530641487553_VIS.fits` | 5,603.5 |
| `102157955_2685490710638126805_VIS.fits` | 365,359.7 |
| `102158273_2697192667642722765_VIS.fits` | 184,598.7 |
| `102158275_2709777393645726835_VIS.fits` | 34,201.0 |
| `102158584_2687900321650859511_VIS.fits` | 256.8 |
| `102158585_2690666515648268601_VIS.fits` | 102.3 |
| `102158886_2641867744653343162_VIS.fits` | 194.8 |
| `102158894_2733212462655784384_VIS.fits` | 161.9 |
| `102159189_2663231466659368059_VIS.fits` | 1,100.3 |
| `102159192_2699004780662092340_VIS.fits` | 200.9 |
| `102159488_2717571775663564534_VIS.fits` | 349.6 |
| `102159773_2677079122669151496_VIS.fits` | 130.3 |
| `102159777_2716319161668542331_VIS.fits` | 5,038.5 |
| `102160610_2729256016684457774_VIS.fits` | 729.5 |

These are large residual discrepancies under the present model/noise/PSF
assumptions. The available data cannot separate source morphology, PSF error,
correlated or misestimated noise, and other model misspecification as causes.
The 0.16 arcsec PSF is assumed, not measured, and the cutouts supply no RMS,
gain, or exposure calibration. Bound contacts indicate fit-domain pressure,
not a measured physical constraint.

## Comparison with Euclid reference parameters

The Euclid columns are another model's estimates, not ground truth. For
ellipticity magnitude `e`, axis ratio is compared using `q=(1-|e|)/(1+|e|)`;
shear comparison uses `hypot(gamma1,gamma2)`. Differences below are fit minus
reference over all 153 finite outputs, including 22 optimizer non-converged
baselines. The converged-only spread was similar and is included in the raw
per-cutout file.

| Fit minus reference | Median | Median absolute difference | RMSE | p16–p84 |
|---|---:|---:|---:|---:|
| Einstein radius (arcsec) | -0.416 | 0.418 | 0.503 | [-0.687, -0.199] arcsec |
| Shear magnitude | +0.0285 | 0.0661 | 0.0918 | [-0.0669, +0.1015] |
| Axis ratio q | -0.308 | 0.308 | 0.363 | [-0.501, -0.109] |

The fit/reference disagreement is broad, with systematic low theta_E and q in
the median comparison. This is a model-to-model discrepancy, not a parameter
accuracy measurement. No centers or position angles are compared.

## Interpretation and reproduction

The fixed pipeline processes all cutouts, but its large residuals, high bound
contact rate, and optimizer non-convergences prevent interpreting the raw scan
as a calibrated detection statistic. These candidate-only data contain no
non-lens negative controls, so the unmodified-candidate trigger fraction is
not a clean false-positive rate. The scan does not support a subhalo mass
threshold; `alpha_Rs_arcsec` is an angular strength without a redshift/cosmology
conversion.

Reproduce the run with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/real_robustness.py \
  --output docs/b5-153-real-robustness.jsonl
```

