# P1.5 real Q1 cutout fits

24 local FITS cutouts; every manifest row attempted. Results: `P1_REAL_REPORT.json`.

The adapter subtracts the median of the fixed outer 15% border and estimates constant sky noise as 1.4826*MAD there. It masks nonfinite pixels and values more than 8 sky sigmas below that median; positive source flux is preserved. No gain/RMS/exposure is supplied; noise is sky-only. Fit PSF is a circular Gaussian with FWHM 0.16 arcsec. DATA.md says this is an unverified approximate VIS width. Image units remain native_flux; amplitude is not an electron calibration. No lens-specific choices or noise/PSF tuning.

Both all-free and gamma~Normal(2,0.1) fits use the same bounded, seeded optimizer. chi2/dof uses only the image-data term; prior penalty is separately recorded. Reference table comparisons are against Euclid model estimates, not truth. Centre is excluded because its coordinate origin is undocumented. Ellipticity comparison uses q=(1-|e|)/(1+|e|); shear comparison uses magnitude.

| File | Shape | Noise sigma (native/pixel) | PSF FWHM | Free chi2/dof | Free converged | Prior chi2/dof | Prior converged | Bound contacts | Ref theta_E | Fit theta_E free | Ref shear | Fit shear free | Ref q | Fit q free |
|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 102019127_NEG598523367504401932_VIS.fits | 121x121 | 0.002945 | 0.16 | 8.539 | True | 8.722 | True | lens.theta_E,source.e1 | 1.403 | 0.4 | 0.147 | 0.04684 | 0.2951 | 0.3851 |
| 102019586_NEG564463213499964061_VIS.fits | 121x121 | 0.002292 | 0.16 | 36.21 | True | 44.58 | False | none | 1.159 | 0.4003 | 0.04207 | 0.1907 | 0.6465 | 0.2736 |
| 102019594_NEG620978815501656388_VIS.fits | 121x122 | 0.002405 | 0.16 | 3.658 | True | 3.658 | True | lens.theta_E,lens.gamma,lens.e1,lens.e2,source.R_sersic | 0.8707 | 0.4 | 0.3185 | 0.1568 | 0.3882 | 0.2222 |
| 102020055_NEG563894315496088157_VIS.fits | 122x121 | 0.002601 | 0.16 | 1.541 | True | 1.542 | True | none | 0.8119 | 0.4479 | 0.2985 | 0.09981 | 0.5633 | 0.4793 |
| 102020528_NEG554976975487853956_VIS.fits | 122x121 | 0.002681 | 0.16 | 37.1 | True | 46.05 | True | none | 0.9574 | 0.401 | 0.05152 | 0.1932 | 0.8603 | 0.2391 |
| 102020532_NEG584961967491933103_VIS.fits | 122x122 | 0.002324 | 0.16 | 3.401 | True | 3.512 | True | lens.theta_E | 0.5934 | 0.4 | 0.143 | 0.1723 | 0.4584 | 0.2862 |
| 102021497_NEG615658160478755691_VIS.fits | 122x121 | 0.002846 | 0.16 | 2.464 | True | 2.463 | True | none | 0.4627 | 0.4071 | 0.05917 | 0.2011 | 0.5957 | 0.3552 |
| 102021984_NEG609678648473892045_VIS.fits | 121x122 | 0.002987 | 0.16 | 18.43 | True | 17.81 | True | lens.theta_E | 0.6712 | 0.4 | 0.08162 | 0.197 | 0.5112 | 0.256 |
| 102023476_NEG610025395461180513_VIS.fits | 121x122 | 0.003451 | 0.16 | 16.94 | True | 16.94 | True | lens.gamma,lens.center_x,lens.center_y,lens.gamma1,source.R_sersic,source.e2,source.center_x | 1.427 | 0.5387 | 0.216 | 0.1867 | 0.8789 | 0.5978 |
| 102023481_NEG647245161459048846_VIS.fits | 121x121 | 0.003415 | 0.16 | 2.642 | True | 2.912 | True | lens.theta_E | 0.5002 | 0.4 | 0.1208 | 0.1717 | 0.8273 | 0.2474 |
| 102042914_NEG522625262291125533_VIS.fits | 122x121 | 0.00293 | 0.16 | 22.4 | True | 22.12 | True | lens.gamma | 0.7006 | 0.4002 | 0.08996 | 0.1424 | 0.7678 | 0.234 |
| 102044188_NEG548070624278595332_VIS.fits | 122x121 | 0.003018 | 0.16 | 29.87 | False | 31.24 | True | lens.gamma | 0.9221 | 0.4012 | 0.04561 | 0.125 | 0.4974 | 0.2376 |
| 102045468_NEG534525411271636586_VIS.fits | 121x121 | 0.002548 | 0.16 | 1.56 | True | 1.56 | True | lens.gamma,lens.center_x,lens.gamma1,source.e1,source.e2 | 0.7954 | 0.7311 | 0.05766 | 0.1678 | 0.4995 | 0.4703 |
| 102046759_NEG522096295262136465_VIS.fits | 121x122 | 0.006567 | 0.16 | 3.183 | True | 2.917 | True | none | 1.274 | 0.4648 | 0.07233 | 0.1474 | 0.7734 | 0.4586 |
| 102158275_2709777393645726835_VIS.fits | 122x122 | 0.002435 | 0.16 | 3.419e+04 | True | 3.419e+04 | True | none | 1.058 | 0.4021 | 0.1165 | 0.1123 | 0.8765 | 0.3174 |
| 102158893_2720583810655664894_VIS.fits | 121x121 | 0.0025 | 0.16 | 103.2 | True | 90.43 | True | lens.gamma | 0.592 | 0.4038 | 0.08791 | 0.1903 | 0.496 | 0.2657 |
| 102159189_2667596748662187959_VIS.fits | 121x121 | 0.002625 | 0.16 | 4.51 | True | 4.52 | True | none | 0.9069 | 0.4464 | 0.06834 | 0.1724 | 0.6151 | 0.5106 |
| 102159192_2695197869661844158_VIS.fits | 122x122 | 0.002593 | 0.16 | 2.422 | True | 2.73 | True | none | 1.023 | 0.4005 | 0.009059 | 0.1643 | 0.8735 | 0.4159 |
| 102159192_2699004780662092340_VIS.fits | 122x121 | 0.003055 | 0.16 | 199.2 | True | 200.6 | True | lens.e1 | 0.9399 | 0.4086 | 0.1283 | 0.1987 | 0.7123 | 0.2542 |
| 102159196_2750719383660442039_VIS.fits | 121x121 | 0.002238 | 0.16 | 2.636 | True | 2.635 | True | source.R_sersic | 1.102 | 0.5026 | 0.1822 | 0.1164 | 0.6112 | 0.3624 |
| 102159773_2677079122669151496_VIS.fits | 122x122 | 0.002746 | 0.16 | 112.6 | True | 107.1 | True | none | 1.437 | 0.4005 | 0.06205 | 0.1799 | 0.6378 | 0.2636 |
| 102159776_2707047371671953097_VIS.fits | 121x122 | 0.003116 | 0.16 | 2.094 | True | 2.098 | True | none | 0.7174 | 0.5356 | 0.07977 | 0.1046 | 0.7577 | 0.6876 |
| 102160335_2682061397679348286_VIS.fits | 122x122 | 0.003189 | 0.16 | 4.294 | True | 4.282 | True | none | 0.6585 | 0.4007 | 0.1791 | 0.1636 | 0.7407 | 0.2761 |
| 102160340_2740784519677600444_VIS.fits | 122x122 | 0.002247 | 0.16 | 1.893 | True | 1.812 | True | none | 0.8173 | 0.4195 | 0.03612 | 0.04978 | 0.6744 | 0.3164 |

## free: summary

Finite: 24/24; converged: 23/24; bound contact: 12/24; median chi2/dof: 4.402.
- theta_E fit-minus-reference median: -0.4656; compared 24. Reference estimates are not truth.
- shear magnitude fit-minus-reference median: 0.06136; compared 24. Reference estimates are not truth.
- q fit-minus-reference median: -0.2704; compared 24. Reference estimates are not truth.

## gamma_prior_sigma_0.1: summary

Finite: 24/24; converged: 23/24; bound contact: 11/24; median chi2/dof: 4.401.
- theta_E fit-minus-reference median: -0.4643; compared 24. Reference estimates are not truth.
- shear magnitude fit-minus-reference median: 0.07675; compared 24. Reference estimates are not truth.
- q fit-minus-reference median: -0.2766; compared 24. Reference estimates are not truth.

## Limitations and failures

Fit failures, status messages, estimates and mask/noise metadata are retained per file in the JSON. These fits use a single-Sersic source and omit foreground lens light, and are exploratory, not science validated. No formal likelihood interpretation is possible until native image units and variance are calibrated. Reference comparisons are rough model-to-model checks only.
