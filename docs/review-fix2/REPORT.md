# Finding 2: estimated noise and PSF mismatch

200 paired lenses. Every branch uses the gamma prior sigma=0.1.

The matched control uses oracle expected variance and true simulated PSF. The estimated branch uses the documented fixed outer-ring MAD and a +10% Gaussian PSF FWHM error. Noise and PSF changes are combined; causal contributions are not separable in this run.

| Outcome | Known variance / matched PSF | Estimated sky noise / PSF +10% |
|---|---:|---:|
| finite fits | 200 | 200 |
| converged | 200 | 199 |
| bound contacts | 3 | 2 |
| gross recovery flags | 8 | 14 |
| theta_E gross errors | 1 | 1 |
| gamma gross errors | 7 | 14 |
| median image chi2/dof | 1.000742287964116 | 1.1801154926627366 |

## lens.theta_E bias / scatter / RMSE

| Branch | Bias | Scatter | RMSE |
|---|---:|---:|---:|
| known_noise_matched_psf | 0.0004662343 | 0.004906702 | 0.004928803 |
| estimated_noise_psf_plus_10pct | -0.001288553 | 0.009252834 | 0.009342125 |

## lens.gamma bias / scatter / RMSE

| Branch | Bias | Scatter | RMSE |
|---|---:|---:|---:|
| known_noise_matched_psf | -0.0209347 | 0.07321601 | 0.07615015 |
| estimated_noise_psf_plus_10pct | 0.0006606466 | 0.08932879 | 0.08933123 |

## lens.gamma1 bias / scatter / RMSE

| Branch | Bias | Scatter | RMSE |
|---|---:|---:|---:|
| known_noise_matched_psf | -0.002248895 | 0.01986117 | 0.01998809 |
| estimated_noise_psf_plus_10pct | -0.008043039 | 0.0188607 | 0.02050406 |

## lens.gamma2 bias / scatter / RMSE

| Branch | Bias | Scatter | RMSE |
|---|---:|---:|---:|
| known_noise_matched_psf | 0.001162357 | 0.01577104 | 0.01581382 |
| estimated_noise_psf_plus_10pct | 0.001131288 | 0.01716474 | 0.01720198 |

## shear_magnitude bias / scatter / RMSE

| Branch | Bias | Scatter | RMSE |
|---|---:|---:|---:|
| known_noise_matched_psf | 0.004287354 | 0.01959563 | 0.02005917 |
| estimated_noise_psf_plus_10pct | 0.005650191 | 0.02045373 | 0.0212198 |

Median estimated image S/N: 4.75742.

The original fit output and optimizer-packed truth vectors are preserved in `paired.jsonl`. The summaries use physical source amplitude and radius values, recalculated without refitting from `paired-physical.jsonl`; all lens/shear fit outputs are unchanged. Machine-readable all-parameter statistics are in `summary.json`. Results describe a matched source family and one stated PSF perturbation, not a full PSF/noise uncertainty distribution.
