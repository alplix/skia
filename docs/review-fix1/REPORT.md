# Finding 1: constrained-slope baseline

200 paired simulated lenses; same image/noise realization fit by both methods.
The constrained method uses a predeclared Gaussian prior gamma ~ Normal(2, 0.1), adds its penalty to the optimizer objective, and reports data chi-square separately. The free-slope run remains unchanged. No input or draw was selected by outcome.

| Metric | Free gamma | Prior sigma=0.1 |
|---|---:|---:|
| converged | 200 | 200 |
| boundary | 10 | 3 |
| theta_E_gross | 1 | 1 |
| gamma_gross | 32 | 11 |
| median_chi2_dof | 0.9991666284304224 | 0.9993049117222986 |

## lens.theta_E

| Statistic | Free | Prior sigma=0.1 |
|---|---:|---:|
| bias | 0.0006400937 | 0.0007670883 |
| scatter | 0.005440111 | 0.005152345 |
| rmse | 0.005477639 | 0.005209135 |

## lens.gamma

| Statistic | Free | Prior sigma=0.1 |
|---|---:|---:|
| bias | -0.008414951 | -0.02434869 |
| scatter | 0.1243585 | 0.07884773 |
| rmse | 0.1246429 | 0.08252166 |

## shear_magnitude

| Statistic | Free | Prior sigma=0.1 |
|---|---:|---:|
| bias | 0.006815984 | 0.00586519 |
| scatter | 0.02325415 | 0.02174219 |
| rmse | 0.02423248 | 0.0225194 |

## lens.gamma1

| Statistic | Free | Prior sigma=0.1 |
|---|---:|---:|
| bias | -0.003871058 | -0.002631406 |
| scatter | 0.02215469 | 0.020944 |
| rmse | 0.02249034 | 0.02110866 |

## lens.gamma2

| Statistic | Free | Prior sigma=0.1 |
|---|---:|---:|
| bias | 0.001431395 | 0.001445335 |
| scatter | 0.02056109 | 0.01923144 |
| rmse | 0.02061085 | 0.01928568 |

Full paired fits and attempt outcomes are in `paired.jsonl`; summary values are in `summary.json`.

Theta_E RMSE decreases 4.9% (0.005478 to 0.005209 arcsec) and shear-magnitude RMSE decreases 7.1% (0.02423 to 0.02252). Gamma gross-error flags fall from 32/200 to 11/200, while gamma bias magnitude increases from 0.0084 to 0.0243. A slope prior changes the statistical model and can bias gamma; the prior result does not replace the free fit. It is a proposed stable scan baseline only. Synthetic images use matched profiles and ideal known noise; this is not real-data calibration or a scientific cross-check.
