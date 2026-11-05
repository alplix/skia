# Simulated recovery protocol v1 (task 1.4)

Defined before the first population run, 2026-10-31. Run 200 independently sampled
smooth lenses, with no subhaloes or real data. Do not reject or replace any draw.
Use exactly the same bounded fitter policy for every lens: FitConfig(starts=2,
seed=731, max_nfev=200). No truth-based initialization, warm starts between lenses,
per-lens tuning or refits based on known errors. Keep every outcome, including
numerical errors, iteration exhaustion and boundary contacts.

Parameter generator: NumPy PCG64 seed 20261031, independent uniform draws in the
order Lens then Source, following dataclass field order in the script's explicit
range dictionaries. Noise seed for lens i (zero-based) is 100000+i.

| Parameter | Uniform range | Unit |
|---|---|---|
| Lens theta_E | 0.85, 1.15 | arcsec |
| Lens gamma | 1.8, 2.2 | dimensionless |
| Lens e1 | 0.05, 0.20 | dimensionless |
| Lens e2 | -0.08, 0.08 | dimensionless |
| Lens centre x, y | -0.03, 0.03 | arcsec, each |
| Shear gamma1, gamma2 | -0.03, 0.05 | dimensionless, each |
| Source amp | 4, 18 | e-/s/arcsec² |
| Source R_sersic | 0.13, 0.23 | arcsec |
| Source n_sersic | 0.8, 1.6 | dimensionless |
| Source e1 | -0.12, 0.20 | dimensionless |
| Source e2 | -0.15, 0.15 | dimensionless |
| Source centre x, y | -0.12, 0.12 | arcsec, each |

All recovery images: 40 x 40 pixels, 0.1 arcsec/pixel, 565 s exposure, Gaussian
PSF FWHM 0.18 arcsec, oversampling 2, sky 0.1 e-/s/pixel, read noise 4.5 e-/pixel.
The complete image is usable. noise_map is the ideal known expected standard
deviation. Simulator and fitter use the same analytical profiles and PSF, so this
is a matched-model test, not an independent scientific cross-check.

Bias = mean(fitted minus true); scatter = population standard deviation of the
same errors (ddof=0); RMSE and 16th/84th percentiles supplement those numbers.
Report all finite returned fits, plus a second table restricted to converged fits
with no boundary flags. Optimization failure = exception or selected best fit not
converged. Count boundary contacts separately, including optimizer-converged ones.
A predeclared gross-recovery flag is |theta_E error| > 0.03 arcsec OR |gamma error|
> 0.15. This heuristic flags accuracy problems; it is not a confidence interval or
a significance/detection threshold. Report its rate over all draws; hard failures
count as gross failures. Give the finite-fit denominator explicitly.

Timing: record all-start fit wall time (compilation included on the first unseen
geometry), excluding imports, data generation and I/O. Also fit 5 synthetic default
100 x 100 images, noise seeds 200000+i, to time the default VIS-like shape; record
first-fit and subsequent-fit medians separately. These 5 are not part of recovery
statistics. CPU JAX is used; numerical-library thread environment is recorded.
No one-core, GPU, cross-platform or volunteer-throughput claim follows.

Reproduce with `JAX_PLATFORMS=cpu .venv/bin/python tests/bench/recovery.py`.
Outputs are raw JSONL, machine-readable protocol/summary, a diagnostic figure and
P1_REPORT.md. The default run is 200 lenses; smaller --count values are smoke tests
and must be labelled accordingly. Repeating or changing the protocol must use a
new output folder and retain prior outcomes.
