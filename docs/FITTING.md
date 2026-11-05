# Smooth fitting (task 1.3)

`skia-fit cutout.npz -o fit.json` fits an EPL/SPLE plus Cartesian shear lens with
one elliptical Sersic source on a single lens plane. The default has 15 free
parameters; experimental `--source-components 2` adds a second Sersic with
shared ellipticity and center, for 18 free parameters.
herculens supplies the profiles; a separate differentiable JAX renderer performs
padded subpixel PSF convolution and detector integration. The renderer and all
15 Jacobian columns are tested against the simulator/central finite differences.

## Input: skia-cutout-v1

NPZ is opened with `allow_pickle=False`. Images may be rectangular. Required fields:

| Field | Meaning |
|---|---|
| image | Sky-subtracted image in metadata's `electron` or `native_flux` unit |
| noise_map | Positive standard deviation in the same image unit, same shape |
| mask | Boolean, same shape; True pixels enter the fit |
| psf_fine | Nonnegative unit-sum odd-sized square PSF kernel on the subpixel grid |
| observation_json | JSON string: format=`skia-cutout-v1`, image_units=`electron` or `native_flux`, pixel_scale in arcsec/pixel, exposure in seconds, integer oversample |

The PSF centre represents zero offset; render scale is pixel_scale/oversample.
Geometry, PSF and noise are fixed supplied calibration inputs, never fitted.
Masking may exclude NaN image/noise values; used pixels must be finite with positive
noise. There must be more than 15 usable pixels. The loader rejects absent headers
and legacy files rather than retrieving geometry from `truth_json`. `model`,
`truth_json` and simulation seeds are ignored. No true lens/source parameters
enter initialization, residuals or optimizer decisions. A test corrupts the
truth/model fields and obtains the identical fit.

`save_simulation` supplies the independent metadata and exact fine-grid PSF. The
original coarse `psf` remains an informational preview and is not read by the
fitter. The Q1 adapter retains native flux, estimates sky/noise from an outer
border and supplies an unverified Gaussian PSF; it does not provide calibrated
units or a measured PSF. See DATA.md and P1_REAL_REPORT.md.

## Likelihood and search

Residuals are `(model-image)/noise_map` for usable pixels. Minimize their squared
sum (Gaussian likelihood with fixed supplied variance). Poisson noise is only
approximated by this likelihood. A synthetic noise map is computed from expected
counts, which is an idealized known-variance assumption, not observational noise
estimation. Lens light, backgrounds, source complexity, correlated noise and PSF
uncertainty are not fitted. This is a point estimate, not Bayesian evidence or
pixelated-source inference. No covariance/confidence-interval claim is made.

All fits share these bounds, with no per-lens tuning:

| Parameter | Lower | Upper | Unit |
|---|---:|---:|---|
| Lens theta_E | 0.4 | 1.6 | arcsec |
| Lens gamma | 1.6 | 2.4 | dimensionless 3D slope |
| Lens e1, e2 | -0.45 | 0.45 | dimensionless, each |
| Lens centre x, y | -0.2 | 0.2 | arcsec, each |
| Shear gamma1, gamma2 | -0.15 | 0.15 | dimensionless, each |
| Source amp | 0.01 | 1000 | e-/s/arcsec² |
| Source R_sersic | 0.03 | 0.8 | arcsec, semi-major axis |
| Source n_sersic | 0.6 | 4 | dimensionless |
| Source e1, e2 | -0.45 | 0.45 | dimensionless, each |
| Source centre x, y | -0.4 | 0.4 | arcsec, each |

Amplitude and radius use natural logs internally. Start 0 uses the fixed default
Lens()/Source() profiles from the simulator, independent of data/truth. Later
starts perturb this same baseline with a PCG64 generator, fixed default seed 731
and scales declared in fitter.py. Defaults: 2 starts, 200 function evaluations per
start, SciPy trust-region reflective method, `x_scale=jac`, all tolerances 1e-8.
CLI options `--starts`, `--seed`, `--max-nfev` apply to the whole fit. JAX uses
float64; CLI defaults to CPU unless JAX_PLATFORMS is explicitly set externally.

Every start's status, chi2 and evaluations are retained. Select the lowest finite
chi2, breaking ties by start index, even if its optimizer did not converge. Return
that convergence state explicitly. If every start raises a numerical error, fail
with an explanation. A parameter is flagged at a boundary when its fractional
distance to either search limit is less than 1e-4. Finite convergence or an
acceptable chi2 does not establish parameter recovery or uniqueness.

## Output and timing

`skia-fit-v1` JSON uses sorted keys, fixed 10-decimal numeric values and a trailing
newline. It includes physical parameters, statistic/dof, convergence, boundary
flags, `parameter_errors_1sigma`, covariance rank, chosen-start evaluations, all
attempts and optimizer configuration. Errors are from a chi2/dof-scaled local
Jacobian covariance; log-amplitudes/radii use a delta-method conversion. Rank
deficiency emits null errors. Coverage is only approximate and undercovers some
parameters in the matched simulation study (B3_COVERAGE.md); do not treat these
as calibrated confidence intervals. Timing is omitted to preserve byte
reproducibility. stderr records elapsed_seconds for
all starts, including JAX compilation on a cold geometry, excluding file I/O and
interpreter imports. The caller must preserve input and software versions for
provenance. Output precision is a serialization policy, not an accuracy guarantee.

Tests check independent-process byte equality in this local environment. Different
CPUs/OS and GPU agreement, validator tolerance and portable builds are outstanding.

### Optional slope prior

`FitConfig(gamma_prior_sigma=sigma)` and CLI `--gamma-prior-sigma sigma` add a
Gaussian penalty `(gamma-2)^2/sigma^2` to the optimizer residual. The JSON retains
the image-data chi2 and prior penalty separately; reported chi2/dof excludes the
prior term. The free-slope fit remains the default. This option is used for a
separate proposed scan baseline, not to overwrite the free-slope report.

The cutout carries its image-unit label into fit JSON. Synthetic output uses
`electron`; the Q1 FITS adapter retains `native_flux` because exposure and BUNIT
are absent. Q1 noise is estimated from the outer-border MAD and its Gaussian
0.16 arcsec PSF is unverified; see [DATA.md](DATA.md) and
[the real-cutout fit report](P1_REAL_REPORT.md). Accordingly, the reported
Sersic amplitude is `<image_unit>/s/arcsec^2`; for Q1 this is a fitting
normalization, not a calibrated physical flux rate.

The two-component experimental option adds `source2.amp`, `source2.R_sersic`,
and `source2.n_sersic`; its ranges are 0.001–1000 image-unit/s/arcsec²,
0.01–0.8 arcsec, and 0.6–4. It shares the first component's ellipticity and
center. On the tested paired simulation study it did not improve scan false
positives or injection recovery and had lower fit convergence; see
[B2_SOURCE_MODEL.md](B2_SOURCE_MODEL.md). It is not a Q1-validated replacement.
