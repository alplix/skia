# Synthetic observation conventions (task 1.2)

`skia.simulator.simulate` ray traces a single lens plane using herculens' elliptical
power-law mass profile (SPLE/EPL) and external Cartesian shear. It evaluates an
elliptical Sérsic source, convolves with a circular Gaussian PSF on a subpixel grid,
then integrates detector pixels. The default field is 100 x 100 pixels at
0.1 arcsec/pixel; these are configurable synthetic assumptions, not a Q1 calibration.

Coordinates: image[y, x]; x increases rightwards and y with row number, with the
origin halfway between central pixels for even dimensions. There is no celestial
WCS or right-ascension handedness. Coordinates, Einstein radius, source half-light
radius, PSF FWHM and subhalo radii/deflections use arcsec. Ellipticity components
are `(1-q)/(1+q) * (cos(2 phi), sin(2 phi))`; phi is measured from +x towards +y.
EPL `gamma` is the 3D density slope (2 is isothermal), with upstream's circularized
Einstein-radius convention. Source `R_sersic` is the **semi-major-axis** half-light
radius; `amp` is e-/s/arcsec² at that radius. Upstream uses an approximate Sérsic
b_n and a small central smoothing scale. Very compact/high-index profiles require
an oversampling convergence study before inference.

The Gaussian is truncated at five sigma on the integration grid and normalized.
Rendering includes a halo of pixels outside the output field before convolution,
so the cutout boundary does not incorrectly discard incident light. Subpixels are
averaged and multiplied by detector area and exposure. The stored `psf` is a
unit-sum, detector-centre-sampled approximation; reproduce rendering with the
recorded FWHM and oversampling rather than convolving a coarse image with it.

`model` is expected source electrons after convolution; independent detector
counts are Poisson(model + sky_rate * exposure), plus Normal(0, read_noise), minus
the expected sky. Sky estimation uncertainty, lens light, correlated noise,
cosmic rays and detector nonlinearities are absent. `noise_map` is the expected
standard deviation `sqrt(model + sky_rate * exposure + read_noise**2)`, not an
estimate from the noisy data. Negative sky-subtracted pixels are allowed. The
mask contains True for all usable pixels; no observational masking is claimed.

Optional spherical subhaloes use lenstronomy's same-plane deflections:

- `TNFW`: scale radius Rs, untruncated NFW deflection normalization alpha_Rs,
  truncation radius r_trunc; smooth Baltz-type truncation, not a sharp cutoff.
- `PJAFFE`: central convergence sigma0, positive core Ra, outer scale Rs > Ra.
  This is a cored dual pseudo-isothermal profile; Ra can approach the singular
  pseudo-Jaffe limit but zero is rejected. Upstream small-radius regularization applies.

These are angular lensing parameterizations. No M200/concentration or total-mass
conversion is inferred without a cosmology and lens/source redshifts. In particular,
this simulator does not substantiate a subhalo sensitivity in solar masses.

`seed` selects a NumPy PCG64 generator; same pinned local environment and seed
produce identical arrays. `save_simulation` writes a pickle-free NPZ with image,
model, noise_map, psf, boolean mask and sorted JSON truth including the
`skia-simulation-v1` format header. This is an internal simulated input format;
New bundles also contain a separate truth-independent `observation_json` header
and `psf_fine` array for the smooth fitter (see FITTING.md). BOINC deterministic
fixed-precision text output and validator tolerance belong to P3.

Tests cover analytic circular isothermal deflections plus shear, zero lens and
zero subhalo limits, subhalo radial symmetry, positive PSF normalization/width,
Gaussian-source analytic flux, exposure/area scaling, blank-field noise moments,
seed reproducibility within and across local Python processes, both injection choices and safe serialization. These tests
are local correctness checks; further validation remains outstanding.
