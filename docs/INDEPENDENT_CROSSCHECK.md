# Independent forward-model cross-check with lenstronomy

## Method

Five fixed-seed EPL+shear/Sersic parameter draws were evaluated by both the
Herculens-backed Skia simulator and lenstronomy 1.13.2. The comparison holds
all physical parameters fixed; it does not optimize either implementation.
Lenstronomy uses `LensModel(['EPL', 'SHEAR'])` and
`LightModel(['SERSIC_ELLIPSE'])` with `sersic_major_axis=True`, matching the
half-light-radius convention used by Herculens. Both models are evaluated on
the same oversampled coordinate grid. The lenstronomy brightness is convolved
with the same normalized Gaussian PSF and integrated to the detector pixels.

Reproduce with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python \
  tests/bench/crosscheck_lenstronomy.py --count 5 \
  --output docs/lenstronomy-crosscheck
```

Protocol, sampled parameters and individual comparisons are in
[`lenstronomy-crosscheck/`](lenstronomy-crosscheck/).

## Results

| Quantity | Median discrepancy | Maximum discrepancy |
|---|---:|---:|
| EPL+shear deflection RMS (arcsec) | 2.45×10⁻¹⁶ | 3.23×10⁻¹⁶ |
| Absolute deflection component (arcsec) | 1.55×10⁻¹⁵ | 1.55×10⁻¹⁵ |
| Source surface-brightness relative RMS | 1.31×10⁻¹⁶ | 1.38×10⁻¹⁶ |
| Integrated noiseless image relative RMS | 9.18×10⁻¹⁶ | 1.80×10⁻¹⁵ |
| Maximum image difference / peak | 1.33×10⁻¹⁵ | 3.86×10⁻¹⁵ |

The implementations agree to floating-point precision on these same-parameter
forward renders. This verifies profile conventions and the Skia forward-model
path for these inputs.

## Independent fit cross-check

To compare the optimizers as well, five new seeded noisy simulated lenses were
fit independently with Herculens/JAX and lenstronomy 1.13.2 plus SciPy's
finite-difference `least_squares`. Both methods received the exact same image,
noise map, mask, parameter bounds, deterministic initial vectors, stopping
criteria, and two-start policy. The fit comparison uses a 32×32 image, fixed
Gaussian PSF and simulator noise; all five fits converged in both methods.

| Difference (Herculens minus lenstronomy; absolute value) | Median | Maximum |
|---|---:|---:|
| Einstein radius (arcsec) | 4.51×10⁻¹⁰ | 3.01×10⁻⁹ |
| Power-law slope γ | 2.78×10⁻⁸ | 1.97×10⁻⁷ |
| Shear magnitude | 1.50×10⁻⁹ | 1.23×10⁻⁸ |
| Lens axis ratio q | 9.11×10⁻⁹ | 2.89×10⁻⁸ |
| Data χ² | 3.82×10⁻¹¹ | 1.17×10⁻¹⁰ |

Individual fit records and the protocol are in
[`lenstronomy-fit-crosscheck/`](lenstronomy-fit-crosscheck/). Reproduce with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true .venv/bin/python \
  tests/bench/lenstronomy_fit_crosscheck.py --count 5 --max-nfev 200 \
  --npix 32 --output docs/lenstronomy-fit-crosscheck
```

This is a small controlled implementation cross-check, not evidence of
accuracy on real Q1 cutouts, source-model adequacy, calibrated uncertainty, or
subhalo sensitivity. The shared forward-model conventions and noise are
deliberately idealized; five simulated fits cannot establish general optimizer
equivalence or validate the scientific model.
