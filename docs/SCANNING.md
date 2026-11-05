# Angular TNFW scans

`skia-scan fit.json cutout.npz --grid grid.yaml -o scan.txt` re-fits the smooth
source and lens parameters at every grid point while keeping a spherical TNFW
deflection field fixed. The grid uses angular coordinates and the TNFW
`alpha_Rs_arcsec` lensing-strength parameter; it does not represent a mass in
solar masses.

The default scan constructs deflection fields one point at a time. The optional
`--vectorized-fields` flag batches this preparation in groups of 16; the B4
profile found that path slower for the tested 27-point grid, so it is
experimental.

The YAML grid requires `profile: TNFW`, positive `Rs_arcsec` and
`r_trunc_arcsec`, and finite nonempty lists for `x_arcsec`, `y_arcsec`, and
`alpha_Rs_arcsec` (nonnegative). Coordinates are relative to the image center
used by the fitter. `delta_chi2` is baseline data chi-square minus the
perturbed model data chi-square, so positive values mean improved image fit.
The Gaussian likelihood uses the cutout's supplied noise map. The scan itself
does not calibrate significance; the exploratory false-positive and injection
results are in [P2_INJECTIONS.md](P2_INJECTIONS.md), and the P2 decision is in
[P2_REPORT.md](P2_REPORT.md).

Output is versioned fixed-precision text with one row per Cartesian grid point.
The full JSON result is available from the Python API. The Q1 noise/PSF remain
uncertain, source structure is simplified, and no physical mass conversion is
available. Euclid Q1 cutouts lack the redshifts and calibration needed here for
an `Msun` conversion.
