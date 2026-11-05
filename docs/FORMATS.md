# Provisional task and result formats

These local formats are for the P3 prototype. They are not yet a BOINC wire
contract; check in a project-supplied BOINC API header and validator tolerance
before claiming that status.

## `skia-task-v1` input ZIP

The ZIP has exactly four stored members in stable order and timestamps:

1. `manifest.json`: `{"format":"skia-task-v1","files":{...}}`, with SHA-256
   digests for the other three members.
2. `baseline.json`: `skia-fit-v1` smooth-fit parameters and optimizer settings.
3. `cutout.npz`: `skia-cutout-v1` image, positive noise standard deviations,
   boolean usable-pixel mask, normalized fine PSF and observation metadata.
4. `grid.yaml`: TNFW angular `Rs_arcsec`, `r_trunc_arcsec`, x/y coordinate axes,
   and `alpha_Rs_arcsec` strength axis. This is not a mass grid.

The reader rejects duplicate/extra member names, oversized files, wrong schema
and digest mismatches. Create a reproducible archive with:

```sh
skia-bundle cutout.npz fit.json grid.yaml -o task.zip
```

The bundler canonicalizes `cutout.npz` to exactly `image`, `noise_map`, `mask`,
`psf_fine` and `observation_json`. Extra arrays such as simulator-only noiseless
`model` or `truth_json` are stripped before they enter a task archive. Use
`skia.cutout.save_cutout` to serialize a `Cutout` object directly, including one
returned by the Q1 FITS adapter.

## `skia-scan-v1` output

UTF-8 text, fixed 10 decimal places, no timings, with a version/profile header
and columns:

```text
# skia-scan-v1 profile=spherical-TNFW mass=not-calibrated
x_arcsec y_arcsec alpha_Rs_arcsec delta_chi2 chi2 converged
```

Rows use Cartesian order x, then y, then alpha_Rs. `delta_chi2` is baseline data
chi-square minus candidate data chi-square; it is not a calibrated significance.

## Checkpoint

`skia-app` writes local `skia-checkpoint-v1` JSON atomically after each grid
point (a warm fit establishes the baseline first). It includes a fingerprint of
all cutout arrays, grid, baseline and optimizer config. Re-running the command
with the same checkpoint resumes remaining grid points; changed inputs are
rejected. The checkpoint is an implementation detail, not a result artifact.

The prototype emits `PROGRESS completed total` records on stderr. These records
are a local CLI progress hook, not BOINC API calls. A project header is still
needed to wire correct BOINC progress and checkpoint functions.
