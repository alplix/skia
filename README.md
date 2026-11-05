# Skia

**Skia** (σκιά, "shadow") searches for dark-matter subhaloes in Euclid images of strong gravitational
lenses. A lens bends a background galaxy into arcs; a small dark clump near an arc perturbs it by a tiny
amount. Skia fits a smooth lens model first, then scans a grid of subhalo positions and strengths and
records how much each extra clump improves the fit. Every lens is independent and costs a lot of CPU
time, which makes it a good volunteer-computing job. It runs as a beta application on the volunteer
project bitboinc (https://bitboinc.athena.org.tr).

## Why

Euclid will find about 10,000 lens candidates in its first data release and more than 100,000 in the
full survey. Forecasts for Euclid VIS (O'Riordan et al. 2023, arXiv:2211.15679) expect detections to be
rare, roughly one per 70 lenses at best, so the value is in a large, reproducible, injection-calibrated
search rather than in any single lens. Classical lens modelling costs hours of CPU per lens, which a
volunteer grid can absorb.

## Method

- **Smooth model:** elliptical power-law mass plus external shear, and a Sersic source, fitted with JAX
  (float64, CPU) on top of herculens/lenstronomy (`src/skia/fitter.py`).
- **Scan:** at every grid point a spherical truncated NFW (TNFW) subhalo of strength `alpha_Rs_arcsec`
  is added, the smooth parameters are refitted, and `delta_chi2` (baseline chi2 minus perturbed chi2)
  is recorded (`src/skia/scanner.py`). Results are fixed-precision text (`skia-scan-v1`).
- **Data:** public Euclid Quick Release 1 (Q1) VIS cutouts of grade-A lens candidates. The cutouts are
  not stored in this repository (`data/` is ignored); terms and limits are in [docs/DATA.md](docs/DATA.md).
- **Tasks:** one lens is split into six bundles of 66 grid points (`skia-task-v1` zip: cutout, baseline
  fit, grid). A task takes about 22 minutes on one core.

## Status and limitations

- `alpha_Rs_arcsec` is an angular lensing strength, **not a mass**. Euclid Q1 does not provide the
  per-lens redshifts and calibration needed to convert it to solar masses.
- `delta_chi2` is an image-fit improvement, **not a calibrated significance**. The P2 study found that
  the real Q1 cutouts produce large improvements even without an injected subhalo, so no detection
  cutoff is supported yet. Read [docs/P2_REPORT.md](docs/P2_REPORT.md) before interpreting any scan.
  This is not evidence that Euclid VIS is insensitive to every subhalo mass.
- The volunteer application is a beta and its results are exploratory.

## Volunteer application (beta)

- Linux x86_64 (glibc 2.31 or newer), one thread, about 22 minutes per task, two matching results
  needed before a task is accepted. Same-platform results are byte-identical; the cross-platform
  difference measured so far is 1.5e-12 (relative), checked with a 1e-8 tolerance.
- It is packaged for the volunteer client with the standard BOINC wrapper. See
  [boinc](boinc/README.md) for the wrapper notes, `job.xml`, templates and the task builder
  `make_tasks.py`, and [docs/P3_PACKAGING.md](docs/P3_PACKAGING.md) for the packaging notes.
- Not done yet: Windows and macOS builds, a second CPU family, license clearance of all bundled
  libraries ([docs/LICENSES.md](docs/LICENSES.md)).

## Development setup

The pinned environment was tested on CPython 3.14.1, macOS arm64, CPU JAX in float64. It is not a
portable release environment.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install pip==25.3
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
make test
```

Simulate, fit, bundle and scan locally:

```sh
.venv/bin/skia-fit work/lens.npz --gamma-prior-sigma 0.1 -o work/fit.json
.venv/bin/skia-bundle work/lens.npz work/fit.json work/grid.yaml -o work/task.zip
.venv/bin/skia-app work/task.zip -o work/result.txt
```

The complete workflow is in [docs/USAGE.md](docs/USAGE.md); file contracts are in
[FORMATS](docs/FORMATS.md), [FITTING](docs/FITTING.md) and [SCANNING](docs/SCANNING.md).

## Repository

| path | content |
|---|---|
| `src/skia/` | fitter, scanner, task bundles, command-line tools (`skia-fit`, `skia-scan`, `skia-bundle`, `skia-app`) |
| `tests/` | unit tests, recovery tests, lenstronomy cross-checks, Q1 injection studies |
| `docs/` | method notes, reports, data and license notes |
| `boinc/` | volunteer-client packaging: wrapper notes, `job.xml`, templates, task builder |

## Reports

- [P1 real-cutout and simulated recovery results](docs/P1_REAL_REPORT.md)
- [P2 injection, false-positive and S/N study](docs/P2_INJECTIONS.md), [decision report](docs/P2_REPORT.md)
- [P2 cost measurements](docs/P2_COST.md)
- [Determinism and packaging](docs/P3_DETERMINISM.md), [docs/P3_PACKAGING.md](docs/P3_PACKAGING.md)

## License

Copyright (c) 2026 Alperen Yavuz. All rights reserved. See [LICENSE](LICENSE). Third-party components
keep their own licenses ([docs/LICENSES.md](docs/LICENSES.md)).
