# Developer usage

Run these commands from the repository root after the editable install in the
README. All examples use local simulated data and write generated artifacts to
ignored `work/`.

## Simulate and fit

```sh
mkdir -p work
JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
from skia.simulator import Observation, simulate, save_simulation
save_simulation(simulate(observation=Observation(npix=40), seed=42), "work/lens.npz")
PY
.venv/bin/skia-fit work/lens.npz --gamma-prior-sigma 0.1 -o work/fit.json
```

The fit is an optimizer result with explicit convergence and bound flags. It is
not a detection-significance calibration. Input units and fit assumptions are
described in [FITTING.md](FITTING.md).

## Make and run a task bundle

Create `work/grid.yaml` with angular coordinates and a TNFW strength grid:

```yaml
profile: TNFW
Rs_arcsec: 0.2
r_trunc_arcsec: 0.8
x_arcsec: [-0.2, 0.2]
y_arcsec: [-0.2, 0.2]
alpha_Rs_arcsec: [0.0, 0.02, 0.05]
```

Then package and run it:

```sh
.venv/bin/skia-bundle work/lens.npz work/fit.json work/grid.yaml -o work/task.zip
.venv/bin/skia-app work/task.zip -o work/result.txt
```

The app writes fixed-precision `skia-scan-v1` text and a local checkpoint next
to the output. Re-run the same command after interruption to resume completed
grid points; changing any input causes a checkpoint fingerprint error. Use
`--checkpoint path.json` to choose another checkpoint path. `PROGRESS n total`
lines go to stderr. They are not BOINC API progress calls.

Grid strengths are `alpha_Rs_arcsec`, not masses in `Msun`. Output
`delta_chi2` is a model comparison only. Do not interpret it as a calibrated
detection or significance. The P2 study found no useful mass threshold and
high Q1 false-positive rates; see [P2_REPORT.md](P2_REPORT.md).

## Verify locally

```sh
make test
```

Tests establish code-level behavior in the current pinned environment. They do
not establish scientific validation, cross-device determinism, old-glibc
compatibility, or BOINC readiness.
