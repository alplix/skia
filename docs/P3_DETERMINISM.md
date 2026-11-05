# P3.2 determinism check

The provisional regression test creates the same `skia-task-v1` input twice,
runs `skia-app` twice in independent scans on this same local macOS arm64 /
Python 3.14.1 environment, and asserts the result text bytes are identical.
It passed as part of `make test` (`75 passed, 7 warnings in 100.05s`). The bundle
writer also emits byte-identical archives for identical source files.

An additional real-data smoke check used Q1 cutout
`102018665_NEG570040238507752998_VIS.fits`, with its already-recorded B5
baseline and fixed 12-point grid. The bundle contained exactly four members
(`manifest.json`, `baseline.json`, `cutout.npz`, `grid.yaml`); the bundled NPZ
contained only `image`, `mask`, `noise_map`, `observation_json`, and `psf_fine`.
Two fresh `skia-app` processes completed all 12 points and emitted byte-identical
result files (SHA-256
`2b24cc51bf3f8c4362ac3b6a601f9affcee44f21b916b5ad5f2b4dc7c2a5419d`). The
bundle SHA-256 was
`d344dc1643423d248be85c22dc72095c63d487a1b5852697872f96f3bd2f25dc`. Its best
raw delta-chi-square was 304.568559 at alpha_Rs=0.02 arcsec; this is one selected
candidate smoke run, not a calibrated detection or mass result. Temporary input,
checkpoint, and output files are under ignored `work/p3-q1-smoke/`.

Checkpoint recovery was also exercised on this same Q1 task. The first process
was terminated after progress 4/12; its checkpoint contained four completed
rows. A second process resumed at 5/12, completed the remaining eight points,
and produced the same result SHA-256 as the uninterrupted runs above. This
demonstrates local same-build recovery for this task only; it does not test
corrupt storage, host migration, or cross-platform checkpoint compatibility.

This is a same-machine regression check only. No second CPU architecture, OS,
Python/JAX build or glibc environment was available, so cross-device numeric
differences and validator tolerance remain **unmeasured**. Stable output within
one environment does not establish cross-platform determinism. The result
format uses fixed ten-decimal values and excludes wall-clock timings, but an
inter-platform tolerance must be measured before validator policy can be set.
