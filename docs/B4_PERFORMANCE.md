# B4 scan profiling and field batching

## Profile

Measured on local macOS arm64 / CPython 3.14.1 / JAX CPU with a seeded 30x30
simulated cutout, one optimizer start, `max_nfev=120`, warm JAX geometry and a
3x3x3 grid (27 points). The scalar scan took 4.9523 s including the baseline
fit (0.1965 s); baseline-subtracted grid work was 0.1761 s/point. All 27 candidate
fits converged. Machine-specific raw profile is in
[`b4-profile.json`](b4-profile.json).

The cProfile run had 28 fits including the baseline. SciPy least-squares used
5.217 s cumulative of 5.445 s profiled time. Jacobian evaluation was the largest
inner cost (145 calls, 4.311 s cumulative). TNFW field preparation was small
relative to fitting.

## Vectorized field path

`halo_deflection_fields` batches TNFW field construction in chunks of 16 and
matches the scalar fields exactly in this experiment (max absolute difference
0 arcsec). Field generation took 45.3 ms batched vs 32.4 ms scalar. End-to-end
scan took 5.3393 s batched vs 4.9523 s scalar, so batching was about 7.8% slower
on this measured grid. `scan()` and `skia-scan` therefore keep scalar field
generation as the default; `--vectorized-fields` remains an experimental option
for larger geometry/grid testing. It is not currently a performance improvement.

The process was serial and common numerical thread environment variables were
set to one, but macOS did not expose CPU affinity and the process was not pinned
to a physical core. Treat seconds/point as this-host timing, not a verified
one-core rate. No GPU backend was available or measured. More realistic cutout
sizes, source models, starts and machine types require separate measurements.

## Expanded real-cutout timing

The B6 fixed baseline-plus-12-point-scan run recorded per-cutout wall time on
all 185 grade-A Q1 images. Summed elapsed time was 7,213.2 s (2.004 h), or
38.99 s per cutout on average; median was 35.22 s and the p16–p84 range was
26.03–51.59 s. The new 32 cutouts averaged 36.23 s and the original 153
averaged 39.57 s under the same run. This includes FITS loading, baseline
optimization, all 12 scan optimizations and per-row output; it is the full
current pipeline time, not a scan-only point cost.

These runs were serial and the benchmark scripts set common numerical-library
thread limits to one. macOS did not provide CPU affinity, and process-level
monitoring showed more than one CPU core in use during optimization; therefore
the reported values are local CPU timings, **not verified one-core rates**.
The installed JAX backend returned `[CpuDevice(id=0)]`; no GPU device was
available. The scalar scan remains the default because the measured field
batching path was slower, and the 185-cutout timing does not justify a new
performance claim for the optional vectorized path.

Raw per-cutout elapsed values are retained in
[`b6-185/real-robustness.jsonl`](b6-185/real-robustness.jsonl); aggregate
fit/scan outcomes are in the neighboring summary JSON.
