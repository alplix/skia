# P2.2 CPU cost measurement

Measured with `tests/bench/cost_model.py` on the local macOS arm64, Python 3.14.1
environment using a 30x30 simulated cutout, one fit start, a 2x2x2 angular TNFW
grid, and `max_nfev=120`. The scan is serial and JAX's compilation was warmed
before timing. Common numerical-library thread limits were set to one.

| Measurement | Observed |
|---|---:|
| Smooth baseline fit | 0.0693 s |
| Full scan including repeated baseline fit (8 grid points) | 0.3286 s |
| Estimated grid-point work (baseline-subtracted) | 0.0324 s / point |
| Estimated baseline + this 8-point grid | 0.3286 s / lens |
| Grid-point fits converged | 8 / 8 |

The raw machine-specific record is [p2-cost.json](p2-cost.json). This is a
small warm-cache simulated benchmark, not a production Q1 timing distribution.
The process was serial and thread limits were one, but macOS did not expose the
CPU-affinity API used by the benchmark, so execution was not pinned to one
physical core. Treat the timing as an indicative local CPU result, not a verified
one-core guarantee. Grid sizes and optimizer behavior can change the cost.

## Full Q1 pipeline timing

The fixed B6 baseline-plus-12-point-scan run on 185 Q1 cutouts took 7,213.2 s
(2.004 h) total, with 38.99 s mean, 35.22 s median and 26.03–51.59 s p16–p84
wall time per cutout. This includes FITS loading, the baseline fit, the full
12-point scan and output writing. It was serial with thread-limit environment
settings, but not pinned to one core; process monitoring saw multicore use.
JAX listed only a CPU device, and no GPU timing is available. Treat this as
full-pipeline local CPU timing, not a verified one-core performance number.
Per-cutout times are in `docs/b6-185/real-robustness.jsonl`.
