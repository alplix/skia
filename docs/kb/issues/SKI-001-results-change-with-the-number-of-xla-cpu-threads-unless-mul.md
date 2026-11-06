---
id: SKI-001
title: Results change with the number of XLA CPU threads unless multi-threaded Eigen is off
type: gotcha
status: confirmed
severity: medium
platforms: [all]
area: determinism
action: decide
next: Decide whether the CLIs and make test should set --xla_cpu_multi_thread_eigen=false like job.xml, so local runs give the volunteer app's digits
blocked_by: []
pattern: PAT-018
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: boinc/job.xml:13 @ a88de16
fixed_in:
---

## Symptom

The same task bundle gives result files that differ in the last printed digits depending on how many threads XLA's CPU
client has, which is the number of cores unless `PJRT_NPROC` says otherwise. Measured on an Apple M4 Pro (12 cores),
macOS 27.0, Python 3.14.5, jaxlib 0.12.0.1, on the benchmark profile `sim` (6 grid points), 2026-11-05: without
`XLA_FLAGS`, `PJRT_NPROC` 1, 2, 4, 6, 16 and 64 gave one `result.txt` (sha1 fc972885afc7...), 8, 12 (the default here)
and 32 another (e73186e47337...). With `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, every thread count from 1 to 64
gave fc972885afc7..., the single-thread result.

## Cause

With multi-threaded Eigen, XLA splits some operations (a contraction or reduction in the forward-mode Jacobian, not
identified further) into blocks by thread count, and the blocks are summed in another order. `boinc/job.xml` sets
`--xla_cpu_multi_thread_eigen=false` for the volunteer app, so its results do not depend on the volunteer's core
count; the CLIs (`skia-fit`, `skia-scan`, `skia-app` run by hand), `make test` and the study scripts in `tests/bench/`
do not set it, so the numbers in the reports were made with the thread count of the machine they ran on. Across
platforms the results differ anyway (README: 1.5e-12 relative between the two platforms measured, validator tolerance
1e-8).

## Fix

None needed for the volunteer app while job.xml keeps the flag (AGENTS.md, Invariants). For local runs the CLIs could
set the flag before importing JAX (`os.environ.setdefault`), as `cli.py` already does for `JAX_PLATFORMS`; that would
change the last digits of local results on machines where multi-threaded Eigen is used, so it is the maintainers'
decision.

## Test

`scripts/bench/run_bench.sh sim 1 a=<run script>` with labels made by `build_variant.sh` with `JOBXML=0` and
`ENV="PJRT_NPROC=8"` against `ENV="PJRT_NPROC=1"`: different hashes; with job.xml's environment: the same.

## Log

- 2026-11-05: created.
