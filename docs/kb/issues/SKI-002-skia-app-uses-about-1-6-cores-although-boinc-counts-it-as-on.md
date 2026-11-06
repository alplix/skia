---
id: SKI-002
title: skia-app uses about 1.6 cores although BOINC counts it as one CPU
type: bug
status: fix-proposed
severity: medium
platforms: [all]
area: performance
action: review
next: Review PJRT_NPROC=1 in job.xml next to app_cli.py's CPU mask (98823a5); check the frozen app on a Linux host
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: boinc/job.xml:10-13, src/skia/app_cli.py:11-40 @ 98823a5
fixed_in:
---

## Symptom

BOINC runs `skia-app` as a one-CPU task, but one task alone used about 1.6 cores: on an Apple M4 Pro (12 cores) the
`task-sim` profile took 62.7 s of wall time and 103.9 s of CPU time (2026-11-05, docs/BENCHMARK.md). `ps -M` showed
twelve busy worker threads; `sample` showed ducc0 FFTs and fused XLA kernels on them. A volunteer who lets BOINC use
only some of the cores gets the others taken anyway, and on a machine where every core runs a task each task still
starts a thread per core (load average 195 with twelve tasks on twelve cores).

## Cause

XLA's CPU client sizes its thread pool from `PJRT_NPROC`, else `NPROC`, else the number of cores (CPU affinity on
Linux). No XLA flag sets it, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` (`boinc/job.xml:10-12`
at a88de16) do not reach it. job.xml's `XLA_FLAGS` held `intra_op_parallelism_threads=1` (13), a TensorFlow session
option, not an XLA flag; XLA ignores a word without `--` without a message (jaxlib 0.12.0.1).

`upstream/public` (2026-11-06, "v3") caps the pools another way, on Linux: `app_cli.py` narrows the process's CPU mask
to one core while the backend starts, and XLA then makes one thread per pool. In the first version, 020bf4f, only the
main thread got its mask back; the pool threads kept the one-core mask, and the core was `os.sched_getcpu()`, which
Python does not have (3.14.10 in the containers), so always `min(full)`, the lowest core. In the Linux arm64 container
the `tf_XLAEigen`, `tf_XLACompile` and `tf_XLAExecute` threads of every task had `Cpus_allowed_list` 0 (in a probe
that timed a jitted matrix product, the work ran on `tf_XLAExecute`), and under x86_64 emulation
`tests/test_app_cli.py` found the same. BOINC runs one task per core, so the tasks shared core 0 while the other cores
idled. 98823a5 (the same day) gives every thread the full mask back after the backend starts (`app_cli.py:29-35`). On
macOS there is no `sched_setaffinity` and the code does nothing.

## Fix

This branch: `<setenv>PJRT_NPROC=1</setenv>` in task 2 of job.xml, and `XLA_FLAGS` reduced to
`--xla_cpu_multi_thread_eigen=false`, which is the part that works (and keeps results independent of the thread count,
SKI-001). The results stay the same: with job.xml's flag, every thread count from 1 to 64 gave the same `result.txt`,
and every benchmark run of this branch matched `upstream/main`. A task alone then takes longer (`task-sim` 1.55x the
wall time) but 6-9 % less CPU time; with every core busy, which is how BOINC runs it, twelve tasks finished 4 % sooner
(220.3 s against 230.6 s, not a clear gain by the rule) and with SKI-003 25.9 % sooner. If the maintainers want
multi-core tasks, the honest way is an `mt` plan class whose thread count is passed to `PJRT_NPROC`; that is a
decision for them.

On Linux `app_cli.py` of `upstream/public` (98823a5) caps the pools as well, so there job.xml's setting is a second
guard; it is the only one where `sched_setaffinity` does not exist, and for app versions built before. Against 020bf4f
(Linux arm64 container, 2026-11-06, docs/BENCHMARK.md) twelve `sim` tasks at once took 131.44 s, against 22.12 s with
`PJRT_NPROC=1` and no narrowed mask (5.94x, 3 interleaved rounds, the same `result.txt`): the XLA work of all twelve
queued on CPU 0. An app version built from 020bf4f gets one free thread per XLA pool with `SKIA_ALL_CORES=1` and
`PJRT_NPROC=1` in job.xml (checked from source in the arm64 and x86_64 containers, not with the frozen app).

## Test

`tests/test_app_cli.py` (Linux): one thread per XLA pool and every thread's CPU mask that of the process, without
`PJRT_NPROC`; it fails on 020bf4f and passes on 98823a5, on arm64 and x86_64. `scripts/bench/mac_ab.sh` (the
`new` label runs with job.xml's environment) and `COPIES=12 scripts/bench/mac_ab.sh 3
task-sim`; `linux_ab.sh` for Linux. Not checked: that the frozen volunteer app (its own jaxlib build) reads
`PJRT_NPROC`; jaxlib 0.12.0.1 of `requirements.lock` does, on macOS and on Linux arm64 and x86_64 (containers).

## Log

- 2026-11-05: created.
- 2026-11-06: upstream `public` 020bf4f pinned XLA's threads to the lowest core; 98823a5 gives them the full mask
  back. The branch, rebased on 98823a5, keeps that `app_cli.py`, `PJRT_NPROC=1` in job.xml and a test of the masks.
