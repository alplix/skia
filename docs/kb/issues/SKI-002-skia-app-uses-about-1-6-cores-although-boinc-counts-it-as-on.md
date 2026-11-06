---
id: SKI-002
title: skia-app uses about 1.6 cores although BOINC counts it as one CPU
type: bug
status: fix-proposed
severity: medium
platforms: [all]
area: performance
action: review
next: Review PJRT_NPROC=1 in boinc/job.xml; check the frozen app reads it (jaxlib 0.12.0.1 does) on a Linux host
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: boinc/job.xml:10-13 @ a88de16
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

## Fix

This branch: `<setenv>PJRT_NPROC=1</setenv>` in task 2 of job.xml, and `XLA_FLAGS` reduced to
`--xla_cpu_multi_thread_eigen=false`, which is the part that works (and keeps results independent of the thread count,
SKI-001). The results stay the same: with job.xml's flag, every thread count from 1 to 64 gave the same `result.txt`,
and every benchmark run of this branch matched `upstream/main`. A task alone then takes longer (`task-sim` 1.55x the
wall time) but 6-9 % less CPU time; with every core busy, which is how BOINC runs it, twelve tasks finished 4 % sooner
(220.3 s against 230.6 s, not a clear gain by the rule) and with SKI-003 25.9 % sooner. If the maintainers want
multi-core tasks, the honest way is an `mt` plan class whose thread count is passed to `PJRT_NPROC`; that is a
decision for them.

## Test

`scripts/bench/mac_ab.sh` (the `new` label runs with job.xml's environment) and `COPIES=12 scripts/bench/mac_ab.sh 3
task-sim`; `linux_ab.sh` for Linux. Not checked: that the frozen volunteer app (its own jaxlib build) reads
`PJRT_NPROC`; jaxlib 0.12.0.1 of `requirements.lock` does, on macOS and on Linux arm64 and x86_64 (containers).

## Log

- 2026-11-05: created.
