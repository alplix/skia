---
id: SKI-003
title: A third of every task repeats the baseline fit: grid points with alpha_Rs 0
type: bug
status: fix-proposed
severity: medium
platforms: [all]
area: performance
action: review
next: Review the reuse of the baseline fit for zero-field points in scan()
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: src/skia/scanner.py:252-272 @ a88de16
fixed_in:
---

## Symptom

Wasted time, no wrong result: every `alpha_Rs` 0 row of a task has `delta_chi2` 0.0000000000 and the baseline's chi2,
because its fit is the baseline refit again. In the production grid (`boinc/make_tasks.py`: `alpha_Rs` 0, 0.02, 0.05)
that is 22 of a task's 66 points.

## Cause

`halo_deflection_field` returns `np.zeros` for strength 0 (`src/skia/scanner.py:112` at a88de16), and `scan()` then
calls `fit_cutout` with the same cutout, start vector and configuration as the baseline refit (254, 263), whose field is
`jnp.zeros` of the same shape (`fitter.py:144`). The fit is deterministic (seed 731, fixed starts), so the result is
the baseline's bit for bit.

## Fix

This branch: `scan()` keeps the baseline `FitResult` and uses it for every point whose field is all +0.0
(`_is_zero_field`, a check of the bit patterns, so a field with a -0.0 is still fitted). After a resume the baseline
fit is not in memory: the first zero-field point is fitted and reused for the rest. The rows, the checkpoint and
`result.txt` are unchanged; the vectorized-field path (`--vectorized-fields`) gets the reuse only where its field is
exactly +0.0. Rejected: skipping those rows (a format change) and computing `alpha_Rs` 0 on the server (the scan must
refit its own baseline).

## Test

`tests/test_scanner.py::test_zero_field_points_reuse_the_baseline_fit_bit_for_bit` (rows equal to fitting every point,
also after a resume that lost the baseline fit; one baseline fit and one fit per non-zero point).
`scripts/bench/mac_ab.sh 3 "sim task-sim"` on an Apple M4 Pro, 2026-11-05 (docs/BENCHMARK.md, Results): `sim` 7.11 s ->
5.89 s (1.21x), three real Q1 cutouts 1.20-1.21x, `task-sim` 62.7 s -> 49.7 s (1.26x), the real 66-point `task`
303.9 s -> 253.6 s (1.20x), CPU time likewise; twelve tasks on twelve cores 1.29x; the same `result.txt` in every run,
also on Linux arm64 and x86_64 (containers).

## Log

- 2026-11-05: created.
