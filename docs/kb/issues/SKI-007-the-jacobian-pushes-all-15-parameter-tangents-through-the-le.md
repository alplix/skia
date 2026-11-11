---
id: SKI-007
title: The Jacobian pushes all 15 parameter tangents through the lens equation, 9 of them zero
type: question
status: confirmed
severity: medium
platforms: [all]
area: performance
action: decide
next: Decide whether a 1.2x faster Jacobian that moves results in the last digits (scripts/bench/patches/split-jacobian.patch) is acceptable
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: src/skia/fitter.py:101-129 @ a88de16
fixed_in:
---

## Symptom

Speed, no wrong result: the Jacobian of the renderer is about 85 % of a task's time (docs/BENCHMARK.md, "Where the time
goes"), and most of it is spent pushing zeros through the EPL deflection series.

## Cause

`_compiled_renderer` returns `jax.jit(jax.jacfwd(render))` (`src/skia/fitter.py:129` at a88de16), which sends all 15
(18) one-hot tangents through the whole renderer. Only the 8 lens parameters (6 EPL, 2 shear) move the source plane;
the 7 (10) source-light tangents are zeros through the EPL series (a 19-term loop in complex numbers, `herculens`
`R_omega`). Timed alone with one XLA thread on the 258x258 oversampled grid of a Q1 cutout (Apple M4 Pro,
2026-11-05): the EPL deflection 1.6 ms, its Jacobian over the 6 EPL parameters 35.6 ms, the whole Jacobian 61 ms.

## Fix

Measured, not applied: `scripts/bench/patches/split-jacobian.patch` computes the source plane's Jacobian over the 8
lens parameters and starts the source parameters' tangents at the source plane, then sends all columns through the
Sersic light, PSF convolution and pixel binning as before. The derivatives are the same, but XLA folds the smaller
graph in another order, so the Jacobian differed from `jacfwd`'s in the last bits (about 1e-13 absolute, up to 7e-12, in 16,000 of
219,615 entries for most parameter vectors, every entry for some), and the fits carry that into the printed digits.
Measured against this branch (both same-results changes in), one XLA thread, 3 rounds: `sim` 1.17x, `q1-mid` 1.18x,
`task-sim` 1.22x faster; rows moved: `task-sim` 4 of 66 rows by 1e-10 (the last printed digit), `q1-mid` 3 of 6 rows
by up to 2.7e-8 in chi2 (1.2e-12 relative, 3.2e-11 relative in `delta_chi2`); no convergence flag changed. That is
inside the validator's 1e-8 tolerance (README) and smaller than the difference between macOS and Linux on the same
Apple M4 with the same jaxlib (`q1-mid`: 5 of 6 rows, up to 3.6e-12 relative in chi2), but it changes results, so it
is the maintainers' decision. Not tried: 6 tangents through the series and
the shear by hand, and herculens' real-arithmetic EPL (`no_complex_numbers=True`, another formula).

## Test

`NEW_PATCHES=scripts/bench/patches/split-jacobian.patch scripts/bench/mac_ab.sh 3 "sim q1-mid task-sim"` and
`docs/BENCHMARK.md` (Results). `make test` on a copy of this branch with the patch applied, 2026-11-05: 114 passed;
the one failure, `test_recovery_script_smoke_and_refuse_overwrite`, is the copy (its script runs `git rev-parse HEAD`
and the copy is not a git checkout).

## Log

- 2026-11-05: created.
