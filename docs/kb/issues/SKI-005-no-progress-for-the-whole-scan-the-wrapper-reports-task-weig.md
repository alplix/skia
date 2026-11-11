---
id: SKI-005
title: No progress for the whole scan: the wrapper reports task weights only
type: bug
status: confirmed
severity: low
platforms: [boinc-client]
area: boinc-lifecycle
action:
next: Write the fraction done to a file after each grid point and name it as fraction_done_filename in boinc/job.xml
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: code reading 2026-11-05; boinc/README.md
where: src/skia/app_cli.py:22 @ a88de16
fixed_in:
---

## Symptom

During a task the client shows the progress the wrapper computes from task weights: 0 while `skia_app.tar.gz` is
unpacked, then 1/21 (about 5 %) for the whole scan, then 100 %. The client's estimate of the time left is therefore
based on the work-unit's `rsc_fpops_est` only.

## Cause

The app prints `PROGRESS done total` to stderr (`src/skia/app_cli.py:22` at a88de16), which the wrapper copies to
`skia_stderr.txt` and does not read. job.xml gives task 1 weight 1 and task 2 weight 20 and no
`<fraction_done_filename>` (boinc/job.xml:5, 15). boinc/README.md lists "progress reporting (weights only)" as not
done.

## Fix

Proposed: after each grid point, also write `done/total` to a small file (atomically) and name it as
`<fraction_done_filename>` of task 2; the wrapper then reports the weighted fraction of that task. A resumed task
reports its position at the first point it finishes.

## Test

Run a task under a client and watch the progress column move after every grid point.

## Log

- 2026-11-05: created.
