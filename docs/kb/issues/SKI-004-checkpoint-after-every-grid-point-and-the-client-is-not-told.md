---
id: SKI-004
title: Checkpoint after every grid point, and the client is not told of it
type: bug
status: confirmed
severity: low
platforms: [boinc-client]
area: checkpoint
action:
next: Name result.txt.checkpoint.json as checkpoint_filename of task 2 in boinc/job.xml and test a suspend and resume with a real client
blocked_by: []
pattern: PAT-004
aliases: []
reported: 2026-11-05
source: code reading 2026-11-05
where: boinc/job.xml:7-16 @ a88de16
fixed_in:
---

## Symptom

Nothing a volunteer sees directly. The client is never told that the scan checkpointed: it sees a checkpoint only when
a wrapper task ends (after the tar step and at the end), so for a whole task (about 22 minutes on one core, README) it
counts the work as unsaved, and the app rewrites and fsyncs its checkpoint after every grid point regardless of the
client's checkpoint interval (default 60 s).

## Cause

`scan()` saves the whole checkpoint, every finished row with its fitted parameters, after the baseline refit and after
every grid point (`src/skia/scanner.py:259`, `271` at a88de16), atomically with fsync (167-173). The app has no BOINC
API, and task 2 of `boinc/job.xml` names no `<checkpoint_filename>`, the wrapper's way to learn of an application's
checkpoints and report them (`boinc_checkpoint_completed`).

## Fix

Proposed: `<checkpoint_filename>result.txt.checkpoint.json</checkpoint_filename>` in task 2 of job.xml (the wrapper
reports a checkpoint when the file changes). The write itself is atomic and small (about 1 kB per row, 66 rows at most),
so its cadence matters less; a time gate of a minute would lose more work on a restart than it saves. Not done on this
branch because it needs a test with a real client.

## Test

Run a task under a BOINC client, suspend it with "leave non-GPU tasks in memory" off, and check that the client's
"CPU time since checkpoint" falls after each grid point and that the task resumes after the last finished point.

## Log

- 2026-11-05: created.
