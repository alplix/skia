---
id: SKI-006
title: Every task unpacks the 176 MB app archive into its slot directory
type: bug
status: new
severity: low
platforms: [linux]
area: performance
action: verify
next: Measure the unpack time and the slot size of skia_app.tar.gz on a Linux host
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: code reading 2026-11-05 (boinc/job.xml, boinc/README.md)
where: boinc/job.xml:2-6 @ a88de16
fixed_in:
---

## Symptom

Not measured. boinc/job.xml task 1 runs `/bin/tar -xzf skia_app.tar.gz` at the start of every task, so each task
decompresses the 175,798,226-byte archive (boinc/README.md) into its own slot directory, and a machine running one task
per core holds one unpacked copy per core.

## Cause

The wrapper runs tasks in the slot directory, and the app version gives the archive with `<copy_file/>`
(boinc/README.md), so nothing is shared between tasks. The size of the unpacked app and the time of the unpack are not
recorded anywhere in this repository.

## Fix

First measure (unpack time and slot size on a Linux host, against the scan's 22 minutes). If it matters: unpack once
per app version (for example into the project directory, guarded by a version file) and run from there, or ship the
app unpacked as several app-version files. The archive is built outside this repository.

## Test

`time tar -xzf skia_app.tar.gz` and `du -sh skia_app` on a Linux host.

## Log

- 2026-11-05: created.
