# Pattern ledger

One row per cross-repository pattern ([shared/patterns](shared/patterns)): has this repository been checked for it,
at which commit, and what was found. `kb.py sync` adds an `unchecked` row whenever a new pattern arrives; checking it
means following the pattern's "How to check a repository" section and filling in the row. Status values:
[shared/vocabulary.md](shared/vocabulary.md#ledger). A row that is `affected` or `fixed` names the local issue, and that
issue names the pattern in its `pattern:` field; its Evidence says where, in one clause, and the issue has the rest.
Write rows with `kb.py ledger` (date and commit filled in) rather than by hand.

When code that a row describes changes a lot, check the pattern again and update Checked, Commit and Evidence.

| Pattern | Status | Checked | Commit | Issue | Evidence |
|---|---|---|---|---|---|
| PAT-001 | n/a | 2026-11-05 | a88de16 |  | no BOINC API call in this repository (git grep boinc_ finds none in src/): the Python app is run by the stock BOINC wrapper 8.2.15 (boinc/README.md, boinc/job.xml), which owns the critical sections |
| PAT-002 | not-affected | 2026-11-05 | a88de16 |  | result.txt is written once, after the scan, from the checkpointed rows (scanner.py:289-294, app_cli.py:24); nothing is appended before a checkpoint |
| PAT-003 | not-affected | 2026-11-05 | a88de16 |  | scanner.py:167-173 writes the checkpoint to a temporary file, fsyncs it and os.replace()s it; a write error raises and the task exits 2 |
| PAT-004 | affected | 2026-11-05 | a88de16 | SKI-004 | scanner.py:259, 271: the whole checkpoint (all rows so far) is rewritten with fsync after every grid point, seconds apart, and job.xml names no checkpoint file for the wrapper |
| PAT-005 | not-affected | 2026-11-05 | a88de16 |  | scanner.py:238-251: an existing checkpoint that cannot be read, parsed or matched raises ValueError, so the task ends with exit 2; it is never treated as absent or truncated |
| PAT-006 | not-affected | 2026-11-05 | a88de16 |  | save_scan uses Path.write_text (scanner.py:294), which raises on a write error; app_cli.py:25-26 turns OSError into exit code 2, which the wrapper reports |
| PAT-007 | n/a | 2026-11-05 | a88de16 |  | the app reports no progress to the client at all: PROGRESS lines go to stderr (app_cli.py:22) and the wrapper reports task weights; recorded as SKI-005 |
| PAT-008 | n/a | 2026-11-05 | a88de16 |  | no BOINC API call in this repository (git grep boinc_ finds none in src/): the Python app is run by the stock BOINC wrapper 8.2.15 (boinc/README.md, boinc/job.xml); the wrapper handles quit and abort |
| PAT-009 | n/a | 2026-11-05 | a88de16 |  | no trickle messages (git grep trickle finds none) |
| PAT-010 | n/a | 2026-11-05 | a88de16 |  | no BOINC API call in this repository (git grep boinc_ finds none in src/): the Python app is run by the stock BOINC wrapper 8.2.15 (boinc/README.md, boinc/job.xml); the app is a child process of the wrapper, so the library's timer does not run in it |
| PAT-011 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-012 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-013 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-014 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-015 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-016 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-017 | n/a | 2026-11-05 | a88de16 |  | no Android build and no OpenCL |
| PAT-018 | affected | 2026-11-05 | a88de16 | SKI-001 | results change with the XLA CPU thread count unless --xla_cpu_multi_thread_eigen=false (job.xml:13 sets it, the CLIs do not) |
| PAT-019 | n/a | 2026-11-05 | a88de16 |  | no Windows build (README); save_scan writes text mode (Path.write_text, scanner.py:294), which a future Windows build would write with CRLF |
| PAT-020 | n/a | 2026-11-05 | a88de16 |  | no MinGW build; the app is Python |
| PAT-021 | not-affected | 2026-11-05 | a88de16 |  | git grep for /Users/, /home/, /opt/, /tmp/ and drive letters finds none; scripts take paths as arguments with repository-relative defaults (boinc/make_tasks.py:56) |
| PAT-022 | n/a | 2026-11-05 | a88de16 |  | no binary is built in this repository: boinc/README.md says the wrapper was built in a Debian 11 chroot (newest symbol GLIBC_2.29); the frozen app is built elsewhere and its glibc need is unverified (docs/P3_PACKAGING.md) |
| PAT-023 | not-affected | 2026-11-05 | a88de16 |  | docs/fetch_q1_cutouts.py appends a manifest row only after a cutout was written and refetches a file without a row; boinc/make_tasks.py writes <lens>.done after all bundles of a lens |
| PAT-024 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-025 | not-affected | 2026-11-05 | a88de16 |  | the wrapper suspends the whole app process, XLA's worker threads included; the app starts no threads of its own |
| PAT-026 | not-affected | 2026-11-05 | a88de16 |  | Python sorting is the same everywhere; the only tie-break that reaches the output, the scan's best row, uses the coordinates (scanner.py:283) and fit_cutout breaks ties by start index (fitter.py:262) |
| PAT-027 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-028 | n/a | 2026-11-05 | a88de16 |  | no build scripts in this repository; the thread settings that results depend on are in boinc/job.xml (SKI-001) |
| PAT-029 | not-affected | 2026-11-05 | a88de16 |  | requirements.lock pins every package with == (no hashes), pyproject.toml pins setuptools; the wrapper build names BOINC 8.2.15 and Debian 11 (boinc/README.md) |
| PAT-030 | n/a | 2026-11-05 | a88de16 |  | no BOINC API call in this repository (git grep boinc_ finds none in src/): the Python app is run by the stock BOINC wrapper 8.2.15 (boinc/README.md, boinc/job.xml); no temporary exits |
| PAT-031 | n/a | 2026-11-05 | a88de16 |  | Python; no thread_local or __thread (git grep) |
| PAT-032 | n/a | 2026-11-05 | a88de16 |  | no C or C++ hot loop; the app is Python, NumPy and XLA, and no macOS build is shipped |
| PAT-033 | n/a | 2026-11-05 | a88de16 |  | CPU only: no GPU, CUDA or OpenCL code (JAX_PLATFORMS=cpu, requirements.lock has CPU jaxlib) |
| PAT-034 | not-affected | 2026-11-10 | 99c579e |  | boinc/skia_in.xml:4-5: rsc_fpops_est 2e12, rsc_fpops_bound 2e14 (100x); measured task cost spread 4.9x (task-sim 62.7 s, a real 66-point task 304 s, docs/BENCHMARK.md 2026-11-05) |
