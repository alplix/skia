# AGENTS.md: skia

Orientation for coding agents and new contributors. The science is described in [README.md](README.md) and the reports
in `docs/`. Line numbers below refer to commit a88de16 unless a line says otherwise.

This file is tracked since 2026-11-05; until then `.gitignore` listed `AGENTS.md` with the maintainer's local working
notes. Private notes belong in `PLAN.md`, `STATUS.md`, `TASKS.md` and `REQUESTS.md`, which stay ignored.

## What one task is

A task gets one input file, `task.zip` (`skia-task-v1`, [docs/FORMATS.md](docs/FORMATS.md)), and writes one,
`result.txt` (`skia-scan-v1`). The ZIP holds `manifest.json` (SHA-256 of the other three), `baseline.json` (a
`skia-fit-v1` smooth fit of the cutout: its parameters and optimizer settings), `cutout.npz` (`skia-cutout-v1`: image,
noise, mask, oversampled PSF, pixel scale, exposure) and `grid.yaml` (a TNFW grid: `Rs_arcsec`, `r_trunc_arcsec` and the
axes `x_arcsec`, `y_arcsec`, `alpha_Rs_arcsec`). `boinc/make_tasks.py` makes them: one baseline fit per Q1 cutout
(`FitConfig(gamma_prior_sigma=0.1)`), then six bundles per lens of 66 points each (two x columns, eleven y values from
-1.0 to 1.0 arcsec, `alpha_Rs` 0, 0.02 and 0.05 arcsec; `Rs` 0.2, `r_trunc` 0.8).

`skia-app` (`src/skia/app_cli.py`) runs `scan()` (`src/skia/scanner.py:220`): it refits the smooth model from the
bundle's parameters with no subhalo (the baseline, 254), then, for every grid point in the order x, y, `alpha_Rs` (234),
computes the TNFW deflection field on the PSF-padded, oversampled pixel grid (100-117) and refits all 15 smooth
parameters (18 with a second Sersic component) from the bundle's parameters (263). It records chi2 and
`delta_chi2` = baseline chi2 - chi2. A fit (`fit_cutout`, `src/skia/fitter.py:200`) is SciPy's bounded `least_squares`
(TRF) from two starts, the given vector and one seeded perturbation of it, with the residual and the forward-mode
Jacobian of a JAX renderer (EPL + shear lens, Sersic source, FFT convolution with the PSF; `fitter.py:101-129`). The
Jacobian is about 85 % of a task's time (docs/BENCHMARK.md).

For volunteers it is a beta for Linux x86_64 (glibc 2.31 or newer), run by the stock BOINC wrapper
([boinc/README.md](boinc/README.md)): `boinc/job.xml` task 1 unpacks `skia_app.tar.gz` (the app frozen with its
Python environment, built outside this repository), task 2 runs `skia_app/skia_app task.zip -o result.txt` with the
thread settings of job.xml. Two matching results are needed; README gives the cross-platform difference measured so
far (1.5e-12 relative) and the tolerance (1e-8).

## Layout

| Path | What it holds |
|---|---|
| `src/skia/fitter.py` | Smooth model: parameter vector (log amplitude and radius), fixed bounds `LOWER`/`UPPER` (31-36), the JAX renderer and its Jacobian (`_compiled_renderer`, cached per geometry), `fit_cutout`, `save_fit` (fixed 10-decimal JSON) |
| `src/skia/scanner.py` | `scan()`: baseline refit, TNFW fields (`halo_deflection_field`, vectorized `halo_deflection_fields`), per-point refits, checkpoint (write, validate, resume), `save_scan` |
| `src/skia/task_bundle.py` | `skia-task-v1` writer (canonical members, stored ZIP, fixed timestamps) and reader (digests, schema) |
| `src/skia/cutout.py` | `Cutout` (validated, read-only arrays), `skia-cutout-v1` NPZ reader and writer |
| `src/skia/realdata.py` | Euclid Q1 VIS FITS to `Cutout`: sky and MAD noise from the outer 15 % border, Gaussian PSF of 0.16 arcsec (assumed) |
| `src/skia/simulator.py` | Seeded synthetic lenses (EPL + shear, Sersic, TNFW or pseudo-Jaffe subhalo), Poisson and read noise |
| `src/skia/cli.py`, `scan_cli.py`, `bundle_cli.py`, `app_cli.py` | `skia-fit`, `skia-scan`, `skia-bundle`, `skia-app` |
| `boinc/` | Wrapper notes, `job.xml`, draft templates `skia_in.xml`/`skia_out.xml`, `zipstub.cpp` (wrapper build), `make_tasks.py` (task queue) |
| `tests/` | `make test` (pytest); `tests/bench/` the study scripts behind the reports in `docs/` (P1-P3, B1-B7) |
| `docs/` | Method notes (FITTING, SCANNING, FORMATS, SIMULATOR, USAGE), data and licence notes, study reports with their raw JSON |
| `scripts/audit_environment.py` | Writes `docs/LICENSES.md` and `docs/licenses/` from the installed environment |
| `scripts/bench/` | Speed A/B of the app on task bundles, as the wrapper runs it: `mac_ab.sh`, `linux_ab.sh` (containers, also Linux x86_64 under emulation), `patches/` result-changing experiments. Method, profile and results: [docs/BENCHMARK.md](docs/BENCHMARK.md) |
| `docs/kb/` | Knowledge base (below) |

`data/`, `work/`, `.venv/` and `build/` are in `.gitignore`. There is no vendored third-party code; the dependencies
are pinned in `requirements.lock`.

## Invariants

Break one of these and results become silently wrong or stop matching between volunteers.

* **Output rows.** `save_scan` (`scanner.py:289-294`): two header lines, then one row per grid point,
  `x_arcsec y_arcsec alpha_Rs_arcsec delta_chi2 chi2 converged`, ten decimals, no timings, in the order x, y,
  `alpha_Rs` of `grid.yaml`. The validator and assimilator are not in this repository; what they parse is this.
* **Determinism.** One machine and one environment give byte-identical results across processes, resumes and repeats
  (docs/P3_DETERMINISM.md): fixed seed 731 per fit (`fitter.py:236`), fixed start order, ties broken by start index
  (262), and the scan's `best` ties broken by coordinates (`scanner.py:283`). Changing the seed, the bounds, the
  starts, the tolerances or the output precision changes results: that is a science change for the maintainers.
* **Thread settings decide the last digits.** With XLA's multi-threaded Eigen on (the default), the results change
  with the number of XLA CPU threads, which is the number of cores unless `PJRT_NPROC` sets it; with
  `--xla_cpu_multi_thread_eigen=false`, which `boinc/job.xml` sets, they were the same for 1 to 64 threads (SKI-001).
  Keep that flag in job.xml. The CLIs and `make test` do not set it, so their last digits can differ from the
  volunteer app's on the same machine.
* **float64 and CPU.** `fitter.py:13` enables JAX x64 at import; the CLIs default to `JAX_PLATFORMS=cpu`.
* **The baseline in the bundle is a start vector.** The app refits the baseline itself (`scanner.py:254`) and
  `delta_chi2` is relative to that refit; `baseline.json`'s `optimizer` block is the fit configuration of the scan.
* **A zero field is the baseline problem.** A grid point with `alpha_Rs_arcsec` 0 gets an all-zero deflection field,
  the same data, start and configuration as the baseline refit, so its fit is the baseline's bit for bit; `scan()`
  reuses the baseline fit for it on this branch (SKI-003). Anything that makes the baseline fit differ from a
  zero-field fit (another start vector, a prior only on the baseline) must drop that reuse.
* **Bundles are canonical.** `task_bundle._read_sources` strips simulator truth from the cutout and rewrites
  baseline and grid canonically; identical inputs give byte-identical ZIPs.
* **Units.** `alpha_Rs_arcsec` is an angular lensing strength, not a mass; `delta_chi2` is a fit improvement, not a
  calibrated significance (README, docs/P2_REPORT.md).

## Checkpoint/resume contract (what the code does at a88de16)

* File: `<output>.checkpoint.json` (`app_cli.py:19`), so `result.txt.checkpoint.json` in the slot directory;
  `skia-checkpoint-v1` JSON with a fingerprint of the cutout arrays, metadata, baseline, grid and fit configuration
  (`scanner.py:156-164`), the baseline refit, and every finished row with its fitted parameters.
* Writing: after the baseline refit and after every grid point (`scanner.py:259`, `271`), atomically (temporary file,
  fsync, `os.replace`, 167-173), not gated by anything the client controls; job.xml names no checkpoint file, so the
  client never learns of these checkpoints (SKI-004).
* Resume: the wrapper's own checkpoint records that task 1 (tar) is done and restarts task 2; `scan()` validates the
  checkpoint (fingerprint, row order and values, 186-217) and continues after the last row. A checkpoint that does not
  match or does not parse ends the task with exit code 2; it is never ignored or truncated.
* `result.txt` is written once, at the end, from the rows; a resumed task gives the same bytes as an uninterrupted
  one (docs/P3_DETERMINISM.md, same machine).
* Progress: `PROGRESS done total` lines on stderr (`app_cli.py:22`) are not read by the wrapper, which reports
  progress by task weight only (SKI-005).

## Building

* Development: README's setup (python3.14 venv, `pip==25.3`, `requirements.lock`, then
  `pip install --no-build-isolation --no-deps -e .`). `scripts/bench/setup_env.sh` makes the same environment in
  `~/skia-work/venv`, outside the repository.
* Volunteer app: built outside this repository (docs/P3_PACKAGING.md: no packager here); `boinc/README.md` says how
  the wrapper was built (BOINC 8.2.15 `samples/wrapper` in a Debian 11 chroot, `zipstub.cpp`, static libstdc++) and how
  the app version is laid out (`<copy_file/>` for the archive and job.xml).
* `boinc/job.xml` task 2 sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1,
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` and, on this branch, `PJRT_NPROC=1` (one XLA CPU thread, SKI-002).
  Until this branch the XLA_FLAGS value also held `intra_op_parallelism_threads=1`, which is not an XLA flag and was
  ignored. On Linux `app_cli.py` also caps XLA at one thread itself (`upstream/public`): it narrows the CPU mask to
  one core while XLA starts, then gives every thread the full mask back; `SKIA_ALL_CORES=1` skips that.
* Dependency licences: `scripts/audit_environment.py` (docs/LICENSES.md); clearance is still open (README).

## Tests

* `make test PYTHON=<venv python>`: 116 tests in 30 s on an Apple M4 Pro, one of them Linux-only (2026-11-06).
  Five start the installed console scripts (`skia-fit`) or scripts in subprocesses, so the package must be installed
  in the environment (the README's editable install; with `PYTHONPATH=src` the subprocesses use this checkout's code).
* Any change: `make test`, then a task bundle run before and after and `cmp` of `result.txt`, with job.xml's
  environment (`scripts/bench/build_variant.sh` writes a `run` script that sets it).
* Checkpoint change: stop a run after a few `PROGRESS` lines, start it again, `cmp` with an uninterrupted run
  (docs/P3_DETERMINISM.md did this by hand).
* Speed change: `scripts/bench/mac_ab.sh` (and `linux_ab.sh`); claim a gain only by the rule in docs/BENCHMARK.md,
  and show `same output`, or keep the change as a patch in `scripts/bench/patches/` with its measured effect.
* Statistics changes: the study scripts in `tests/bench/` and their reports; `make recovery` repeats P1.

## Environment notes

* XLA's CPU client starts one worker thread per core; FFTs (ducc0) and fused kernels run on them, so one `skia-app`
  used about 1.6 cores on an M4 Pro. `PJRT_NPROC` (or `NPROC`) sets that number; no XLA flag does, and
  `OMP_NUM_THREADS` does not reach it (SKI-002).
* XLA also sizes its pools from the CPU mask (that is how `app_cli.py` gets one thread on Linux), but pool threads
  keep a mask narrowed while they start. 020bf4f did not give it back, and Python has no `os.sched_getcpu`, so every
  task ran its XLA work on CPU 0: twelve tasks at once took 5.94x as long (2026-11-06, Linux arm64 container; SKI-002).
  98823a5 gives every thread the full mask back; `tests/test_app_cli.py` checks the pool sizes and the masks.
* `XLA_FLAGS`: an unknown `--flag` ends the process with exit code 1 and an `F`-level "Unknown flag in XLA_FLAGS" line
  (jaxlib 0.12.0.1, macOS, 2026-11-05; also for names that appear in jaxlib's strings, such as
  `--xla_cpu_disable_loop_unrolling`), so a typo in job.xml fails every task; a word without `--` is ignored without a
  message. Test any change of job.xml's environment with a run (`build_variant.sh` labels use it).
* macOS gives no CPU affinity, and the maintainer's timings (docs/B4_PERFORMANCE.md, docs/P2_COST.md) saw more than
  one core in use: those are local CPU timings, not one-core rates.
* The Q1 cutouts (`data/q1_cutouts/`, 185 files) and the Euclid reference tables (`data/ref/`) are not in the
  repository: docs/DATA.md, `docs/fetch_q1_cutouts.py` (needs astroquery, not in `requirements.lock`, and the ESA
  archive, which answered "Archive down for maintenance" on the afternoon of 2026-11-05).

## Conventions for changes

* Measured claims: say what was compared, on which platforms and inputs ("byte-identical on ..."). Date what records
  a finding.
* Never change science results silently: seeds, starts, bounds, tolerances, priors, the PSF or noise model, the grid
  or the output precision are changed only with their measured effect and the maintainers' agreement. A speed-up that
  moves the last digits is a result change too: keep it behind a switch or as a patch.
* Keep job.xml's thread settings: a task must use one CPU, as BOINC counts it, and its result must not depend on the
  volunteer's number of cores.

## Not in this repository

* The server side: the work generator's deployment, validator (and its tolerance), assimilator, the app version and
  its `skia_app.tar.gz`, `skia_wrapper` (a binary, ignored).
* The packaging of the frozen app (docs/P3_PACKAGING.md), the BOINC source.
* The input data (docs/DATA.md) and the maintainer's working notes (`PLAN.md`, `STATUS.md`, `TASKS.md`, `REQUESTS.md`,
  `handoff/`, `reviews/`, ignored).

## Knowledge base

`docs/kb/` holds this repository's known bugs and platform traps, and the bug patterns shared with the other bitboinc
applications. Workflow: [docs/kb/shared/PROCESS.md](docs/kb/shared/PROCESS.md). It has two parts:

* **This repository's records**, edited here only: `issues/` (one file per issue, `SKI-NNN`), `ledger.md` (this
  repository's result for every shared pattern), `kb.conf` (the issue prefix), `README.md`, and `INDEX.md`, which
  `kb.py index` writes.
* **The shared part, `docs/kb/shared/`**, the same in every bitboinc repository: the procedure (`PROCESS.md`,
  `REFERENCE.md`), `vocabulary.md`, `boinc-api.md`, the bug patterns (`patterns/PAT-NNN-*.md`), `templates/`, the tool
  `kb.py` and `MANIFEST`. Never change it in one repository only. After editing any file there, run
  `python3 docs/kb/shared/kb.py sync ../kampylos ../nostos ../phygas ../planodia ../yarrow` from this repository,
  naming every other bitboinc repository checked out next to this one, including any that are not public. It copies
  the newest version of each shared file to all of them and gives each ledger an `unchecked` row for a new pattern.
  Then commit `docs/kb` in every repository. `MANIFEST` holds the hashes of the last sync, and `kb.py lint` warns when
  a shared file here differs from it.

Most shared patterns are about the BOINC C API. This app does not call it: the stock wrapper does, so those patterns
are `n/a` here and the wrapper's job.xml is where checkpoint and progress reporting are wired (SKI-004, SKI-005).

- Ready to fix, with next steps: `python3 docs/kb/shared/kb.py list --open --action fix` (also `--area`,
  `--platform`). Then `kb.py show SKI-NNN` for the issue and `kb.py show PAT-NNN` for the pattern it links, with the
  BOINC API facts the pattern cites. In the code, read the functions that `where:` and the Cause name.
- All repositories checked out next to this one: `kb.py todo`, then `kb.py todo --pattern PAT-NNN` for the next step
  and place of each issue; `kb.py check PAT-NNN` checks a pattern everywhere.
- Before debugging a symptom: `grep -ril "<words>" ../*/docs/kb/issues docs/kb/shared/patterns`, then `kb.py show`
  the ids it finds. Read other repositories through the shell (`kb.py show`, `git -C ../<repo> grep -n`, `sed -n`):
  opening one of their files in an editor tool can load their whole AGENTS.md as well.
- New report: `kb.py new issue "symptom in a few words" area=... next=...`.
- After a fix: `kb.py set SKI-NNN status=fix-proposed --log "..."` (`status=fixed fixed_in=<commit>` once merged),
  and update the passages of this file it prints. The commit names the issue (`KB: SKI-NNN`) and the pattern, if any
  (`Pattern: PAT-NNN`).
- Before a pull request: `kb.py index && kb.py lint --strict`.
