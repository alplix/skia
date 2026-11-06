# Benchmarking Skia (`scripts/bench/`)

How to measure whether a change makes `skia-app` faster and whether it keeps the results, the scripts that do it,
where the time goes, which changes to try, and the results. Written 2026-11-05. The method is the one used for the
other bitboinc applications, adapted to a Python and JAX app run by the BOINC wrapper.

**Status (2026-11-05).** First run on an Apple M4 Pro and in Linux containers (arm64 natively, x86_64 under
emulation). Two same-results changes are measured and on this branch (Results): the scan reuses the baseline fit for
the grid points with `alpha_Rs` 0 (SKI-003), and `boinc/job.xml` gives XLA one CPU thread (SKI-002). One
result-changing speed-up is measured and kept as a patch (SKI-007). The volunteer app itself (the frozen
`skia_app.tar.gz`) is built outside this repository and was not measured; the suite runs the same code from source
with the pinned environment.

## How to run

On a Mac, from the repository root, with the BOINC client paused:

```
scripts/bench/mac_ab.sh                        # 3 rounds of the profiles sim, q1-fast, q1-mid, q1-slow
scripts/bench/mac_ab.sh 3 "task-sim task"      # whole 66-point tasks (simulated, then real)
NEW_ENV="PJRT_NPROC=2" scripts/bench/mac_ab.sh # the working tree with more environment
NEW_PATCHES=scripts/bench/patches/split-jacobian.patch scripts/bench/mac_ab.sh 3 sim
COPIES=12 scripts/bench/mac_ab.sh 3 task-sim   # 12 tasks at once: the throughput of a machine BOINC fills
```

The first run makes the pinned Python environment in `~/skia-work/venv` (`setup_env.sh`: python3.14, `pip==25.3`,
`requirements.lock`; a few minutes). It then prepares two labels in `~/skia-work/trees/`: `base`, the `src/` and
`boinc/job.xml` of `upstream/main` (or `origin/main`; `BASE_REF=` picks another ref), and `new`, this working tree.
Each label is a copy of the code and a `run` script that starts `skia-app` the way the volunteer wrapper does: the
`<setenv>` lines of the label's own `boinc/job.xml` task 2, then `skia_app task.zip -o result.txt`. So a change of
job.xml is measured like a change of the code. Each profile ends with a table
(`~/skia-work/bench-runs/<date>-<profile>/summary.txt`, the runs in `results.tsv`):

```
label        runs  median s    min s    max s    cpu s  RSS MB  rows   vs 1st  cpu vs  verdict; output
base            3      7.11     7.07     7.15     11.3     492     6   1.000x  1.000x  baseline; same output
reuse           3      5.89     5.88     6.01      9.3     492     6   1.208x  1.211x  ACCEPT (+17.2 %); same output
```

`verdict` applies the rule below to wall time; `cpu vs` is the ratio of CPU time (user + system), which is what a
volunteer pays when every core runs a task. `same output` means every run of that label wrote the same `result.txt`
as the first label (the file has no timings, so the whole file is hashed); `rows` is the number of grid rows written
and must equal the profile's grid size. With no change in `src/` or `boinc/`, `new` equals `base`: run that once,
because its spread is the noise a real change has to beat.

On Linux, or in a container on any Docker host: `scripts/bench/linux_ab.sh [rounds] [profiles]` (default `sim q1-mid`)
with `ARCH=arm64` or `ARCH=amd64`. It prepares the labels on the host, then runs them in `python:3.14-slim-trixie` with
`requirements.lock` installed in `~/skia-work/linux-venv-<arch>` (made on the first run). Linux x86_64 is the platform
the volunteer app ships for; on Apple Silicon it runs under emulation (Rosetta), so use it for "runs and gives the
same output as base", not for times.

By hand, for other labels:

```
REF=upstream/main scripts/bench/build_variant.sh base                       # ~/skia-work/trees/base/run
ENV="PJRT_NPROC=1" scripts/bench/build_variant.sh one-thread                # working tree, more environment
scripts/bench/build_variant.sh split scripts/bench/patches/split-jacobian.patch
SKIA_PYTHON=~/skia-work/venv/bin/python scripts/bench/run_bench.sh sim 3 base=$HOME/skia-work/trees/base/run \
    one-thread=$HOME/skia-work/trees/one-thread/run
```

`build_variant.sh` copies the tracked files of `src/` and `boinc/` (modified ones as they are on disk; a new file must
be `git add -N`ed) or `git archive`s `REF`, so an untracked file cannot leak in; `JOBXML=0` leaves job.xml's
environment out. Each label has a `build.info` (code, patches, environment), which `run_bench.sh` copies into
`results.tsv` with the Python and package versions, the task bundle's md5 and the load.

## Method

### Two kinds of speed change

* **Same results**: work that is skipped because it is repeated exactly (SKI-003), thread settings that do not change
  the arithmetic (SKI-002, checked at every thread count from 1 to 64), less Python overhead around the same XLA
  computations. `result.txt` must stay byte-identical to the baseline's in the same environment. Such a change needs a
  measured gain, the output check and `make test`, nothing more.
* **Result-changing**: anything that changes a number in `result.txt`: another start vector, number of starts,
  tolerance or `max_nfev`, another bound or prior, another PSF or noise model, a coarser grid, and also an equivalent
  formula that XLA evaluates in another order. JAX graphs are fragile here: computing the same derivative through a
  differently shaped graph changed the Jacobian in the last bits (SKI-007), and a least-squares fit can carry that into
  the printed digits. Measure it the same way, report how far the rows moved, and treat it as a science change that the
  maintainers decide on; keep it as a patch in `scripts/bench/patches/` until then.

### Rules for every measurement

1. **Run both sides the same way.** Both labels run from source with the same Python environment and their own
   job.xml environment. The frozen volunteer app (its own Python, NumPy and jaxlib build) is not what is measured.
2. **Find out where the time goes first** (below): `python -m cProfile` for the Python side, `sample <pid>` on macOS
   (or `perf` on Linux) for the threads, and small JAX timings with `jax.block_until_ready`. In a profile of the app,
   time spent waiting for XLA shows up as `numpy.asarray`, which blocks on the asynchronously dispatched computation.
3. **Use a profile that exercises the change.** The six-point profiles show a change of the per-point work within
   seconds, but start-up (imports, about 1.2 s, and XLA compilation) weighs more in them than in a task: confirm on
   `task-sim` or `task` (66 points, a production bundle).
4. **One task and a full machine are different measurements.** XLA uses every core by default, so one task alone runs
   faster with threads; but BOINC counts it as one CPU and runs one task per core. Measure single runs for the time of a
   task and `COPIES=<cores>` for the throughput of a volunteer's machine, and look at `cpu s` in both.
5. **Interleave and repeat.** `run_bench.sh` runs A B C A B C ... in fresh directories (a fresh directory is also what
   keeps the app from resuming from a `result.txt.checkpoint.json`); use at least 3 rounds and compare medians.
6. **Accept by the rule, not by eye.** A candidate is faster when its median is at least 5 % below the baseline's and
   its slowest run beats the baseline's fastest, or when its median is at least 10 % below. Anything else is "no clear
   gain".
7. **Check the output of every run**, in the same environment: other platforms differ from each other in the last
   digits anyway (README: 1.5e-12 relative), and so do runs with other XLA thread settings unless job.xml's
   `--xla_cpu_multi_thread_eigen=false` is set (SKI-001).
8. **Measure on a quiet machine.** Pause the BOINC client and other heavy programs, keep a laptop on power, and look at
   the load the scripts record at the start and end of each profile. On a loaded machine an interleaved comparison is
   still fair, but its absolute times are not worth quoting.
9. **Write it down.** Date, machine, OS, Python and package versions, load, profile, rounds, medians and spread,
   verdict and output go into Results below; a platform that was not measured is called unmeasured.

## The pieces

| File | What it does |
|---|---|
| `scripts/bench/mac_ab.sh` | Makes the environment if needed, prepares `base` and `new`, runs the profiles |
| `scripts/bench/linux_ab.sh` | The same in a `python:3.14` container, arm64 or amd64 |
| `scripts/bench/setup_env.sh` | The pinned environment outside the repository |
| `scripts/bench/build_variant.sh` | One label: a copy of `src/` and `boinc/`, patches, and a `run` script with job.xml's environment |
| `scripts/bench/run_bench.sh` | Makes or reuses the profile's bundle, runs the labels interleaved, hashes `result.txt` |
| `scripts/bench/make_bench_data.py` | A profile's task bundle (simulated or real cutout, baseline fit as `boinc/make_tasks.py` does, grid), cached by the profile's text |
| `scripts/bench/measure.py` | Wall time, CPU time and peak memory of one run, the same on macOS and Linux |
| `scripts/bench/summarize_bench.py` | The summary table and the verdict |
| `scripts/bench/profiles/*.prof` | The profiles (below) |
| `scripts/bench/patches/` | Result-changing experiments, applied to a label's copy only |

### Inputs

| Profile | Cutout | Grid | What it is for |
|---|---|---|---|
| `sim` | simulated, 121x121 pixels of 0.1 arcsec, PSF FWHM 0.16 arcsec, as a Q1 cutout | 6 points: x -0.6; y -0.6, 0.6; `alpha_Rs` 0, 0.02, 0.05 | quick check without downloads |
| `q1-fast`, `q1-mid`, `q1-slow` | real Q1 VIS cutouts at about the 16th, 50th and 84th percentile of the B6 timing (`docs/b6-185/`); `q1-mid` is the cutout of the P3 smoke test | the 6 points of `sim` | the per-point cost on real data, which needs more optimizer steps than simulated data |
| `task-sim` | the cutout of `sim` | 66 points, the first bundle `boinc/make_tasks.py` makes for a lens | a whole task without downloads |
| `task` | the cutout of `q1-mid` | the 66 points of `task-sim` | a whole production task |

The baseline fit inside each bundle is the one of `boinc/make_tasks.py` (`FitConfig(gamma_prior_sigma=0.1)`). One
point in three has `alpha_Rs` 0, as in every production task. The real cutouts are public Euclid data but not in the
repository: `make_bench_data.py` looks for them in `$SKIA_DATA`, `data/q1_cutouts/` and `~/skia-work/data/q1/`, and
`docs/fetch_q1_cutouts.py` fetches them. A bundle is made once and kept in `~/skia-work/bench-data/`, so later runs
and the containers scan the same bytes; its md5 is in every `results.tsv`.

## Where the time goes (measured 2026-11-05, Apple M4 Pro)

`python -m cProfile` of `skia-app` with one XLA thread on the six-point simulated bundle (21.2 s): 18.5 s in
`numpy.asarray`, that is, waiting for XLA, of which 17.6 s under the Jacobian callback of `least_squares` (276 calls,
64 ms each); SVDs of the Jacobian 0.27 s, compilation 0.22 s, imports about 1.2 s. The model image alone takes 4.3 ms.
Inside the Jacobian (`jax.jacfwd` of the renderer over 15 parameters on the 258x258 oversampled grid, 61 ms):

| Piece, timed alone with one XLA thread | Time |
|---|---|
| EPL deflection (19-term series in complex numbers, `herculens`) | 1.6 ms |
| its forward-mode Jacobian over the 6 EPL parameters | 35.6 ms |
| Sersic source, and its Jacobian over 7 parameters | 0.7 ms, 4.5 ms |
| FFT convolution with the PSF, 1 image and 15 images | 0.7 ms, 7.6 ms |

So the derivatives of the EPL series dominate. `jacfwd` sends all 15 one-hot tangents through it, although only the
6 EPL parameters (8 lens parameters with the shear) change the source plane: the other 9 tangents are zeros pushed
through the series. That is SKI-007; doing it with 8 tangents gives the same derivatives but not the same bits.

Not in the time of a task: a third of its grid points (`alpha_Rs` 0) gave the baseline fit again (SKI-003), and with
XLA's default of one thread per core a task used about 1.6 cores (`sample`: twelve worker threads running ducc0 FFTs
and fused XLA kernels; SKI-002).

## Candidate speed-ups

| # | Change | Kind | State |
|---|---|---|---|
| 1 | Reuse the baseline fit for zero-field grid points (SKI-003) | same results | on this branch, measured below |
| 2 | One XLA CPU thread, `PJRT_NPROC=1` in job.xml (SKI-002) | same results with job.xml's flags | on this branch, measured below |
| 3 | Lens tangents only through the lens equation (SKI-007) | result-changing (last bits) | `patches/split-jacobian.patch`, measured below |
| 4 | Fewer tangents still: the 6 EPL parameters through the series and the shear by hand | result-changing | not tried |
| 5 | The real-arithmetic EPL of herculens (`no_complex_numbers=True`) | result-changing (another formula) | not tried |
| 6 | A persistent XLA compilation cache, fewer imports | same results | not tried: start-up is about 2 % of a task |
| 7 | Unpack the app once per app version instead of once per task (SKI-006) | same results | not measured (the archive is not in this repository) |

## What the suite does not cover

* The frozen volunteer app and the BOINC wrapper: their start-up, the unpacking of `skia_app.tar.gz` (SKI-006) and the
  build's own library versions.
* Checkpoint and resume (the test suite and docs/P3_DETERMINISM.md do that), progress reporting (SKI-005).
* Windows and macOS builds, which do not exist yet; Linux x86_64 natively (only under emulation here).
* The statistics: whether a speed-up that moves the last digits changes any conclusion of the reports in `docs/`.

## Results

### Apple M4 Pro, 2026-11-05: reuse of the baseline fit, one XLA thread, the split Jacobian

Machine: Apple M4 Pro (8 performance + 4 efficiency cores, 24 GB), macOS 27.0, on power, BOINC client's tasks
suspended; load average 2.2 to 3.8 at the start of the single-task runs (this Mac's idle level). Python 3.14.5
(Homebrew), jax and jaxlib 0.12.0.1, NumPy 2.5.4, SciPy 1.18.1, herculens 0.3.0, lenstronomy 1.13.2
(`requirements.lock`). Every run with its label's job.xml environment; 3 rounds, interleaved; times are medians, and the
spread (max - min) of every label was at most 3.2 % of its median in the single-task runs (the `results.tsv` files). The Q1 cutouts were cut from IRSA's copy of
the same MER mosaics (`EUC_MER_BGSUB-MOSAIC-VIS`), 121x121 pixels, because ESA's archive was switching to DR1 that
afternoon: `q1-fast`'s cutout gives the maintainer's sky and noise estimates to every digit and the same baseline
`nfev` (64) as `docs/b6-185/`; `q1-mid` and `q1-slow` are one column narrower than ESA's 122x121.

Labels: `base` is `upstream/main` (a88de16); `reuse` is base with the zero-field reuse of `scanner.py` (SKI-003);
`nproc1` is base with `PJRT_NPROC=1` (SKI-002); `both` is this branch (both changes, and job.xml without the ignored
`intra_op_parallelism_threads=1`).

One task at a time (wall time, and CPU time = user + system):

| Profile | base wall / CPU | reuse | nproc1 | both | output |
|---|---|---|---|---|---|
| `sim` (6 points) | 7.11 s / 11.3 s | 5.89 s (1.21x, ACCEPT) / 9.3 s | 10.03 s / 10.4 s | 8.18 s / 8.5 s | same in all |
| `q1-fast` | 15.65 / 26.3 | 13.05 (1.20x, ACCEPT) / 21.6 | 23.69 / 24.1 | 19.65 / 19.9 | same |
| `q1-mid` | 26.46 / 44.8 | 21.81 (1.21x, ACCEPT) / 36.8 | 40.94 / 41.4 | 33.57 / 34.0 | same |
| `q1-slow` | 16.51 / 27.8 | 13.63 (1.21x, ACCEPT) / 22.7 | 25.17 / 25.6 | 20.59 / 21.0 | same |
| `task-sim` (66 points) | 62.74 / 103.9 | 49.69 (1.26x, ACCEPT) / 82.1 | 97.04 / 97.8 | 76.33 / 77.0 | same |
| `task` (66 points, real) | 304.39 / 523.9 | 253.63 (1.20x, ACCEPT) / 435.7 | not run | 401.14 / 402.5 | same |

* **Reuse (SKI-003), same results:** 1.20x to 1.26x faster in wall and CPU time on every profile (`task`: a separate
  run of `base` against `reuse`, 303.89 s against 253.63 s). A third of the points
  have `alpha_Rs` 0; their fits start at the optimum and converge in fewer steps than the others, so the saving is less
  than a third.
* **One XLA thread (SKI-002), same results:** alone, a task is 1.5x slower in wall time, because `base` spreads its
  work over the idle cores, and uses 6 to 9 % less CPU time. `both` uses 1.30x to 1.35x less CPU time than `base` on
  every profile (on the real task 402.5 s against 523.9 s).

Twelve tasks at once (`COPIES=12`, every core busy, as the BOINC client runs one-CPU tasks), `task-sim`, time for the
batch of twelve:

| Label | Batch | CPU (sum) | Verdict |
|---|---|---|---|
| `base` | 230.55 s (216.5-231.0) | 2489 s | |
| `nproc1` | 220.34 s (212.5-223.0) | 2425 s | 1.046x, no clear gain |
| `reuse` | 179.38 s (171.4-182.6) | 1944 s | 1.285x, ACCEPT |
| `both` | 170.94 s (168.1-180.3) | 1902 s | 1.349x, ACCEPT |

All twelve copies of every run wrote the same `result.txt` as `base`. On a full machine the extra threads of `base` buy
nothing (the load average reached 195), so one thread is a small gain there and its main effect is that a task uses
the one CPU BOINC gives it; together with the reuse, a volunteer's machine finishes 1.35x more tasks. Per task, CPU
time doubles against a task alone: the efficiency cores and the lower clock with every core busy.

**Thread count and results (SKI-001).** On the `sim` bundle, without `XLA_FLAGS`, `PJRT_NPROC` 1, 2, 4, 6, 16 and 64
gave one `result.txt` and 8, 12 (this Mac's default) and 32 another. With job.xml's
`--xla_cpu_multi_thread_eigen=false`, 1 to 64 threads gave the same file, the one of a single thread. So job.xml's
flag is what keeps volunteers' results independent of their core count, and `PJRT_NPROC=1` does not change them.

**Split Jacobian (SKI-007, result-changing), `patches/split-jacobian.patch` on top of `both`:**

| Profile | both | split | Rows that moved |
|---|---|---|---|
| `sim` | 8.08 s | 6.93 s (1.17x, ACCEPT) | 1 of 6, by 1e-10 in `delta_chi2` (the last printed digit) |
| `q1-mid` | 33.24 s | 28.22 s (1.18x, ACCEPT) | 3 of 6, up to 2.7e-8 in chi2 (1.2e-12 relative; 3.2e-11 relative in `delta_chi2`) |
| `task-sim` | 74.85 s | 61.59 s (1.22x, ACCEPT) | 4 of 66, by 1e-10 |

No convergence flag changed; the differences are within the validator's tolerance and about the size of the
cross-platform difference README reports, but they are differences: a decision for the maintainers.

**XLA flags, same results?** The Jacobian of the `q1-mid` bundle with one thread and job.xml's flag took 64.1 ms;
`--xla_cpu_use_thunk_runtime=false`, `--xla_cpu_enable_concurrency_optimized_scheduler=true` and `=false`,
`--xla_cpu_use_xnnpack=false` and `=true`, `--xla_cpu_prefer_vector_width=128` and `=512`, `--xla_cpu_use_onednn=false`
and `--xla_cpu_enable_fast_min_max=true` each gave 63.9 to 64.4 ms and the same Jacobian bytes: nothing to gain there. An unknown flag ends the process with exit code 1.
`--xla_cpu_disable_loop_unrolling=true` and `--xla_cpu_disable_slp_vectorizer=true` are not flags of this jaxlib:
the process stopped with "Unknown flag in XLA_FLAGS".

### Linux arm64 in a container on the same Mac, 2026-11-05

`ARCH=arm64 scripts/bench/linux_ab.sh 3 "sim q1-mid task-sim"`: Docker Desktop 29.9.2 (12 CPUs), image
`python:3.14-slim-trixie` (`sha256:af8a5aa14780...`), Python 3.14.10, the same packages; `base` is `upstream/main`,
`new` this branch. Load average in the VM 0 to 1.6.

| Profile | base wall / CPU | new wall / CPU | output |
|---|---|---|---|
| `sim` | 8.21 s / 13.0 s | 9.30 s / 9.5 s (1.38x less CPU) | same |
| `q1-mid` | 30.96 / 52.5 | 38.12 / 38.4 (1.37x) | same |
| `task-sim` | 73.82 / 124.7 | 86.98 / 87.3 (1.43x) | same |

As on macOS: the same output, one task alone slower on one thread, 1.37x to 1.43x less CPU per task. Between the
platforms the bytes differ slightly: `sim` gave the same file on macOS and Linux arm64, `task-sim` differed in 1 row
of 66 by 1e-10, `q1-mid` in 5 rows of 6 by up to 8.1e-8 in chi2 (3.6e-12 relative), with the same CPU and the same
jaxlib (README: 1.5e-12 between the platforms the maintainer measured). That is more than the split Jacobian moves the
rows on one platform.

### Linux x86_64 under emulation, 2026-11-05

`ARCH=amd64 scripts/bench/linux_ab.sh 1 "sim q1-mid"`: the same image for linux/amd64 under Rosetta ("VirtualApple @
2.50GHz", 12 CPUs), Python 3.14.10, the same packages. This is the platform the volunteer app ships for, but emulated:
the times say nothing about real x86_64 hardware.

| Profile | base wall / CPU | new wall / CPU | output |
|---|---|---|---|
| `sim` | 22.85 s / 35.0 s | 26.37 s / 26.9 s | same |
| `q1-mid` | 82.71 / 140.7 | 108.79 / 109.6 | same |

Both labels run and write the same `result.txt`, and the new label uses one core (CPU time equals wall time), so
`PJRT_NPROC` is read by the x86_64 jaxlib too. Not measured: Linux x86_64 on real hardware, and the frozen volunteer
app with its own library builds.

### Next steps (plan of 2026-11-05)

1. Maintainers: review SKI-003 (reuse) and SKI-002 (one XLA thread); decide SKI-007 (split Jacobian) and SKI-001
   (thread flag for the CLIs).
2. Rebuild the volunteer app from this branch and check on a Linux x86_64 host that a task uses one core and gives the
   `result.txt` of the old app (same environment, same bundle).
3. Measure SKI-006 (the per-task unpacking) with the real archive; wire SKI-004 and SKI-005 into job.xml and test them
   with a client.
4. Further speed is in the Jacobian's EPL tangents and is result-changing: try 6 tangents through the series with the
   shear by hand, and herculens' real-arithmetic EPL, as patches.

