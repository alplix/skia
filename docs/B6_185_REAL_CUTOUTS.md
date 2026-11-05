# B6 robustness on the 185 Q1 cutouts

## Scope and fixed configuration

The current Q1 set contains 185 grade-A candidate cutouts with Euclid reference
models. It contains no non-lens controls. The 32 files listed in
[`b6-new32-files.txt`](b6-new32-files.txt) were analyzed first, then all 185
manifest entries were analyzed in one run. The older 153-cutout outputs were
left intact. An exact check found that all 153 old file names are present in
the 185 run and their baseline chi-square, fitted Einstein radius, and best
scan delta-chi-square are unchanged.

Each run uses the same fixed configuration: one fit start, seed 731,
`max_nfev=80`, `gamma_prior_sigma=0.1`; then a 12-point TNFW grid with
`Rs=0.2 arcsec`, truncation radius `0.8 arcsec`, x and y in `{-0.3, 0.3}`
arcsec, and `alpha_Rs` in `{0, 0.02, 0.05} arcsec`. No per-cutout tuning,
manual initialization, removal, or rerun was done. The extra 32 were selected
from the existing manifest and were already fetched; this analysis made no
network requests.

## Robustness outcomes

| Run | Cutouts | Baseline converged | Scan points converged | Scan boundary contacts | Hard errors |
|---|---:|---:|---:|---:|---:|
| New 32 only | 30/32 | 348/384 | 104 | 0 |
| All 185 | 161/185 | 1,917/2,220 | 676 | 0 |
| Earlier 153 reference | 131/153 | 1,569/1,836 | 572 | 0 |

Every non-convergence in these runs has the same optimizer cause: the fixed
maximum of 80 function evaluations was reached (`status=0`, “The maximum number
of function evaluations is exceeded.”). There were 2 affected baselines and
36 scan grid points in the new-32 run; 24 affected baselines and 303 scan grid
points in the 185 run. The runner recorded the file, grid coordinates,
strength, status, evaluation count, message, boundary contacts, and chi-square
for every affected row. No FITS, manifest, noise-estimation, or other hard
exception occurred. These remaining cases are unresolved optimizer-limit
failures, not bugs that could be corrected without changing the fixed
configuration.

The 185 run's optimizer and boundary rates are similar to the historical 153
run. The added subset itself had fewer non-convergences (2/32 baselines and
36/384 grid points), but it is small and should not be taken as evidence that
the optimizer failure mode has been solved.

## Raw outputs

All per-cutout outcomes and exact aggregate summaries are retained under
[`b6-new32/`](b6-new32/) and [`b6-185/`](b6-185/). These folders contain both
the robustness and injection-recovery JSONL and summary JSON files. The
injection study and its interpretation are reported separately in
[`P2_185_INJECTIONS.md`](P2_185_INJECTIONS.md). The comparison values in the
table above can be reproduced with:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/real_robustness.py \
  --files-list docs/b6-new32-files.txt \
  --output docs/b6-new32/real-robustness.jsonl

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0 \
  .venv/bin/python tests/bench/real_robustness.py \
  --output docs/b6-185/real-robustness.jsonl
```
