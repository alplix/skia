# B5 full Q1 baseline and scan robustness pass

## Protocol and overall result

All 24 supplied Q1 cutouts were loaded with the same outer-border MAD noise
estimate and assumed Gaussian 0.16 arcsec PSF, then fit with one start,
`Normal(gamma=2, sigma=0.1)`, and `max_nfev=80`. Each baseline ran the same
12-point 2x2x3 TNFW grid used in P2. The complete output records per-image
adapter metadata, baseline optimizer status, every scan point, all bounds,
and every non-converged grid point in
[`b5-real-robustness.jsonl`](b5-real-robustness.jsonl); the failure index is
[`b5-real-robustness.summary.json`](b5-real-robustness.summary.json).

- Input/adapter hard errors: 0/24.
- Baseline fits: 20/24 converged; 4/24 stopped at the 80-evaluation cap.
- Scan fits: 253/288 converged; 35/288 stopped at the same cap (status 0,
  `The maximum number of function evaluations is exceeded.`).
- Boundary contacts: 10 baseline fits and 99/288 grid fits.
- Median baseline image chi-square/dof: 4.913. This is not a calibrated statistic.
- Three severe residual cases have baseline chi-square/dof above 100:
  `102158275_2709777393645726835_VIS.fits` (34201.0),
  `102159192_2699004780662092340_VIS.fits` (200.9), and
  `102159773_2677079122669151496_VIS.fits` (130.3). Their fits are finite;
  cause is unresolved model/noise/PSF mismatch, not assigned to source structure.

## Baseline failures

All four are optimizer evaluation-limit failures, not file or adapter errors:

| Cutout | nfev | Status | Other diagnostics |
|---|---:|---:|---|
| `102019594_NEG620978815501656388_VIS.fits` | 80 | 0 | 5 bound contacts; chi-square/dof 3.658 |
| `102044188_NEG548070624278595332_VIS.fits` | 80 | 0 | lens theta_E bound; chi-square/dof 31.568 |
| `102045468_NEG534525411271636586_VIS.fits` | 80 | 0 | 4 bound contacts; chi-square/dof 1.660 |
| `102159196_2750719383660442039_VIS.fits` | 80 | 0 | 4 bound contacts; chi-square/dof 2.635 |

## Scan failures by cutout

Every failed grid fit has status 0, nfev 80, and the max-evaluations message.
The raw summary JSON lists each x/y/alpha point and boundary parameters.

| Cutout | Unconverged points / 12 |
|---|---:|
| `102019127_NEG598523367504401932_VIS.fits` | 2 |
| `102019586_NEG564463213499964061_VIS.fits` | 1 |
| `102020528_NEG554976975487853956_VIS.fits` | 1 |
| `102021984_NEG609678648473892045_VIS.fits` | 1 |
| `102023481_NEG647245161459048846_VIS.fits` | 5 |
| `102044188_NEG548070624278595332_VIS.fits` | 11 |
| `102045468_NEG534525411271636586_VIS.fits` | 2 |
| `102158893_2720583810655664894_VIS.fits` | 4 |
| `102159189_2667596748662187959_VIS.fits` | 1 |
| `102159192_2695197869661844158_VIS.fits` | 2 |
| `102159773_2677079122669151496_VIS.fits` | 3 |
| `102160335_2682061397679348286_VIS.fits` | 1 |
| `102160340_2740784519677600444_VIS.fits` | 1 |

## Bugs and unresolved causes

No adapter, shape, serialization, checkpoint, or scan-order code defect appeared
in this all-24 run, so no bug fix was indicated by these outcomes. The explicit
optimizer failures are evaluation-budget exhaustion. Bound contacts are search
domain/model limitations. High chi-square is a residual-model discrepancy; with
no measured Q1 PSF or RMS map and a single-Sersic source, its physical cause
cannot be isolated. No result was removed, reinitialized from truth, or retried
with a per-lens budget. The Q1 mass-sensitivity no-go remains unchanged.

Reproduce with `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
.venv/bin/python tests/bench/real_robustness.py --output docs/b5-real-robustness.jsonl`.
