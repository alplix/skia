"""Reproducible parameter recovery and fit timing; no real data or hidden refits."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '1')

import jax
import numpy as np

from skia.cutout import Cutout
from skia.fitter import FitConfig, PARAMETER_NAMES, fit_cutout, parameter_vector, physical_parameters
from skia.simulator import Lens, Source, Observation, gaussian_psf, simulate

LENS_RANGES = {
    'theta_E': (.85, 1.15), 'gamma': (1.8, 2.2), 'e1': (.05, .20), 'e2': (-.08, .08),
    'center_x': (-.03, .03), 'center_y': (-.03, .03), 'gamma1': (-.03, .05), 'gamma2': (-.03, .05),
}
SOURCE_RANGES = {
    'amp': (4, 18), 'R_sersic': (.13, .23), 'n_sersic': (.8, 1.6), 'e1': (-.12, .20),
    'e2': (-.15, .15), 'center_x': (-.12, .12), 'center_y': (-.12, .12),
}


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def _one(lens: Lens, source: Source, observation: Observation, index: int, seed: int) -> dict:
    truth = physical_parameters(parameter_vector(lens, source))
    row = {'index': index, 'noise_seed': seed, 'truth': truth}
    try:
        sim = simulate(lens, source, observation, seed=seed)
        cutout = Cutout(sim.image, sim.noise_map, sim.mask,
                        gaussian_psf(observation.pixel_scale/observation.oversample, observation.psf_fwhm),
                        observation.pixel_scale, observation.exposure, observation.oversample)
        fit = fit_cutout(cutout, FitConfig())
        row.update({'parameters': fit.parameters, 'converged': fit.converged,
                    'boundary_parameters': fit.boundary_parameters, 'chi2': fit.chi2, 'dof': fit.dof,
                    'elapsed_seconds': fit.elapsed_seconds, 'nfev': fit.nfev, 'njev': fit.njev,
                    'best_start': fit.best_start, 'attempts': fit.attempts,
                    'status': fit.status, 'message': fit.message})
        row['gross_recovery_failure'] = (abs(fit.parameters['lens.theta_E']-truth['lens.theta_E']) > .03
                                         or abs(fit.parameters['lens.gamma']-truth['lens.gamma']) > .15)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        row.update({'parameters': None, 'converged': False, 'gross_recovery_failure': True,
                    'boundary_parameters': [], 'error': str(error)})
    return row


def _metrics(rows: list[dict]) -> dict:
    finite = [r for r in rows if r['parameters'] is not None]
    clean = [r for r in finite if r['converged'] and not r['boundary_parameters']]
    metrics = {}
    for label, subset in [('all_finite', finite), ('converged_interior', clean)]:
        values = {}
        for name in PARAMETER_NAMES:
            errors = np.array([r['parameters'][name]-r['truth'][name] for r in subset])
            values[name] = (None if not errors.size else {
                'bias': float(errors.mean()), 'scatter': float(errors.std(ddof=0)),
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'p16': float(np.quantile(errors, .16)), 'p84': float(np.quantile(errors, .84)),
            })
        metrics[label] = {'count': len(subset), 'parameters': values}
    times = [r['elapsed_seconds'] for r in finite]
    chi = [r['chi2']/r['dof'] for r in finite]
    metrics.update({'count': len(rows), 'finite_count': len(finite),
                    'hard_failures': sum(r['parameters'] is None for r in rows),
                    'optimization_failures': sum(not r['converged'] for r in rows),
                    'boundary_count': sum(bool(r['boundary_parameters']) for r in rows),
                    'gross_recovery_failures': sum(r['gross_recovery_failure'] for r in rows),
                    'timing_median_seconds': float(np.median(times)) if times else None,
                    'timing_p95_seconds': float(np.quantile(times, .95)) if times else None,
                    'chi2_per_dof_median': float(np.median(chi)) if chi else None})
    return metrics


def _report(output: Path, summary: dict, bench: list[dict]) -> None:
    n = summary['count']
    lines = ['# P1 simulated smooth-model report', '',
             f'Protocol v1; {n} recovery draws. ' + ('FULL 200+ LENS RUN.' if n >= 200 else 'SMOKE TEST ONLY; does not satisfy task 1.4.'), '',
             'Matched-model synthetic data only. No Q1 cutouts, scientific validation, subhalo detection or cross-platform claims.', '',
             '## Reproduction', '',
             f'`JAX_PLATFORMS=cpu .venv/bin/python tests/bench/recovery.py --count {n} --bench-count {len(bench)} --output {output.as_posix()}`', '',
             'Protocol and population ranges are in P1_PROTOCOL.md; exact configuration/environment are in P1_protocol.json.',
             'Raw per-draw truth, estimates, optimizer attempts and failures: P1_recovery.jsonl. Summary with RMSE/error percentiles: P1_summary.json.',
             'The generator supplies an ideal expected-count noise map. All 15 parameters are fitted with two fixed starts; no lens is discarded or retuned.', '',
             '## Outcomes (all attempted draws)', '',
             '| Outcome | Count | Rate |', '|---|---:|---:|']
    for label, key in [('Finite results', 'finite_count'), ('Hard numerical failures', 'hard_failures'),
                       ('Optimization failures (including hard failures)', 'optimization_failures'),
                       ('Fits touching a bound', 'boundary_count'), ('Gross recovery failures', 'gross_recovery_failures')]:
        lines.append(f"| {label} | {summary[key]}/{n} | {summary[key]/n:.2%} |")
    lines += ['', 'Optimization failure means the selected best result is unconverged or no finite fit is returned.',
              'Bound contacts are counted separately. Gross recovery failure means |theta_E error| > 0.03 arcsec',
              'or |gamma error| > 0.15, or a hard numerical failure. These are predeclared diagnostic thresholds, not statistical significances.', '']
    for label, title in [('all_finite', 'All finite returned fits (including convergence/bound failures)'),
                         ('converged_interior', 'Converged fits with no bound contacts')]:
        group = summary[label]
        lines += [f'## {title}', '', f"Denominator: {group['count']}/{n} draws. Errors are estimate minus truth. Scatter is population std (ddof=0).", '',
                  '| Parameter | Bias | Scatter | RMSE |', '|---|---:|---:|---:|']
        for name, stats in group['parameters'].items():
            if stats is None:
                lines.append(f'| {name} | unavailable | unavailable | unavailable |')
            else:
                lines.append(f"| {name} | {stats['bias']:.7g} | {stats['scatter']:.7g} | {stats['rmse']:.7g} |")
        lines += ['', 'Angles/radii are arcsec; source amplitude is e-/s/arcsec²; remaining parameters dimensionless.', '']
    lines += ['## Fit quality and timing', '',
              f"Median chi2/dof among all finite fits: {summary['chi2_per_dof_median']}.",
              f"40 x 40 fit wall time median: {summary['timing_median_seconds']} s; p95: {summary['timing_p95_seconds']} s.",
              'The first fit of each shape includes JAX compilation; time excludes imports, simulation and file I/O.',
              'Wall time covers all optimizer starts. Numerical-library thread settings are recorded; JAX CPU execution is not certified single-core.', '',
              '| 100 x 100 timing sample | Converged | Fit wall time (s) |', '|---|---|---:|']
    for row in bench:
        lines.append(f"| {row['index']} | {row['converged']} | {row.get('elapsed_seconds', 'failed')} |")
    warm = [r['elapsed_seconds'] for r in bench[1:] if 'elapsed_seconds' in r]
    if warm:
        lines += ['', f'100 x 100 subsequent-fit median ({len(warm)} samples): {float(np.median(warm))} s.']
    lines += ['', '## Interpretation and remaining work', '',
              'These errors characterize only the declared small, relatively simple simulated population and search box.',
              'The mass slope/source size/brightness and ellipticity/shear can be degenerate. Convergence and chi2 alone',
              'do not prove parameter accuracy; consult gross errors and the all-finite table rather than selecting successful cases.',
              'Simulator/fitter profile agreement is not an independent lensing-method cross-check. Noise, source family and PSF are matched and idealized.',
              'The 40 x 40 recovery field differs from default 100 x 100 cutouts; the latter have only a timing sample here.',
              'Real-cutout testing (1.5), real/synthetic injection calibration, alternative-method checks, coverage studies,',
              'cross-machine tolerances and portable binaries remain outstanding. P1 is not complete and P2 is not authorized by a review.', '',
              '![Recovery diagnostics](P1_recovery.png)', '']
    (output/'P1_REPORT.md').write_text('\n'.join(lines))


def _plot(output: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finite = [r for r in rows if r['parameters'] is not None]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for column, name in enumerate(('lens.theta_E', 'lens.gamma')):
        truth = np.array([r['truth'][name] for r in finite])
        estimate = np.array([r['parameters'][name] for r in finite])
        bad = np.array([not r['converged'] or bool(r['boundary_parameters']) for r in finite], dtype=bool)
        axes[0, column].scatter(truth[~bad], estimate[~bad], s=14, alpha=.7, label='Converged interior')
        axes[0, column].scatter(truth[bad], estimate[bad], marker='x', s=22, c='tab:red', label='Unconverged / boundary')
        if truth.size:
            lo, hi = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
            axes[0, column].plot([lo, hi], [lo, hi], color='black', lw=1)
        axes[0, column].set(xlabel=f'True {name}', ylabel=f'Estimated {name}')
        axes[0, column].legend(fontsize=8)
        axes[1, column].hist(estimate-truth, bins=25, color='steelblue', edgecolor='white')
        axes[1, column].axvline(0, color='black', lw=1)
        axes[1, column].set(xlabel=f'{name}: estimate minus truth', ylabel='Finite fits')
    fig.suptitle(f'Skia P1: {len(rows)} simulated draws, {len(finite)} finite fits; matched-model test')
    fig.savefig(output/'P1_recovery.png', dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run an immutable recovery dataset/report; refuses to overwrite existing raw results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--bench-count', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('docs'))
    args = parser.parse_args(argv)
    if args.count < 1 or args.bench_count < 0:
        parser.error('count >= 1 and bench-count >= 0 are required')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'P1_recovery.jsonl').exists():
        parser.error('raw results already exist; choose a new output directory')
    observation = Observation(npix=40)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    protocol = {'format': 'skia-recovery-v1', 'count': args.count, 'bench_count': args.bench_count,
                'parameter_seed': 20261031, 'noise_seed_base': 100000, 'benchmark_seed_base': 200000,
                'lens_ranges': LENS_RANGES, 'source_ranges': SOURCE_RANGES,
                'observation': asdict(observation), 'fit_config': asdict(FitConfig()),
                'git_revision': revision, 'source_sha256': {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*sorted((root/'src/skia').glob('*.py')), Path(__file__).resolve()]},
                'python': platform.python_version(), 'os': platform.system(), 'machine': platform.machine(),
                'packages': {p: version(p) for p in ('jax', 'jaxlib', 'herculens', 'lenstronomy', 'numpy', 'scipy')},
                'jax_backend': jax.default_backend(),
                'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'XLA_FLAGS')},
                'gross_error_thresholds': {'lens.theta_E': .03, 'lens.gamma': .15}}
    _json(output/'P1_protocol.json', protocol)
    rng = np.random.Generator(np.random.PCG64(20261031))
    rows = []
    with (output/'P1_recovery.jsonl').open('x') as handle:
        for i in range(args.count):
            lens = Lens(**{k: float(rng.uniform(*v)) for k, v in LENS_RANGES.items()})
            source = Source(**{k: float(rng.uniform(*v)) for k, v in SOURCE_RANGES.items()})
            row = _one(lens, source, observation, i, 100000+i)
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            handle.flush()
            if (i+1) % 20 == 0 or i+1 == args.count:
                print(f"recovery {i+1}/{args.count}; optimization_failures={sum(not r['converged'] for r in rows)}", flush=True)
    bench = [_one(Lens(), Source(), Observation(), i, 200000+i) for i in range(args.bench_count)]
    summary = _metrics(rows)
    _json(output/'P1_summary.json', summary)
    _json(output/'P1_timing.json', bench)
    _report(output, summary, bench)
    _plot(output, rows)
    print(f"RECOVERY SUMMARY: {args.count} attempted, {summary['finite_count']} finite, "
          f"{summary['optimization_failures']} optimization failures, {summary['boundary_count']} boundary fits, "
          f"{summary['gross_recovery_failures']} gross recovery failures", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
