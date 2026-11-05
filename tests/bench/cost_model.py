"""Serial angular-grid cost benchmark; writes machine-specific timing JSON."""
import os

# Set common numerical-library thread limits before importing NumPy/JAX.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_name] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from skia.cutout import load_cutout
from skia.fitter import FitConfig, fit_cutout, save_fit
from skia.scanner import scan
from skia.simulator import Observation, simulate, save_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='work/p2-cost.json')
    parser.add_argument('--points-per-axis', type=int, default=2)
    parser.add_argument('--max-nfev', type=int, default=120)
    args = parser.parse_args(argv)
    if args.points_per_axis < 1 or args.max_nfev < 1:
        parser.error('points-per-axis and max-nfev must be positive')

    work = Path('work/p2-cost-benchmark')
    work.mkdir(parents=True, exist_ok=True)
    cutout_path = work / 'simulated-cutout.npz'
    save_simulation(simulate(observation=Observation(npix=30), seed=20261031), cutout_path)
    cutout = load_cutout(cutout_path)
    config = FitConfig(starts=1, seed=731, max_nfev=args.max_nfev, gamma_prior_sigma=.1)

    # First call absorbs JAX compilation so reported fit timings measure steady-state work.
    fit_cutout(cutout, config)
    start = time.perf_counter()
    baseline_fit = fit_cutout(cutout, config)
    baseline_seconds = time.perf_counter() - start
    fit_path = work / 'baseline.json'
    save_fit(baseline_fit, fit_path)
    baseline = json.loads(fit_path.read_text(encoding='utf-8'))

    axis = np.linspace(-.4, .4, args.points_per_axis).tolist()
    strengths = np.linspace(0, .03, args.points_per_axis).tolist()
    grid = {'profile': 'TNFW', 'Rs_arcsec': .2, 'r_trunc_arcsec': .8,
            'x_arcsec': axis, 'y_arcsec': axis, 'alpha_Rs_arcsec': strengths}
    scan(cutout, baseline, grid)  # compile/warm all scan shapes
    start = time.perf_counter()
    result = scan(cutout, baseline, grid)
    scan_seconds = time.perf_counter() - start
    points = len(result['results'])
    affinity_supported = hasattr(os, 'sched_getaffinity')
    payload = {
        'format': 'skia-cost-v1', 'machine': platform.platform(),
        'processor': platform.processor() or platform.machine(),
        'python': platform.python_version(), 'jax_backend': 'cpu',
        'numerical_thread_limits': 1, 'serial_process': True,
        'cpu_affinity_api_available': affinity_supported,
        'cpu_affinity_enforced': False,
        'pixel_shape': list(cutout.image.shape), 'usable_pixels': int(cutout.mask.sum()),
        'starts': config.starts, 'max_nfev': config.max_nfev,
        'grid_shape': [len(axis), len(axis), len(strengths)], 'grid_points': points,
        'baseline_seconds': baseline_seconds, 'full_scan_seconds_including_baseline': scan_seconds,
        'estimated_grid_seconds_per_point': max(0., scan_seconds-baseline_seconds)/points,
        'estimated_lens_seconds_for_grid': baseline_seconds+max(0., scan_seconds-baseline_seconds),
        'fits_converged': sum(bool(row['converged']) for row in result['results']),
        'timings_warm_jax_cache': True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
