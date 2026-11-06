#!/usr/bin/env python3
"""Make the task bundle (skia-task-v1 ZIP) of one benchmark profile, the input every label of a run gets.

usage: make_bench_data.py <profile file> <cache directory>
Prints the path of the bundle. The bundle is made once and kept in <cache directory>/<profile name>-<hash of the
profile text>/task.zip, so every later run, on this machine or in a container that sees the same directory, scans the
same bytes. Delete that directory to make it again (the baseline fit inside is made with the code of this checkout,
so a bundle made after a change of the fitter can differ from an older one: run_bench.sh records its md5).

A profile is `key value` lines, `#` starts a comment:
  cutout sim                one simulated cutout shaped like a Q1 VIS cutout (121x121 pixels of 0.1 arcsec, PSF FWHM
                            0.16 arcsec sampled at half pixels), electrons; keys seed, theta_E, gamma (defaults below)
  cutout q1 <file name>     a real Euclid Q1 VIS cutout (docs/DATA.md, docs/fetch_q1_cutouts.py), read with
                            skia.realdata.load_q1_fits as boinc/make_tasks.py does; looked up in $SKIA_DATA, then the
                            repository's data/q1_cutouts/, then $SKIA_WORK/data/q1
  x, y, alpha               the grid axes in arcsec (x_arcsec, y_arcsec, alpha_Rs_arcsec), space-separated
The baseline fit is the one of boinc/make_tasks.py (FitConfig(gamma_prior_sigma=0.1): 2 starts, seed 731, at most 200
evaluations per start), and so is the TNFW shape (Rs 0.2, r_trunc 0.8 arcsec).
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')


def read_profile(path):
    keys = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, _, value = line.partition(' ')
            keys[key] = value.split()
    for key in ('cutout', 'x', 'y', 'alpha'):
        if key not in keys:
            sys.exit(f'{path}: no "{key}" line')
    return keys


def find_q1(name):
    repo = Path(__file__).resolve().parents[2]
    places = [os.environ.get('SKIA_DATA'), repo / 'data' / 'q1_cutouts',
              Path(os.environ.get('SKIA_WORK', Path.home() / 'skia-work')) / 'data' / 'q1']
    for place in places:
        if place and (Path(place) / name).is_file():
            return Path(place) / name
    sys.exit(f'{name} not found in $SKIA_DATA, data/q1_cutouts or $SKIA_WORK/data/q1: fetch it with '
             'docs/fetch_q1_cutouts.py (public Euclid Q1 data, about 66 kB) or skip this profile')


def make(keys, out):
    import numpy as np
    import yaml
    from skia.cutout import load_cutout, save_cutout
    from skia.fitter import FitConfig, fit_cutout, save_fit
    from skia.task_bundle import create_task_bundle

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kind = keys['cutout'][0]
        if kind == 'sim':
            from skia.simulator import Lens, Observation, Source, save_simulation, simulate
            obs = Observation(npix=121, pixel_scale=0.1, psf_fwhm=0.16, oversample=2)
            lens = Lens(theta_E=float(keys.get('theta_E', ['0.7'])[0]), gamma=float(keys.get('gamma', ['2.0'])[0]))
            sim = simulate(lens=lens, source=Source(amp=3.0, R_sersic=0.15), observation=obs,
                           seed=int(keys.get('seed', ['20261105'])[0]))
            save_simulation(sim, tmp / 'sim.npz')
            cutout = load_cutout(tmp / 'sim.npz')
        elif kind == 'q1':
            from skia.realdata import load_q1_fits
            cutout, _, _ = load_q1_fits(find_q1(keys['cutout'][1]))
        else:
            sys.exit(f'unknown cutout kind {kind}: sim or q1')
        fit = fit_cutout(cutout, FitConfig(gamma_prior_sigma=.1))
        if not fit.converged:
            print(f'warning: the baseline fit did not converge ({fit.message}); boinc/make_tasks.py would skip this '
                  'cutout, the benchmark uses it anyway', file=sys.stderr)
        save_cutout(cutout, tmp / 'cutout.npz')
        save_fit(fit, tmp / 'fit.json')
        grid = {'profile': 'TNFW', 'Rs_arcsec': 0.2, 'r_trunc_arcsec': 0.8,
                'x_arcsec': [float(v) for v in keys['x']], 'y_arcsec': [float(v) for v in keys['y']],
                'alpha_Rs_arcsec': [float(v) for v in keys['alpha']]}
        (tmp / 'grid.yaml').write_text(yaml.safe_dump(grid), encoding='utf-8')
        out.mkdir(parents=True, exist_ok=True)
        create_task_bundle(tmp / 'cutout.npz', tmp / 'fit.json', tmp / 'grid.yaml', out / 'task.zip.part')
        os.replace(out / 'task.zip.part', out / 'task.zip')
        points = len(grid['x_arcsec']) * len(grid['y_arcsec']) * len(grid['alpha_Rs_arcsec'])
        (out / 'about.txt').write_text(
            f'cutout {" ".join(keys["cutout"])} shape {cutout.image.shape[0]}x{cutout.image.shape[1]} '
            f'usable {int(np.count_nonzero(cutout.mask))}; baseline chi2 {fit.chi2:.6f} converged {fit.converged} '
            f'nfev {fit.nfev}; grid points {points}\n', encoding='utf-8')


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    profile, cache = Path(argv[0]), Path(argv[1])
    text = profile.read_bytes()
    out = cache / f'{profile.stem}-{hashlib.sha256(text).hexdigest()[:10]}'
    if not (out / 'task.zip').is_file():
        make(read_profile(profile), out)
    print(out / 'task.zip')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
