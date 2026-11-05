"""Build the Skia beta task queue: one baseline fit per Q1 cutout, then task bundles that each scan a few grid columns.

Usage (repository root):  .venv/bin/python boinc/make_tasks.py --out work/boinc_queue --workers 4
Provisional beta grid (not a science decision): x,y = -1.0..1.0 step 0.2 arcsec, alpha_Rs in {0, 0.02, 0.05} arcsec, TNFW Rs 0.2, r_trunc 0.8.
Tasks are ordered round-robin over lenses (every lens gets its first chunk before any lens gets its second).
"""
import argparse, json, os, sys, tempfile
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import yaml

from skia.cutout import save_cutout
from skia.fitter import FitConfig, fit_cutout, save_fit
from skia.realdata import load_q1_fits
from skia.task_bundle import create_task_bundle

XS = [round(float(v), 1) for v in np.arange(-1.0, 1.0001, 0.2)]
YS = list(XS)
ALPHAS = [0.0, 0.02, 0.05]
COLS_PER_TASK = 2


def one_lens(args):
    path, out = args
    path = Path(path); out = Path(out)
    lens = path.stem.removesuffix('_VIS')
    done = out / f'{lens}.done'
    if done.exists():
        return lens, json.loads(done.read_text())
    try:
        cut, noise, meta = load_q1_fits(path)
        fit = fit_cutout(cut, FitConfig(gamma_prior_sigma=.1))
        if not fit.converged:
            return lens, {'skipped': 'baseline fit did not converge'}
        names = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_cutout(cut, tmp / 'c.npz'); save_fit(fit, tmp / 'f.json')
            for k in range(0, len(XS), COLS_PER_TASK):
                grid = {'profile': 'TNFW', 'Rs_arcsec': 0.2, 'r_trunc_arcsec': 0.8,
                        'x_arcsec': XS[k:k + COLS_PER_TASK], 'y_arcsec': YS, 'alpha_Rs_arcsec': ALPHAS}
                (tmp / 'g.yaml').write_text(yaml.safe_dump(grid))
                name = f'skia_{lens}_c{k // COLS_PER_TASK}.zip'
                create_task_bundle(tmp / 'c.npz', tmp / 'f.json', tmp / 'g.yaml', out / name)
                names.append(name)
        info = {'tasks': names, 'points_per_task': len(XS[:COLS_PER_TASK]) * len(YS) * len(ALPHAS)}
        done.write_text(json.dumps(info))
        return lens, info
    except Exception as e:  # keep going; record the reason
        return lens, {'skipped': f'{type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/q1_cutouts'); ap.add_argument('--out', default='work/boinc_queue')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.data).glob('*_VIS.fits'))
    if a.limit: files = files[:a.limit]
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(one_lens, [(f, out) for f in files]), 1):
            res.append(r); print(f'{i}/{len(files)} {r[0]} {"skip: " + r[1]["skipped"] if "skipped" in r[1] else "ok"}', flush=True)
    lens_tasks = [info['tasks'] for _, info in res if 'tasks' in info]
    order = [lt[j] for j in range(max(map(len, lens_tasks), default=0)) for lt in lens_tasks if j < len(lt)]
    (out / 'queue_index.json').write_text(json.dumps({'order': order, 'skipped': {l: i['skipped'] for l, i in res if 'skipped' in i}}, indent=1))
    print(f'{len(order)} tasks, {len(lens_tasks)} lenses ok')


if __name__ == '__main__':
    main()
