"""Exploratory angular-strength injection study on simulations and Q1 cutouts."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from skia.fitter import FitConfig, SmoothRenderer, fit_cutout
from skia.realdata import load_q1_fits
from skia.scanner import halo_deflection_field, scan
from skia.simulator import Observation, simulate, save_simulation
from skia.cutout import load_cutout
from skia.fitter import parameter_vector
from skia.simulator import Lens, Source

THRESHOLDS = (2., 5., 10.)
INJECTIONS = (.02, .05)
GRID_STRENGTHS = [0., .02, .05]
POSITIONS = [-.3, .3]


def _fit_payload(result):
    from dataclasses import asdict
    return {'format': 'skia-fit-v1', 'parameters': result.parameters,
            'optimizer': asdict(result.config)}


def _measure(cutout, payload, config, grid):
    outcome = scan(cutout, payload, grid)
    return {'max_delta_chi2': float(outcome['best']['delta_chi2']),
            'best': {k: outcome['best'][k] for k in ('x_arcsec','y_arcsec','alpha_Rs_arcsec')},
            'converged_points': sum(bool(r['converged']) for r in outcome['results']),
            'grid_points': len(outcome['results'])}


def _inject(cutout, vector, x, y, alpha):
    renderer = SmoothRenderer(cutout)
    smooth = renderer.evaluate(vector)
    field = halo_deflection_field(cutout, x, y, alpha, .2, .8)
    perturbed = renderer.evaluate(vector, field)
    # On real images this model difference preserves all observed residuals.
    return replace(cutout, image=cutout.image + perturbed - smooth)


def _integrated_snr(cutout):
    """Positive integrated cutout flux / independent-pixel noise (dimensionless)."""
    usable=cutout.mask
    signal=np.maximum(cutout.image[usable],0.)
    return float(signal.sum()/np.sqrt(np.square(cutout.noise_map[usable]).sum()))


def _row(kind, label, cutout, baseline, config, grid, x, y):
    null = _measure(cutout, baseline, config, grid)
    result = {'kind':kind,'label':label,'integrated_snr':_integrated_snr(cutout),
              'injected_alpha_Rs_arcsec':0.,'x_arcsec':x,'y_arcsec':y,'scan':null}
    vector = parameter_vector(Lens(**{k.removeprefix('lens.'):v for k,v in baseline['parameters'].items() if k.startswith('lens.') }),
                              Source(**{k.removeprefix('source.'):v for k,v in baseline['parameters'].items() if k.startswith('source.') }))
    for alpha in INJECTIONS:
        injected = _inject(cutout, vector, x, y, alpha)
        rec = _measure(injected, baseline, config, grid)
        result.setdefault('injections',[]).append({'injected_alpha_Rs_arcsec':alpha,'scan':rec})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/p2-injection-results.json')
    parser.add_argument('--simulation-count',type=int,default=20)
    parser.add_argument('--real-count',type=int,default=24)
    parser.add_argument('--max-nfev',type=int,default=80)
    args=parser.parse_args(argv)
    if min(args.simulation_count,args.real_count,args.max_nfev)<1: parser.error('counts and max-nfev must be positive')
    config=FitConfig(starts=1,seed=731,max_nfev=args.max_nfev,gamma_prior_sigma=.1)
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,
          'x_arcsec':POSITIONS,'y_arcsec':POSITIONS,'alpha_Rs_arcsec':GRID_STRENGTHS}
    rows=[]
    work=Path('work/p2-injections');work.mkdir(parents=True,exist_ok=True)
    for i in range(args.simulation_count):
        path=work/f'sim-{i:03}.npz'
        save_simulation(simulate(observation=Observation(npix=30),seed=42000+i),path)
        cutout=load_cutout(path)
        basefit=fit_cutout(cutout,config);baseline=_fit_payload(basefit)
        rows.append(_row('simulation',f'sim-{i:03}',cutout,baseline,config,grid,.3,.3))
        print(f'simulation {i+1}/{args.simulation_count}',flush=True)
    files=sorted(Path('data/q1_cutouts').glob('*_VIS.fits'))
    if args.real_count>len(files): parser.error(f'real-count exceeds available {len(files)} cutouts')
    # Deterministic evenly-spaced sample; default includes all 24 supplied images.
    indices=np.linspace(0,len(files)-1,args.real_count,dtype=int)
    for j,index in enumerate(indices):
        path=files[int(index)];cutout,_,_=load_q1_fits(path)
        basefit=fit_cutout(cutout,config);baseline=_fit_payload(basefit)
        rows.append(_row('real_q1',path.name,cutout,baseline,config,grid,.3,.3))
        print(f'Q1 cutout {j+1}/{len(indices)}: {path.name}',flush=True)
    summary={}
    groups=sorted({r['kind'] for r in rows})
    for kind in groups:
        subset=[r for r in rows if r['kind']==kind]
        null=np.array([r['scan']['max_delta_chi2'] for r in subset])
        summary[kind]={'n_null':len(null),'null_median_max_delta_chi2':float(np.median(null)),
                       'null_q95_max_delta_chi2':float(np.quantile(null,.95)),
                       'false_positive_counts':{str(t):int((null>=t).sum()) for t in THRESHOLDS},
                       'injections':{}}
        for alpha in INJECTIONS:
            values=np.array([next(x['scan']['max_delta_chi2'] for x in r['injections'] if x['injected_alpha_Rs_arcsec']==alpha) for r in subset])
            summary[kind]['injections'][str(alpha)]={'n':len(values),'median_max_delta_chi2':float(np.median(values)),
              'recovery_counts':{str(t):int((values>=t).sum()) for t in THRESHOLDS}}
    payload={'format':'skia-p2-injections-v1','statistic':'maximum data delta_chi2 over 4 positions x 3 angular strengths',
             'thresholds_delta_chi2':THRESHOLDS,'injection_strengths_alpha_Rs_arcsec':INJECTIONS,
             'grid':grid,'config':{'starts':config.starts,'seed':config.seed,'max_nfev':config.max_nfev,'gamma_prior_sigma':config.gamma_prior_sigma},
             'physical_mass_unit':'not-calibrated','summary':summary,'rows':rows,
             'limits':['Injected alpha_Rs is an angular lensing-strength proxy, not a mass.',
                       'Real-image injections preserve residuals by adding the fitted-model with-halo minus no-halo image difference.',
                       'Thresholds are provisional raw delta-chi-square cuts; this study does not establish calibrated significance.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,sort_keys=True),flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
