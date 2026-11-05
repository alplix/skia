"""Paired null/injection comparison of single and two-Sersic source fits."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from skia.cutout import load_cutout
from skia.fitter import FitConfig, SmoothRenderer, fit_cutout, parameter_vector, two_sersic_parameter_vector
from skia.scanner import halo_deflection_field, scan
from skia.simulator import Lens, Observation, Source, gaussian_psf, save_simulation, simulate
from skia.cutout import Cutout

THRESHOLDS=(2.,5.,10.)


def _baseline(result):
    return {'format':'skia-fit-v1','parameters':result.parameters,'optimizer':asdict(result.config)}


def _scan(cutout,baseline,grid):
    out=scan(cutout,baseline,grid)
    return {'max_delta_chi2':out['best']['delta_chi2'],'converged_points':sum(r['converged'] for r in out['results']),
            'grid_points':len(out['results'])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=12)
    parser.add_argument('--max-nfev',type=int,default=40)
    parser.add_argument('--output',default='docs/b2-source-results.json')
    args=parser.parse_args(argv)
    if args.count<1 or args.max_nfev<1:parser.error('count and max-nfev must be positive')
    config_base={'starts':1,'seed':731,'max_nfev':args.max_nfev,'gamma_prior_sigma':.1}
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,'x_arcsec':[-.3,.3],
          'y_arcsec':[-.3,.3],'alpha_Rs_arcsec':[0.,.03]}
    work=Path('work/b2-source');work.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(args.count):
        obs=Observation(npix=24,oversample=1)
        lens=Lens(theta_E=.9+.02*(i%6),gamma=2.0,e1=.10,e2=.03)
        source=Source(amp=10.,R_sersic=.18,n_sersic=1.2,e1=.05,e2=-.02,center_x=.05,center_y=.03)
        source2=Source(amp=6.,R_sersic=.05,n_sersic=2.0,e1=source.e1,e2=source.e2,
                       center_x=source.center_x,center_y=source.center_y)
        sim=simulate(lens,source,obs,seed=73000+i)
        path=work/f'base-{i:03}.npz';save_simulation(sim,path);basecut=load_cutout(path)
        true_vector=two_sersic_parameter_vector(lens,source,source2)
        renderer=SmoothRenderer(basecut)
        true_smooth=renderer.evaluate(true_vector)
        rng=np.random.Generator(np.random.PCG64(99000+i))
        noise=rng.normal(0,basecut.noise_map)
        null=Cutout(true_smooth+noise,basecut.noise_map,basecut.mask,basecut.psf_fine,
                    basecut.pixel_scale,basecut.exposure,basecut.oversample,'electron')
        field=halo_deflection_field(basecut,.3,.3,.03,.2,.8)
        with_halo=renderer.evaluate(true_vector,field)
        injected=Cutout(null.image+with_halo-true_smooth,null.noise_map,null.mask,null.psf_fine,
                        null.pixel_scale,null.exposure,null.oversample,'electron')
        item={'seed':73000+i,'models':{}}
        for components in (1,2):
            cfg=FitConfig(**config_base,source_components=components)
            null_fit=fit_cutout(null,cfg);payload=_baseline(null_fit)
            inj_fit=fit_cutout(injected,cfg);inj_payload=_baseline(inj_fit)
            item['models'][str(components)]={
                'null':_scan(null,payload,grid),'injection':_scan(injected,inj_payload,grid),
                'null_fit_converged':null_fit.converged,'injection_fit_converged':inj_fit.converged,
                'null_chi2_per_dof':null_fit.chi2/null_fit.dof,'injection_chi2_per_dof':inj_fit.chi2/inj_fit.dof}
        rows.append(item)
        print(f'lens {i+1}/{args.count}',flush=True)
    summary={}
    for components in (1,2):
        models=[r['models'][str(components)] for r in rows]
        summary[str(components)]={
          'n':len(models),
          'null_median_chi2_per_dof':float(np.median([m['null_chi2_per_dof'] for m in models])),
          'null_false_positive_counts':{str(t):sum(m['null']['max_delta_chi2']>=t for m in models) for t in THRESHOLDS},
          'injection_recovery_counts':{str(t):sum(m['injection']['max_delta_chi2']>=t for m in models) for t in THRESHOLDS},
          'null_scan_points_converged':sum(m['null']['converged_points'] for m in models),
          'injection_scan_points_converged':sum(m['injection']['converged_points'] for m in models)}
    payload={'format':'skia-b2-source-complexity-v1','source_truth':'two Sersic components, shared ellipticity/center',
      'fit_comparison':['one Sersic','two Sersic components sharing ellipticity and center'],
      'grid':grid,'injected_alpha_Rs_arcsec':.03,'thresholds_delta_chi2':THRESHOLDS,
      'fit_config':{**config_base,'source_components':'1 or 2'},'summary':summary,'rows':rows,
      'limits':['Small controlled simulated population; not Q1 false-positive calibration.',
                'Noise is seeded independent Gaussian using the simulator single-source noise map.',
                'Shared-center/shared-ellipticity two-Sersic family cannot represent arbitrary source structure.',
                'alpha_Rs is angular lensing strength, not subhalo mass.']}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    print(json.dumps(summary,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
