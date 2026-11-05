"""Paired free-slope vs Normal(gamma=2, sigma=.1) recovery study."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from recovery import LENS_RANGES, SOURCE_RANGES, _json
from skia.fitter import FitConfig, parameter_vector, fit_cutout
from skia.simulator import Lens, Source, Observation, gaussian_psf, simulate
from skia.cutout import Cutout


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--output',type=Path,default=Path('docs/review-fix1'))
    args=parser.parse_args(argv)
    if args.count<1: parser.error('count must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'paired.jsonl'
    if raw.exists(): parser.error('refusing to overwrite prior outcomes')
    rng=np.random.Generator(np.random.PCG64(20261031)); rows=[]
    obs=Observation(npix=40)
    with raw.open('x') as handle:
      for i in range(args.count):
        lens=Lens(**{k:float(rng.uniform(*v)) for k,v in LENS_RANGES.items()})
        source=Source(**{k:float(rng.uniform(*v)) for k,v in SOURCE_RANGES.items()})
        sim=simulate(lens,source,obs,seed=100000+i)
        cut=Cutout(sim.image,sim.noise_map,sim.mask,gaussian_psf(.05,.18),.1,565,2)
        row={'i':i,'truth':{'lens':lens.__dict__,'source':source.__dict__},'fits':{}}
        for label,sigma in [('free',None),('gamma_prior_sigma_0.1',.1)]:
          try:
            result=fit_cutout(cut,FitConfig(gamma_prior_sigma=sigma))
            row['fits'][label]={'parameters':result.parameters,'converged':result.converged,
              'boundary_parameters':result.boundary_parameters,'chi2':result.chi2,'dof':result.dof,
              'prior_penalty':result.prior_penalty,'elapsed_seconds':result.elapsed_seconds,
              'attempts':result.attempts}
          except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            row['fits'][label]={'error':str(exc),'converged':False,'parameters':None}
        rows.append(row);handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
        if (i+1)%20==0 or i+1==args.count: print(f'paired {i+1}/{args.count}',flush=True)
    summary={}
    for method in ('free','gamma_prior_sigma_0.1'):
      finite=[r for r in rows if r['fits'][method]['parameters'] is not None]
      out={}
      for key in ('lens.theta_E','lens.gamma','lens.gamma1','lens.gamma2'):
        field='lens';param=key.split('.')[1]
        errors=np.array([r['fits'][method]['parameters'][key]-r['truth'][field][param] for r in finite])
        out[key]={'bias':float(errors.mean()),'scatter':float(errors.std()),'rmse':float(np.mean(errors**2)**.5)}
      magerr=np.array([np.hypot(r['fits'][method]['parameters']['lens.gamma1'],r['fits'][method]['parameters']['lens.gamma2'])-
                         np.hypot(r['truth']['lens']['gamma1'],r['truth']['lens']['gamma2']) for r in finite])
      thetaerr=np.array([r['fits'][method]['parameters']['lens.theta_E']-r['truth']['lens']['theta_E'] for r in finite])
      gammaerr=np.array([r['fits'][method]['parameters']['lens.gamma']-r['truth']['lens']['gamma'] for r in finite])
      out['shear_magnitude']={'bias':float(magerr.mean()),'scatter':float(magerr.std()),'rmse':float(np.mean(magerr**2)**.5)}
      out.update({'finite':len(finite),'converged':sum(r['fits'][method]['converged'] for r in rows),
                  'boundary':sum(bool(r['fits'][method].get('boundary_parameters')) for r in rows),
                  'theta_E_gross':int(np.sum(np.abs(thetaerr)>.03)),
                  'gamma_gross':int(np.sum(np.abs(gammaerr)>.15)),
                  'median_chi2_dof':float(np.median([r['fits'][method]['chi2']/r['fits'][method]['dof'] for r in finite]))})
      summary[method]=out
    _json(args.output/'summary.json',summary)
    report=['# Finding 1: constrained-slope baseline','',f'{len(rows)} paired simulated lenses; same image/noise realization fit by both methods.',
      'The constrained method uses a predeclared Gaussian prior gamma ~ Normal(2, 0.1), adds its penalty to the optimizer objective, and reports data chi-square separately. The free-slope run remains unchanged. No input or draw was selected by outcome.','',
      '| Metric | Free gamma | Prior sigma=0.1 |', '|---|---:|---:|']
    for metric in ('converged','boundary','theta_E_gross','gamma_gross','median_chi2_dof'):
      report.append(f"| {metric} | {summary['free'][metric]} | {summary['gamma_prior_sigma_0.1'][metric]} |")
    for name in ('lens.theta_E','lens.gamma','shear_magnitude','lens.gamma1','lens.gamma2'):
      report += ['',f'## {name}','','| Statistic | Free | Prior sigma=0.1 |','|---|---:|---:|']
      for stat in ('bias','scatter','rmse'):
       report.append(f"| {stat} | {summary['free'][name][stat]:.7g} | {summary['gamma_prior_sigma_0.1'][name][stat]:.7g} |")
    report += ['',f'Full paired fits and attempt outcomes are in `{raw.name}`; summary values are in `summary.json`.','',
      'A slope prior changes the statistical model and can bias gamma; the prior result does not replace the free fit. It is a proposed stable scan baseline only. Synthetic images use matched profiles and ideal known noise; this is not real-data calibration or a scientific cross-check.','']
    (args.output/'REPORT.md').write_text('\n'.join(report))
    print('SUMMARY: '+json.dumps(summary,sort_keys=True),flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
