"""Seeded nominal-interval coverage study for local linearized fit errors."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS','cpu')
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name,'1')

import numpy as np

from recovery import LENS_RANGES, SOURCE_RANGES
from skia.cutout import Cutout
from skia.fitter import FitConfig, PARAMETER_NAMES, fit_cutout, parameter_vector, physical_parameters
from skia.simulator import Lens, Observation, Source, gaussian_psf, simulate


def _one(lens,source,observation,index):
    truth=physical_parameters(parameter_vector(lens,source))
    row={'index':index,'truth':truth}
    try:
        sim=simulate(lens,source,observation,seed=100000+index)
        cut=Cutout(sim.image,sim.noise_map,sim.mask,
                   gaussian_psf(observation.pixel_scale/observation.oversample,observation.psf_fwhm),
                   observation.pixel_scale,observation.exposure,observation.oversample)
        fit=fit_cutout(cut,FitConfig())
        row.update(parameters=fit.parameters,parameter_errors=fit.parameter_errors,
                   covariance_rank=fit.covariance_rank,converged=fit.converged,
                   boundary_parameters=list(fit.boundary_parameters),chi2=fit.chi2,dof=fit.dof)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        row.update(parameters=None,parameter_errors=None,covariance_rank=0,converged=False,
                   boundary_parameters=[],error=str(error))
    return row


def _coverage(rows):
    result={}
    for subset_name,subset in [('all_finite',[r for r in rows if r['parameters'] is not None]),
                               ('converged_interior',[r for r in rows if r['parameters'] is not None and r['converged'] and not r['boundary_parameters']])]:
        columns={}
        for name in PARAMETER_NAMES:
            valid=[r for r in subset if r['parameter_errors'] and r['parameter_errors'].get(name) is not None
                   and np.isfinite(r['parameter_errors'][name]) and r['parameter_errors'][name]>0]
            if not valid:
                columns[name]={'available':0,'nominal_68_count':0,'nominal_68_fraction':None,
                               'nominal_95_count':0,'nominal_95_fraction':None}
                continue
            z=np.array([(r['parameters'][name]-r['truth'][name])/r['parameter_errors'][name] for r in valid])
            columns[name]={'available':len(valid),'nominal_68_count':int((np.abs(z)<=1).sum()),
                           'nominal_68_fraction':float((np.abs(z)<=1).mean()),
                           'nominal_95_count':int((np.abs(z)<=1.96).sum()),
                           'nominal_95_fraction':float((np.abs(z)<=1.96).mean())}
        result[subset_name]={'count':len(subset),'parameters':columns}
    result.update(attempted=len(rows),finite=sum(r['parameters'] is not None for r in rows),
                  failed=sum(r['parameters'] is None for r in rows),
                  unconverged=sum(r['parameters'] is not None and not r['converged'] for r in rows),
                  bound_contacts=sum(bool(r['boundary_parameters']) for r in rows),
                  rank_deficient=sum(r['parameters'] is not None and r['covariance_rank']<len(PARAMETER_NAMES) for r in rows))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--output',type=Path,default=Path('docs/b3-coverage'))
    args=parser.parse_args(argv)
    if args.count<1:parser.error('count must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'coverage.jsonl'
    if raw.exists():parser.error('coverage results already exist; select a new output directory')
    root=Path(__file__).resolve().parents[2]
    protocol={'format':'skia-linearized-coverage-v1','count':args.count,'parameter_seed':20261031,
              'noise_seed_base':100000,'observation':asdict(Observation(npix=40)),
              'fit_config':asdict(FitConfig()),'lens_ranges':LENS_RANGES,'source_ranges':SOURCE_RANGES,
              'git_revision':subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip(),
              'nominal_intervals':'1 sigma and 1.96 sigma from local Jacobian covariance'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    rng=np.random.Generator(np.random.PCG64(20261031));rows=[]
    with raw.open('x') as handle:
        for i in range(args.count):
            lens=Lens(**{k:float(rng.uniform(*v)) for k,v in LENS_RANGES.items()})
            source=Source(**{k:float(rng.uniform(*v)) for k,v in SOURCE_RANGES.items()})
            row=_one(lens,source,Observation(npix=40),i);rows.append(row)
            handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
            if (i+1)%20==0 or i+1==args.count: print(f'coverage {i+1}/{args.count}',flush=True)
    summary=_coverage(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print(f"COVERAGE SUMMARY: {summary['attempted']} attempted, {summary['finite']} finite, {summary['failed']} hard failures, {summary['unconverged']} unconverged, {summary['rank_deficient']} rank deficient")
    return 0

if __name__=='__main__':raise SystemExit(main())
