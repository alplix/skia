"""Calibrate local two-Sersic fit errors on held-out seeded simulations."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np

from recovery import LENS_RANGES, SOURCE_RANGES
from skia.cutout import Cutout, load_cutout
from skia.fitter import (FitConfig, SmoothRenderer, TWO_SERSIC_PARAMETER_NAMES,
                         fit_cutout, two_sersic_parameter_vector)
from skia.simulator import Lens, Observation, Source, save_simulation, simulate


def _wilson(successes, total, z=1.959963984540054):
    if total==0:return [None,None]
    p=successes/total;denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(max(0.,center-radius)),float(min(1.,center+radius))]


def _usable_z(row, name):
    if row['parameters'] is None or row['parameter_errors'] is None:return None
    estimate=row['parameter_errors'].get(name)
    if estimate is None or not np.isfinite(estimate) or estimate<=0:return None
    z=(row['parameters'][name]-row['truth'][name])/estimate
    return float(z) if np.isfinite(z) else None


def _interval_count(values, limit):
    return {'count':int(len(values)),'inside':int(np.count_nonzero(np.abs(values)<=limit)),
            'fraction':None if not len(values) else float(np.mean(np.abs(values)<=limit)),
            'wilson_95':_wilson(int(np.count_nonzero(np.abs(values)<=limit)),len(values))}


def _group_summary(rows, indices):
    eligible=[row for row in rows if indices(row)]
    summary={'rows':len(eligible),'parameters':{}}
    calibration=[row for row in eligible if row['index']%2==0]
    evaluation=[row for row in eligible if row['index']%2==1]
    summary['calibration_rows']=len(calibration);summary['evaluation_rows']=len(evaluation)
    for name in TWO_SERSIC_PARAMETER_NAMES:
        train=np.asarray([z for row in calibration if (z:=_usable_z(row,name)) is not None])
        test=np.asarray([z for row in evaluation if (z:=_usable_z(row,name)) is not None])
        if not len(train) or not len(test):
            summary['parameters'][name]={'calibration_available':len(train),'evaluation_available':len(test),
                'raw_evaluation_68':_interval_count(test,1.),'raw_evaluation_95':_interval_count(test,1.96),
                'calibration_scale_68':None,'calibration_scale_95':None,
                'scaled_evaluation_68':_interval_count([],1.),'scaled_evaluation_95':_interval_count([],1.96)}
            continue
        scale68=float(np.quantile(np.abs(train),.6827,method='higher'))
        scale95=float(np.quantile(np.abs(train),.95,method='higher')/1.96)
        summary['parameters'][name]={'calibration_available':len(train),'evaluation_available':len(test),
            'raw_evaluation_68':_interval_count(test,1.),
            'raw_evaluation_95':_interval_count(test,1.96),
            'calibration_scale_68':scale68,'calibration_scale_95':scale95,
            'scaled_evaluation_68':_interval_count(test,scale68),
            'scaled_evaluation_95':_interval_count(test,1.96*scale95)}
    return summary


def summarize(rows):
    """Report raw and held-out-scaled coverage for all finite and clean fits."""
    finite=lambda row:row['parameters'] is not None
    interior=lambda row:finite(row) and row['converged'] and not row['boundary_parameters']
    available=sum(finite(row) for row in rows)
    return {'attempted':len(rows),'finite':available,'hard_failures':len(rows)-available,
        'unconverged_finite':sum(finite(row) and not row['converged'] for row in rows),
        'boundary_contacts':sum(bool(row['boundary_parameters']) for row in rows),
        'rank_deficient':sum(finite(row) and row['covariance_rank']<len(TWO_SERSIC_PARAMETER_NAMES) for row in rows),
        'split':'even index for calibration, odd index for held-out evaluation',
        'all_finite':_group_summary(rows,finite),
        'converged_interior':_group_summary(rows,interior)}


def cross_validated_coverage(rows, folds=5):
    """Evaluate per-parameter interval scales trained on the other folds."""
    if type(folds) is not int or folds<2:raise ValueError('folds must be an integer >=2')
    predicates={'all_finite':lambda row:row['parameters'] is not None,
        'converged_interior':lambda row:row['parameters'] is not None and row['converged'] and not row['boundary_parameters']}
    result={'folds':folds,'split':f'index modulo {folds}; each fold evaluated using the other folds for calibration',
            'groups':{}}
    for group_name,predicate in predicates.items():
        eligible=[row for row in rows if predicate(row)]
        group={'rows':len(eligible),'parameters':{}}
        for name in TWO_SERSIC_PARAMETER_NAMES:
            heldout=[];scale68=[];scale95=[]
            for fold in range(folds):
                train=[row for row in eligible if row['index']%folds!=fold]
                test=[row for row in eligible if row['index']%folds==fold]
                train_z=np.asarray([z for row in train if (z:=_usable_z(row,name)) is not None])
                test_z=np.asarray([z for row in test if (z:=_usable_z(row,name)) is not None])
                if not len(train_z) or not len(test_z):continue
                k68=float(np.quantile(np.abs(train_z),.6827,method='higher'))
                k95=float(np.quantile(np.abs(train_z),.95,method='higher')/1.96)
                heldout.extend((float(z),k68,k95) for z in test_z)
                scale68.append(k68);scale95.append(k95)
            z=np.asarray([row[0] for row in heldout]);k68=np.asarray([row[1] for row in heldout]);k95=np.asarray([row[2] for row in heldout])
            group['parameters'][name]={'heldout_count':len(z),
                'raw_68':_interval_count(z,1.),'raw_95':_interval_count(z,1.96),
                'cross_validated_68':_interval_count(z,k68),
                'cross_validated_95':_interval_count(z,1.96*k95),
                'fold_scale_68':{'median':None if not scale68 else float(np.median(scale68)),
                    'minimum':None if not scale68 else float(min(scale68)),'maximum':None if not scale68 else float(max(scale68))},
                'fold_scale_95':{'median':None if not scale95 else float(np.median(scale95)),
                    'minimum':None if not scale95 else float(min(scale95)),'maximum':None if not scale95 else float(max(scale95))}}
        result['groups'][group_name]=group
    return result


def _fit_one(index, rng, output, observation, config):
    lens=Lens(**{name:float(rng.uniform(*bounds)) for name,bounds in LENS_RANGES.items()})
    source=Source(**{name:float(rng.uniform(*bounds)) for name,bounds in SOURCE_RANGES.items()})
    source2=Source(amp=source.amp*float(rng.uniform(.15,.6)),
        R_sersic=float(rng.uniform(.04,.10)),n_sersic=float(rng.uniform(.8,3.)),
        e1=source.e1,e2=source.e2,center_x=source.center_x,center_y=source.center_y)
    sim=simulate(lens,source,observation,seed=520000+index)
    temporary=output/f'.sim-{index:04}.npz';save_simulation(sim,temporary)
    base=load_cutout(temporary);temporary.unlink()
    renderer=SmoothRenderer(base);vector=two_sersic_parameter_vector(lens,source,source2)
    truth_image=renderer.evaluate(vector)
    noise_map=np.sqrt(truth_image+observation.sky_rate*observation.exposure+observation.read_noise**2)
    noise_rng=np.random.Generator(np.random.PCG64(620000+index))
    cutout=Cutout(truth_image+noise_rng.normal(0,noise_map),noise_map,base.mask,base.psf_fine,
        base.pixel_scale,base.exposure,base.oversample,base.image_unit)
    truth_parameters=dict(zip(TWO_SERSIC_PARAMETER_NAMES,vector.tolist(),strict=True))
    truth_parameters['source.amp']=source.amp;truth_parameters['source.R_sersic']=source.R_sersic
    truth_parameters['source2.amp']=source2.amp;truth_parameters['source2.R_sersic']=source2.R_sersic
    fit=fit_cutout(cutout,config)
    return {'index':index,'noise_seed':620000+index,'truth':truth_parameters,
        'parameters':fit.parameters,'parameter_errors':fit.parameter_errors,
        'covariance_rank':fit.covariance_rank,'converged':fit.converged,
        'boundary_parameters':list(fit.boundary_parameters),'chi2':fit.chi2,'dof':fit.dof,
        'nfev':fit.nfev,'message':fit.message}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--max-nfev',type=int,default=200)
    parser.add_argument('--output',type=Path,default=Path('docs/b3-two-sersic-coverage'))
    parser.add_argument('--analyze-existing',action='store_true',help='write five-fold calibration evaluation from existing coverage.jsonl without refitting')
    args=parser.parse_args(argv)
    if args.count<2 or args.max_nfev<1:parser.error('count must be >=2 and max-nfev must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'coverage.jsonl'
    if args.analyze_existing:
        if not raw.exists():parser.error(f'no existing raw study at {raw}')
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        summary=cross_validated_coverage(rows)
        target=args.output/'crossval_summary.json'
        target.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
        print('B3 FIVE-FOLD COVERAGE SUMMARY: '+json.dumps({
            name:{'rows':value['rows'],'parameters':len(value['parameters'])}
            for name,value in summary['groups'].items()},sort_keys=True),flush=True)
        return 0
    if raw.exists():parser.error('coverage results already exist; choose a new output directory')
    observation=Observation(npix=40)
    config=FitConfig(starts=2,seed=731,max_nfev=args.max_nfev,source_components=2)
    protocol={'format':'skia-two-sersic-coverage-v1','count':args.count,
        'parameter_seed':20261103,'noise_seed_base':620000,'observation':asdict(observation),
        'fit_config':asdict(config),'lens_ranges':LENS_RANGES,'primary_source_ranges':SOURCE_RANGES,
        'secondary_source':{'amplitude_fraction_primary':[.15,.6],'R_sersic_arcsec':[.04,.10],
                            'n_sersic':[.8,3.0],'geometry':'shared with primary'},
        'noise':'independent Gaussian with known per-pixel standard deviation sqrt(expected source + sky + read_noise^2)',
        'nominal_intervals':'reported local Jacobian errors; calibration multipliers estimated on even indices and evaluated on odd indices',
        'calibration_quantiles':{'68_percent':.6827,'95_percent':.95,'method':'higher'},
        'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()}
    (args.output/'protocol.json').write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    rng=np.random.Generator(np.random.PCG64(20261103));rows=[]
    with raw.open('x') as handle:
        for index in range(args.count):
            try:
                row=_fit_one(index,rng,args.output,observation,config)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                row={'index':index,'noise_seed':620000+index,'truth':None,'parameters':None,
                    'parameter_errors':None,'covariance_rank':0,'converged':False,
                    'boundary_parameters':[],'error':str(error)}
            rows.append(row);handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
            if (index+1)%10==0 or index+1==args.count:
                print(f'two-Sersic coverage {index+1}/{args.count}',flush=True)
    summary=summarize(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('B3 TWO-SERSIC COVERAGE SUMMARY: '+json.dumps({k:summary[k] for k in
        ('attempted','finite','hard_failures','unconverged_finite','boundary_contacts','rank_deficient')}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
