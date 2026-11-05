"""Paired false-alarm and injection study for one/two-Sersic source models."""
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
from skia.fitter import FitConfig, SmoothRenderer, fit_cutout, two_sersic_parameter_vector
from skia.scanner import halo_deflection_field, scan
from skia.simulator import Lens, Observation, Source, save_simulation, simulate

THRESHOLDS=(2.,5.,10.)
GRID={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,
      'x_arcsec':[-.3,.3],'y_arcsec':[-.3,.3],
      'alpha_Rs_arcsec':[0.,.01,.03,.06]}
INJECTION_POSITION=(.3,.3)
INJECTION_STRENGTHS=(.01,.03,.06)


def _model_observation(cutout, config, grid):
    fit=fit_cutout(cutout,config)
    baseline={'format':'skia-fit-v1','parameters':fit.parameters,'optimizer':asdict(fit.config)}
    result=scan(cutout,baseline,grid)
    return {'fit':fit,'scan':result,'max_delta_chi2':float(result['best']['delta_chi2']),
            'best':{k:result['best'][k] for k in ('x_arcsec','y_arcsec','alpha_Rs_arcsec')},
            'converged_grid_points':sum(bool(row['converged']) for row in result['results']),
            'total_grid_points':len(result['results'])}


def _wilson(successes, total, z=1.959963984540054):
    if total==0:return [None,None]
    p=successes/total;denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(max(0.,center-radius)),float(min(1.,center+radius))]


def _summary(rows):
    result={}
    null_one=[row['models']['1']['null']['max_delta_chi2'] for row in rows]
    null_two=[row['models']['2']['null']['max_delta_chi2'] for row in rows]
    for components in (1,2):
        observations=[row['models'][str(components)] for row in rows]
        null=[model['null']['max_delta_chi2'] for model in observations]
        by_strength={}
        for strength in INJECTION_STRENGTHS:
            label=f'{strength:.2f}'
            injected=[model['injections'][label]['max_delta_chi2'] for model in observations]
            by_strength[label]={
                'fit_converged':sum(model['injections'][label]['fit_converged'] for model in observations),
                'grid_converged':sum(model['injections'][label]['grid_converged'] for model in observations),
                'grid_points':len(observations)*len(GRID['x_arcsec'])*len(GRID['y_arcsec'])*len(GRID['alpha_Rs_arcsec']),
                'thresholds':{str(threshold):{
                    'recoveries':int(sum(value>=threshold for value in injected)),
                    'recovery_fraction':float(np.mean(np.asarray(injected)>=threshold)),
                    'recovery_wilson_95':_wilson(sum(value>=threshold for value in injected),len(injected)),
                } for threshold in THRESHOLDS},
                'paired_at_threshold_2':{
                    'recovered_in_1_only':sum(row['models']['1']['injections'][label]['max_delta_chi2']>=2>
                        row['models']['2']['injections'][label]['max_delta_chi2'] for row in rows),
                    'recovered_in_2_only':sum(row['models']['1']['injections'][label]['max_delta_chi2']<2<=
                        row['models']['2']['injections'][label]['max_delta_chi2'] for row in rows),
                },
            }
        result[str(components)]={
            'count':len(observations),
            'null_fit_converged':sum(model['null']['fit_converged'] for model in observations),
            'null_grid_converged':sum(model['null']['grid_converged'] for model in observations),
            'grid_points_per_scan':len(GRID['x_arcsec'])*len(GRID['y_arcsec'])*len(GRID['alpha_Rs_arcsec']),
            'null_median_chi2_per_dof':float(np.median([model['null']['chi2_per_dof'] for model in observations])),
            'thresholds':{str(threshold):{
                'false_alarms':int(sum(value>=threshold for value in null)),
                'false_alarm_fraction':float(np.mean(np.asarray(null)>=threshold)),
                'false_alarm_wilson_95':_wilson(sum(value>=threshold for value in null),len(null)),
            } for threshold in THRESHOLDS},
            'injections_by_alpha_Rs_arcsec':by_strength,
            'paired_at_threshold_2':{
                'null_triggered_in_1_only':sum(a>=2 and b<2 for a,b in zip(null_one,null_two,strict=True)),
                'null_triggered_in_2_only':sum(a<2 and b>=2 for a,b in zip(null_one,null_two,strict=True)),
            },
        }
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=40)
    parser.add_argument('--max-nfev',type=int,default=60)
    parser.add_argument('--output',type=Path,default=Path('docs/b2-source-v3'))
    args=parser.parse_args(argv)
    if args.count<1 or args.max_nfev<1:parser.error('count and max-nfev must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'results.jsonl'
    if raw.exists():parser.error('results already exist; choose a new output directory')
    protocol={'format':'skia-b2-source-population-v2','count':args.count,
        'parameter_seed':20261102,'noise_seed_base':310000,'observation':asdict(Observation(npix=24,oversample=1)),
        'fit_config':{'starts':1,'seed':731,'max_nfev':args.max_nfev,'gamma_prior_sigma':.1},
        'lens_ranges':LENS_RANGES,'primary_source_ranges':SOURCE_RANGES,
        'secondary_source':{'amplitude_fraction_primary':[.15,.6],'R_sersic_arcsec':[.04,.10],
                            'n_sersic':[.8,3.0],'geometry':'shared with primary'},
        'scan_grid':GRID,'injection_position_arcsec':{'x':INJECTION_POSITION[0],'y':INJECTION_POSITION[1]},
        'injected_alpha_Rs_arcsec':INJECTION_STRENGTHS,
        'thresholds_delta_chi2':THRESHOLDS,
        'method':'Each two-component truth has one shared-geometry two-Sersic source. Paired one/two-Sersic fits use identical null and injected images. Independent Gaussian noise uses the per-pixel expected Poisson-plus-read-noise standard deviation computed from the richer noiseless source and simulator sky/read-noise settings. Injection adds the true rendered halo model delta. False alarms are raw max data delta-chi2 thresholds, not calibrated significance.',
        'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()}
    (args.output/'protocol.json').write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    rng=np.random.Generator(np.random.PCG64(20261102));rows=[]
    with raw.open('x') as handle:
      for i in range(args.count):
        lens=Lens(**{name:float(rng.uniform(*bounds)) for name,bounds in LENS_RANGES.items()})
        source=Source(**{name:float(rng.uniform(*bounds)) for name,bounds in SOURCE_RANGES.items()})
        source2=Source(amp=source.amp*float(rng.uniform(.15,.6)),
            R_sersic=float(rng.uniform(.04,.10)),n_sersic=float(rng.uniform(.8,3.)),
            e1=source.e1,e2=source.e2,center_x=source.center_x,center_y=source.center_y)
        observation=Observation(npix=24,oversample=1)
        simulation=simulate(lens,source,observation,seed=310000+i)
        base_path=args.output/f'.sim-{i:04}.npz';save_simulation(simulation,base_path)
        base=load_cutout(base_path);base_path.unlink()
        renderer=SmoothRenderer(base)
        true_vector=two_sersic_parameter_vector(lens,source,source2)
        truth=renderer.evaluate(true_vector)
        noise_rng=np.random.Generator(np.random.PCG64(410000+i))
        noise_map=np.sqrt(truth+observation.sky_rate*observation.exposure+observation.read_noise**2)
        null=Cutout(truth+noise_rng.normal(0,noise_map),noise_map,base.mask,base.psf_fine,
                    base.pixel_scale,base.exposure,base.oversample,base.image_unit)
        injection_cutouts={}
        x,y=INJECTION_POSITION
        for alpha in INJECTION_STRENGTHS:
            field=halo_deflection_field(base,x,y,alpha,GRID['Rs_arcsec'],GRID['r_trunc_arcsec'])
            injected_image=null.image+renderer.evaluate(true_vector,field)-truth
            injection_cutouts[f'{alpha:.2f}']=Cutout(injected_image,null.noise_map,null.mask,
                null.psf_fine,null.pixel_scale,null.exposure,null.oversample,null.image_unit)
        models={}
        for components in (1,2):
            config=FitConfig(starts=1,seed=731,max_nfev=args.max_nfev,gamma_prior_sigma=.1,
                             source_components=components)
            model_row={}
            for label,cutout in [('null',null)]:
                observed=_model_observation(cutout,config,GRID)
                model_row[label]={'max_delta_chi2':observed['max_delta_chi2'],'best':observed['best'],
                    'fit_converged':bool(observed['fit'].converged),
                    'chi2_per_dof':float(observed['fit'].chi2/observed['fit'].dof),
                    'grid_converged':int(observed['converged_grid_points']),
                    'grid_points':int(observed['total_grid_points'])}
            model_row['injections']={}
            for label,cutout in injection_cutouts.items():
                observed=_model_observation(cutout,config,GRID)
                model_row['injections'][label]={'max_delta_chi2':observed['max_delta_chi2'],
                    'best':observed['best'],'fit_converged':bool(observed['fit'].converged),
                    'chi2_per_dof':float(observed['fit'].chi2/observed['fit'].dof),
                    'grid_converged':int(observed['converged_grid_points']),
                    'grid_points':int(observed['total_grid_points'])}
            models[str(components)]=model_row
        row={'index':i,'parameter_seed':20261102,'noise_seed':410000+i,
             'truth':{'lens':asdict(lens),'source1':asdict(source),'source2':asdict(source2)},'models':models}
        rows.append(row);handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
        print(f'source population {i+1}/{args.count}',flush=True)
    summary=_summary(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('B2 SOURCE SUMMARY: '+json.dumps(summary,sort_keys=True),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
