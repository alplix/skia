"""Measure one-Sersic lens-parameter error coverage on two-Sersic truths."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

_BENCH_PATH=Path(__file__).resolve().parent
sys.path[:]=[entry for entry in sys.path if Path(entry or '.').resolve()!=_BENCH_PATH]
import numpy as np
from skia.cutout import Cutout
from skia.fitter import FitConfig,SmoothRenderer,fit_cutout,two_sersic_parameter_vector
from skia.simulator import Lens,Observation,Source,gaussian_psf

_spec=importlib.util.spec_from_file_location('skia_recovery_for_coverage',_BENCH_PATH/'recovery.py')
_recovery=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_recovery)
LENS_RANGES=_recovery.LENS_RANGES
SOURCE_RANGES=_recovery.SOURCE_RANGES
PARAMETERS=tuple('lens.'+name for name in LENS_RANGES)


def _wilson(success,total,z=1.959963984540054):
    if not total:return [None,None]
    p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [float(max(0.,center-radius)),float(min(1.,center+radius))]


def summarize(rows,folds=5):
    """Summarize raw and fold-held-out coverage for the eight lens parameters."""
    if type(folds) is not int or folds<2:raise ValueError('folds must be an integer >=2')
    groups={'all_finite':lambda row:row['parameters'] is not None,
            'converged_interior':lambda row:row['parameters'] is not None and row['converged'] and not row['boundary_parameters']}
    output={'attempted':len(rows),'finite':sum(row['parameters'] is not None for row in rows),
        'hard_failures':sum(row['parameters'] is None for row in rows),
        'unconverged_finite':sum(row['parameters'] is not None and not row['converged'] for row in rows),
        'boundary_contacts':sum(bool(row['boundary_parameters']) for row in rows),'folds':folds,'groups':{}}
    for group_name,eligible_fn in groups.items():
        eligible=[row for row in rows if eligible_fn(row)]
        group={'rows':len(eligible),'parameters':{}}
        for name in PARAMETERS:
            z_by_index={}
            for row in eligible:
                error=row['parameter_errors'].get(name) if row.get('parameter_errors') else None
                if error is None or not np.isfinite(error) or error<=0:continue
                value=(row['parameters'][name]-row['truth'][name])/error
                if np.isfinite(value):z_by_index[row['index']]=float(value)
            zs=np.asarray(list(z_by_index.values()),dtype=float)
            heldout=[];scale68=[];scale95=[]
            for fold in range(folds):
                train=np.asarray([z for index,z in z_by_index.items() if index%folds!=fold])
                test=[(index,z) for index,z in z_by_index.items() if index%folds==fold]
                if not train.size or not test:continue
                k68=float(np.quantile(np.abs(train),.6827,method='higher'))
                k95=float(np.quantile(np.abs(train),.95,method='higher')/1.96)
                scale68.append(k68);scale95.append(k95)
                heldout.extend((z,k68,k95) for _,z in test)
            def interval(values,limit):
                count=int(np.count_nonzero(np.abs(values)<=limit))
                return {'available':int(len(values)),'inside':count,
                    'fraction':None if not len(values) else float(count/len(values)),
                    'wilson_95':_wilson(count,len(values))}
            h=np.asarray([r[0] for r in heldout]);k68=np.asarray([r[1] for r in heldout]);k95=np.asarray([r[2] for r in heldout])
            group['parameters'][name]={'raw_68':interval(zs,1.),'raw_95':interval(zs,1.96),
                'cross_validated_68':interval(h,np.asarray(k68)) if len(h) else interval(h,1.),
                'cross_validated_95':interval(h,np.asarray(k95)*1.96) if len(h) else interval(h,1.96),
                'median_fold_scale_68':None if not scale68 else float(np.median(scale68)),
                'median_fold_scale_95':None if not scale95 else float(np.median(scale95)),
                'fold_scale_68_range':None if not scale68 else [float(min(scale68)),float(max(scale68))],
                'fold_scale_95_range':None if not scale95 else [float(min(scale95)),float(max(scale95))]}
        output['groups'][group_name]=group
    return output


def compare_matched_reference(rows,reference_rows):
    """Compare mismatch coverage to matched-model fits for identical B3 draws."""
    reference={row['index']:row for row in reference_rows};result={}
    for name in PARAMETERS:
        paired=[]
        for row in rows:
            other=reference.get(row['index'])
            if other is None or row['parameters'] is None or other['parameters'] is None:continue
            if not np.isclose(row['truth'][name],other['truth'][name],rtol=0,atol=1e-12):
                raise ValueError(f'Paired truth mismatch for {name} at index {row["index"]}')
            if row.get('source_truth'):
                for source_name,source_value in row['source_truth']['source1'].items():
                    reference_name='source.'+source_name
                    if not np.isclose(source_value,other['truth'][reference_name],rtol=0,atol=1e-12):
                        raise ValueError(f'Paired truth mismatch for {reference_name} at index {row["index"]}')
            error=row['parameter_errors'].get(name) if row.get('parameter_errors') else None
            other_error=other['parameter_errors'].get(name) if other.get('parameter_errors') else None
            if error is None or other_error is None or error<=0 or other_error<=0:continue
            z=(row['parameters'][name]-row['truth'][name])/error
            other_z=(other['parameters'][name]-other['truth'][name])/other_error
            if np.isfinite([z,other_z]).all():paired.append((abs(z)<=1,abs(z)<=1.96,abs(other_z)<=1,abs(other_z)<=1.96))
        values=np.asarray(paired,dtype=bool)
        result[name]={'paired_count':int(len(values)),
            'mismatch_68_inside':int(values[:,0].sum()) if len(values) else 0,
            'matched_68_inside':int(values[:,2].sum()) if len(values) else 0,
            'mismatch_only_68':int(np.count_nonzero(values[:,0]&~values[:,2])) if len(values) else 0,
            'matched_only_68':int(np.count_nonzero(values[:,2]&~values[:,0])) if len(values) else 0,
            'mismatch_95_inside':int(values[:,1].sum()) if len(values) else 0,
            'matched_95_inside':int(values[:,3].sum()) if len(values) else 0,
            'mismatch_only_95':int(np.count_nonzero(values[:,1]&~values[:,3])) if len(values) else 0,
            'matched_only_95':int(np.count_nonzero(values[:,3]&~values[:,1])) if len(values) else 0}
    return result


def _make_row(index,parameter_rng,secondary_rng,observation,config):
    lens=Lens(**{name:float(parameter_rng.uniform(*bounds)) for name,bounds in LENS_RANGES.items()})
    source=Source(**{name:float(parameter_rng.uniform(*bounds)) for name,bounds in SOURCE_RANGES.items()})
    secondary=Source(amp=source.amp*float(secondary_rng.uniform(.15,.6)),
        R_sersic=float(secondary_rng.uniform(.04,.10)),n_sersic=float(secondary_rng.uniform(.8,3.)),
        e1=source.e1,e2=source.e2,center_x=source.center_x,center_y=source.center_y)
    truth_lens={'lens.'+name:float(value) for name,value in asdict(lens).items()}
    psf=gaussian_psf(observation.pixel_scale/observation.oversample,observation.psf_fwhm)
    blank=Cutout(np.zeros((observation.npix,observation.npix)),np.ones((observation.npix,observation.npix)),
        np.ones((observation.npix,observation.npix),dtype=bool),psf,observation.pixel_scale,
        observation.exposure,observation.oversample)
    renderer=SmoothRenderer(blank)
    truth_image=renderer.evaluate(two_sersic_parameter_vector(lens,source,secondary))
    sky=observation.sky_rate*observation.exposure
    expected=truth_image+sky
    noise=np.sqrt(expected+observation.read_noise**2)
    noise_seed=100000+index;noise_rng=np.random.Generator(np.random.PCG64(noise_seed))
    image=noise_rng.poisson(expected).astype(float)+noise_rng.normal(0,observation.read_noise,truth_image.shape)-sky
    observed=Cutout(image,noise,blank.mask,blank.psf_fine,
        blank.pixel_scale,blank.exposure,blank.oversample,blank.image_unit)
    fit=fit_cutout(observed,config)
    return {'index':index,'parameter_seed':20261031,'noise_seed':noise_seed,
        'secondary_seed':20261104,'truth':truth_lens,'parameters':fit.parameters,
        'parameter_errors':fit.parameter_errors,'converged':bool(fit.converged),
        'boundary_parameters':list(fit.boundary_parameters),'covariance_rank':fit.covariance_rank,
        'chi2_per_dof':float(fit.chi2/fit.dof),'nfev':fit.nfev,
        'source_truth':{'source1':asdict(source),'source2':asdict(secondary)}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('--max-nfev',type=int,default=200)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--reference-coverage',type=Path,default=Path('docs/b3-coverage/coverage.jsonl'))
    parser.add_argument('--analyze-existing',action='store_true',help='summarize saved coverage.jsonl without refitting')
    parser.add_argument('--output',type=Path,default=Path('docs/b3-source-mismatch-coverage'))
    args=parser.parse_args(argv)
    if args.count<2 or args.max_nfev<1 or args.folds<2:parser.error('count/folds must be >=2 and max-nfev positive')
    args.output.mkdir(parents=True,exist_ok=True);raw=args.output/'coverage.jsonl'
    if args.analyze_existing:
        if not raw.exists():parser.error(f'no existing raw study at {raw}')
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        summary=summarize(rows,args.folds)
        if args.reference_coverage.exists():
            reference=[json.loads(line) for line in args.reference_coverage.read_text().splitlines()]
            summary['paired_model_matched_comparison']=compare_matched_reference(rows,reference)
        (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
        print(f"B3 SOURCE-MISMATCH REANALYSIS: attempted={summary['attempted']} finite={summary['finite']}",flush=True)
        return 0
    if raw.exists():parser.error('coverage results already exist; choose a new output directory')
    observation=Observation(npix=40)
    config=FitConfig(starts=2,seed=731,max_nfev=args.max_nfev,source_components=1)
    protocol={'format':'skia-one-sersic-source-mismatch-coverage-v2','count':args.count,
        'parameter_seed':20261031,'noise_seed_base':100000,'secondary_seed':20261104,
        'observation':asdict(observation),
        'fit_config':asdict(config),'lens_ranges':LENS_RANGES,'primary_source_ranges':SOURCE_RANGES,
        'secondary_source':{'amplitude_fraction_primary':[.15,.6],'R_sersic_arcsec':[.04,.10],
            'n_sersic':[.8,3.0],'geometry':'shared with primary'},
        'noise':'Poisson counts plus Gaussian read noise from two-Sersic expected counts; noise map sqrt(expected + read_noise^2)',
        'coverage':'one-Sersic local Jacobian errors cover lens truth; raw and five-fold scales reported',
        'fit_population':'paired with first count draws of B3 one-Sersic matched-model coverage by lens/source parameters, noise seeds and Poisson-plus-read-noise generation; only the secondary source differs; no injected halo',
        'matched_reference':args.reference_coverage.as_posix(),
        'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        'git_worktree_dirty':bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip())}
    (args.output/'protocol.json').write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    parameter_rng=np.random.Generator(np.random.PCG64(20261031))
    secondary_rng=np.random.Generator(np.random.PCG64(20261104));rows=[]
    with raw.open('x') as handle:
        for index in range(args.count):
            try:row=_make_row(index,parameter_rng,secondary_rng,observation,config)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                row={'index':index,'parameter_seed':20261031,'noise_seed':100000+index,'truth':None,'parameters':None,
                    'parameter_errors':None,'converged':False,'boundary_parameters':[],'error':str(error)}
            rows.append(row);handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
            print(f'source mismatch coverage {index+1}/{args.count}',flush=True)
    summary=summarize(rows,args.folds)
    if args.reference_coverage.exists():
        reference=[json.loads(line) for line in args.reference_coverage.read_text().splitlines()]
        summary['paired_model_matched_comparison']=compare_matched_reference(rows,reference)
    (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('B3 SOURCE-MISMATCH COVERAGE SUMMARY: '+json.dumps({
        'attempted':summary['attempted'],'finite':summary['finite'],'unconverged':summary['unconverged_finite'],
        'boundary_contacts':summary['boundary_contacts'],'groups':{
            name:{parameter:{'raw68':stats['raw_68']['fraction'],'raw95':stats['raw_95']['fraction'],
                             'cv68':stats['cross_validated_68']['fraction'],'cv95':stats['cross_validated_95']['fraction']}
                  for parameter,stats in values['parameters'].items()}
            for name,values in summary['groups'].items()}},sort_keys=True),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
