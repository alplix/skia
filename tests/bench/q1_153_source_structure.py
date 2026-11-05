"""Compare one/two-Sersic residual fit gains with fixed-grid TNFW gains on Q1."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
import json
from pathlib import Path

import numpy as np

from skia.fitter import FitConfig, fit_cutout, two_sersic_parameter_vector
from skia.realdata import load_q1_fits
from skia.scanner import vector_from_fit
from skia.simulator import Lens, Source

CONFIG=FitConfig(starts=1,seed=731,max_nfev=80,gamma_prior_sigma=.1,source_components=2)


def _compare(one_chi2: float, two_chi2: float, t_nfw_delta_chi2: float) -> dict:
    """Compare raw data-chi-square gains and fraction of baseline residual chi2."""
    values=(one_chi2,two_chi2,t_nfw_delta_chi2)
    if not np.isfinite(values).all() or one_chi2<=0:
        raise ValueError('comparison chi-square values must be finite and baseline positive')
    source_delta=one_chi2-two_chi2
    return {'source_complexity_delta_chi2':float(source_delta),
            'tnfw_delta_chi2':float(t_nfw_delta_chi2),
            'source_complexity_percent_of_one_sersic_chi2':float(100*source_delta/one_chi2),
            'tnfw_percent_of_one_sersic_chi2':float(100*t_nfw_delta_chi2/one_chi2),
            'source_complexity_delta_minus_tnfw_delta_chi2':float(source_delta-t_nfw_delta_chi2),
            'source_complexity_gain_larger':bool(source_delta>t_nfw_delta_chi2)}


def _summarize(rows: list[dict], output: Path) -> dict:
    valid=[row for row in rows if 'comparison' in row]
    if not valid:raise ValueError('no valid source-complexity comparisons')
    summary={'format':'skia-q1-source-structure-v1','n_cutouts':len(rows),
      'n_comparisons':len(valid),'hard_errors':sum('hard_error' in row for row in rows),
      'two_sersic_converged':sum(row['two_sersic']['converged'] for row in valid),
      'two_sersic_bound_contact_count':sum(bool(row['two_sersic']['boundary_parameters']) for row in valid),
      'source_complexity_gain_larger_count':sum(row['comparison']['source_complexity_gain_larger'] for row in valid),
      'metrics':{}}
    for name in ('source_complexity_delta_chi2','tnfw_delta_chi2',
                 'source_complexity_percent_of_one_sersic_chi2','tnfw_percent_of_one_sersic_chi2',
                 'source_complexity_delta_minus_tnfw_delta_chi2'):
        values=np.asarray([row['comparison'][name] for row in valid],dtype=float)
        summary['metrics'][name]={'median':float(np.median(values)),
          'p16':float(np.quantile(values,.16)),'p84':float(np.quantile(values,.84)),
          'n_positive':int(np.sum(values>0))}
    output.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/p2-153-source-structure.jsonl')
    parser.add_argument('--b5-input',default='docs/b5-153-real-robustness.jsonl')
    parser.add_argument('--files-list',help='newline-separated FITS filenames; defaults to cutouts in B5 input')
    args=parser.parse_args(argv)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    b5_rows={}
    for line in Path(args.b5_input).read_text(encoding='utf-8').splitlines():
        if line:
            row=json.loads(line);b5_rows[row['file']]=row
    data_dir=Path('data/q1_cutouts')
    if args.files_list:
        names=[line.strip() for line in Path(args.files_list).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(names)!=len(set(names)):parser.error('files-list contains duplicate names')
        if any(Path(name).name!=name or not name.endswith('_VIS.fits') for name in names):
            parser.error('files-list entries must be FITS basenames')
        files=sorted(data_dir/name for name in names)
    else:
        files=sorted(data_dir/name for name in b5_rows)
    selected={path.name for path in files}
    if not selected or set(b5_rows)!=selected:
        parser.error('selected FITS files need matching B5 rows')
    if any(not path.is_file() for path in files):parser.error('selected cutout file is missing')
    rows={}
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            if line:
                row=json.loads(line)
                if row['file'] in rows:raise ValueError(f'duplicate source comparison for {row["file"]}')
                rows[row['file']]=row
    if not set(rows).issubset({path.name for path in files}):raise ValueError('output contains unknown cutout rows')
    with output.open('a',encoding='utf-8') as stream:
      for number,path in enumerate(files,1):
        if path.name in rows:
            print(f'source structure {number}/{len(files)} already complete: {path.name}',flush=True);continue
        baseline=b5_rows[path.name]
        if 'baseline' not in baseline or 'scan' not in baseline:raise ValueError(f'missing B5 baseline/scan for {path.name}')
        cutout,_,_=load_q1_fits(path)
        parameters=baseline['baseline']['parameters']
        vector_from_fit({'format':'skia-fit-v1','parameters':parameters})
        lens=Lens(**{key.removeprefix('lens.'):value for key,value in parameters.items() if key.startswith('lens.')})
        source=Source(**{key.removeprefix('source.'):value for key,value in parameters.items() if key.startswith('source.')})
        compact_source=Source(amp=3.,R_sersic=.06,n_sersic=1.,e1=source.e1,e2=source.e2,
                              center_x=source.center_x,center_y=source.center_y)
        initial=two_sersic_parameter_vector(lens,source,compact_source)
        try:
            fit=fit_cutout(cutout,CONFIG,initial_vector=initial)
            baseline_chi2=float(baseline['baseline']['chi2'])
            comparison=_compare(baseline_chi2,fit.chi2,float(baseline['scan']['best_delta_chi2']))
            item={'file':path.name,'tile_index':int(baseline['tile_index']),
              'one_sersic':{'chi2':baseline_chi2,'chi2_per_dof':baseline['baseline']['chi2_per_dof'],
                'converged':baseline['baseline']['converged'],'boundary_parameters':baseline['baseline']['boundary_parameters']},
              'two_sersic':{'chi2':fit.chi2,'chi2_per_dof':fit.chi2/fit.dof,'converged':fit.converged,
                'status':fit.status,'message':fit.message,'nfev':fit.nfev,
                'boundary_parameters':list(fit.boundary_parameters),'parameters':fit.parameters},
              'tnfw_scan':{'best_delta_chi2':baseline['scan']['best_delta_chi2'],
                'best':baseline['scan']['best'],'grid_points':baseline['scan']['grid_points'],
                'converged_points':baseline['scan']['converged_points']},
              'comparison':comparison}
        except Exception as error:
            item={'file':path.name,'tile_index':int(baseline['tile_index']),
                  'hard_error':f'{type(error).__name__}: {error}'}
        stream.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');stream.flush()
        rows[path.name]=item
        print(f'source structure {number}/{len(files)} {path.name}',flush=True)
    summary=_summarize([rows[path.name] for path in files],output.with_suffix('.summary.json'))
    print(json.dumps(summary,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
