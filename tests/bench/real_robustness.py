"""Run the full baseline+scan on every supplied Q1 cutout and retain failures."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from skia.fitter import FitConfig,fit_cutout
from skia.realdata import load_q1_fits
from skia.scanner import scan


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/b5-153-real-robustness.jsonl')
    parser.add_argument('--files-list',help='newline-separated FITS filenames; defaults to every manifested cutout')
    parser.add_argument('--max-nfev',type=int,default=80)
    args=parser.parse_args(argv)
    if args.max_nfev<1:parser.error('max-nfev must be positive')
    data_dir=Path('data/q1_cutouts')
    if args.files_list:
        names=[line.strip() for line in Path(args.files_list).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(names)!=len(set(names)):parser.error('files-list contains duplicate names')
        if any(Path(name).name!=name or not name.endswith('_VIS.fits') for name in names):
            parser.error('files-list entries must be FITS basenames')
        files=sorted(data_dir/name for name in names)
    else:
        files=sorted(data_dir.glob('*_VIS.fits'))
    manifest_path=Path('data/q1_cutouts/manifest.csv')
    with manifest_path.open(encoding='utf-8',newline='') as handle:
        manifest={row['file']:row for row in csv.DictReader(handle)}
    if not files:parser.error('no Q1 cutouts found')
    selected={path.name for path in files}
    if not selected.issubset(set(manifest)):
        parser.error('selected cutout files do not match manifest.csv')
    if any(not path.is_file() for path in files):parser.error('selected cutout file is missing')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():parser.error('output exists; choose a new file')
    cfg=FitConfig(starts=1,seed=731,max_nfev=args.max_nfev,gamma_prior_sigma=.1)
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,
          'x_arcsec':[-.3,.3],'y_arcsec':[-.3,.3],'alpha_Rs_arcsec':[0.,.02,.05]}
    total_points=0;unconverged=[];baseline_failures=[];bounds=0
    comparisons=[];hard_errors=[]
    with output.open('x') as stream:
      for number,path in enumerate(files,1):
        started=time.perf_counter();row={'file':path.name,'tile_index':int(manifest[path.name]['tile_index'])}
        try:
            cut,noise,metadata=load_q1_fits(path)
            base=fit_cutout(cut,cfg)
            row.update(noise={'sky':noise.sky,'sigma':noise.sigma,'usable':noise.fit_pixels},metadata=metadata,
                       baseline={'converged':base.converged,'status':base.status,'message':base.message,
                         'nfev':base.nfev,'chi2':base.chi2,'chi2_per_dof':base.chi2/base.dof,
                         'boundary_parameters':list(base.boundary_parameters),'parameters':base.parameters})
            reference=manifest[path.name]
            fitted=base.parameters
            fit_e=float(np.hypot(fitted['lens.e1'],fitted['lens.e2']))
            ref_e=float(np.hypot(float(reference['ref_ell_comp_0']),float(reference['ref_ell_comp_1'])))
            comparison={
              'theta_E_fit_arcsec':fitted['lens.theta_E'],
              'theta_E_reference_arcsec':float(reference['ref_einstein_radius_arcsec']),
              'theta_E_difference_arcsec':fitted['lens.theta_E']-float(reference['ref_einstein_radius_arcsec']),
              'shear_fit':float(np.hypot(fitted['lens.gamma1'],fitted['lens.gamma2'])),
              'shear_reference':float(np.hypot(float(reference['ref_shear_g1']),float(reference['ref_shear_g2']))),
              'axis_ratio_fit':(1-fit_e)/(1+fit_e),'axis_ratio_reference':(1-ref_e)/(1+ref_e)}
            comparison['shear_difference']=comparison['shear_fit']-comparison['shear_reference']
            comparison['axis_ratio_difference']=comparison['axis_ratio_fit']-comparison['axis_ratio_reference']
            row['reference_comparison']=comparison
            comparisons.append(comparison)
            if not base.converged:baseline_failures.append({'file':path.name,'status':base.status,'message':base.message,'nfev':base.nfev})
            payload={'format':'skia-fit-v1','parameters':base.parameters,'optimizer':{
                'starts':cfg.starts,'seed':cfg.seed,'max_nfev':cfg.max_nfev,
                'gamma_prior_sigma':cfg.gamma_prior_sigma,'source_components':cfg.source_components}}
            result=scan(cut,payload,grid)
            failures=[]
            for point in result['results']:
                total_points+=1;bounds+=bool(point['boundary_parameters'])
                if not point['converged']:
                    failure={'x_arcsec':point['x_arcsec'],'y_arcsec':point['y_arcsec'],
                      'alpha_Rs_arcsec':point['alpha_Rs_arcsec'],'status':point['status'],
                      'nfev':point['nfev'],'message':point['message'],
                      'boundary_parameters':point['boundary_parameters'],'chi2':point['chi2']}
                    failures.append(failure);unconverged.append({'file':path.name,**failure})
            row.update(scan={'grid_points':len(result['results']),'converged_points':len(result['results'])-len(failures),
              'baseline_converged':result['baseline_converged'],'baseline_chi2':result['baseline_chi2'],
              'best_delta_chi2':result['best']['delta_chi2'],'best':result['best'],
              'unconverged_points':failures})
        except Exception as error:
            row['hard_error']=f'{type(error).__name__}: {error}'
            hard_errors.append({'file':path.name,'error':row['hard_error']})
        row['elapsed_seconds']=time.perf_counter()-started
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
        print(f'Q1 robustness {number}/{len(files)} {path.name}',flush=True)
    def metric(values):
        array=np.asarray(values,dtype=float)
        return {'median':float(np.median(array)),'median_absolute':float(np.median(np.abs(array))),
                'rmse':float(np.sqrt(np.mean(array**2))),
                'p16':float(np.quantile(array,.16)),'p84':float(np.quantile(array,.84))}
    reference_summary={}
    for key in ('theta_E_difference_arcsec','shear_difference','axis_ratio_difference'):
        reference_summary[key]=metric([row[key] for row in comparisons])
    summary={'format':'skia-b5-robustness-v1','cutouts':len(files),'grid_points_total':total_points,
      'unconverged_scan_points':len(unconverged),'boundary_contacts':bounds,
      'baseline_failures':baseline_failures,'unconverged_points':unconverged,
      'hard_errors':hard_errors,'reference_comparison_n':len(comparisons),
      'reference_comparison':reference_summary}
    sidecar=output.with_suffix('.summary.json')
    sidecar.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cutouts':len(files),'grid_points':total_points,'unconverged':len(unconverged),
                      'baseline_failures':len(baseline_failures),'boundary_contacts':bounds},sort_keys=True),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
