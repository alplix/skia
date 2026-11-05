"""Profile scan and compare scalar versus vectorized TNFW field construction."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
import cProfile
import json
import pstats
from io import StringIO
from pathlib import Path
import platform
import time

import numpy as np

from skia.cutout import load_cutout
from skia.fitter import FitConfig, fit_cutout, save_fit
from skia.scanner import halo_deflection_field,halo_deflection_fields,scan
from skia.simulator import Observation,save_simulation,simulate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points-per-axis',type=int,default=2)
    parser.add_argument('--output',default='docs/b4-profile.json')
    args=parser.parse_args(argv)
    if args.points_per_axis<1:parser.error('points-per-axis must be positive')
    work=Path('work/b4-profile');work.mkdir(parents=True,exist_ok=True)
    path=work/'cutout.npz';save_simulation(simulate(observation=Observation(npix=30),seed=42042),path)
    cut=load_cutout(path);config=FitConfig(starts=1,seed=731,max_nfev=120,gamma_prior_sigma=.1)
    fit_cutout(cut,config) # JAX compile warm-up
    fit=fit_cutout(cut,config);fit_path=work/'fit.json';save_fit(fit,fit_path)
    import json as json_module
    baseline=json_module.loads(fit_path.read_text())
    axis=np.linspace(-.4,.4,args.points_per_axis).tolist();strengths=np.linspace(0,.03,args.points_per_axis).tolist()
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,'x_arcsec':axis,'y_arcsec':axis,'alpha_Rs_arcsec':strengths}
    points=[(float(x),float(y),float(a)) for x in axis for y in axis for a in strengths]
    scalar_start=time.perf_counter()
    scalar=np.array([halo_deflection_field(cut,x,y,a,.2,.8) for x,y,a in points])
    scalar_seconds=time.perf_counter()-scalar_start
    batch_start=time.perf_counter()
    batch=halo_deflection_fields(cut,points,.2,.8,chunk_size=16)
    batch_seconds=time.perf_counter()-batch_start
    max_difference=float(np.max(np.abs(scalar-batch)))
    scan(cut,baseline,grid);scan(cut,baseline,grid,vectorized_fields=True) # warm scan shapes
    start=time.perf_counter();result=scan(cut,baseline,grid);scan_seconds=time.perf_counter()-start
    start=time.perf_counter();batch_result=scan(cut,baseline,grid,vectorized_fields=True);vectorized_scan_seconds=time.perf_counter()-start
    if [r['chi2'] for r in result['results']] != [r['chi2'] for r in batch_result['results']]:
        raise RuntimeError('Scalar and vectorized scan results do not match')
    profiler=cProfile.Profile();profiler.enable();scan(cut,baseline,grid);profiler.disable()
    stream=StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(15)
    affinity=hasattr(os,'sched_getaffinity')
    payload={'format':'skia-scan-profile-v1','machine':platform.platform(),'processor':platform.processor() or platform.machine(),
      'python':platform.python_version(),'jax_backend':'cpu','thread_env_limit':1,'serial_process':True,
      'cpu_affinity_api_available':affinity,'cpu_affinity_enforced':False,
      'pixel_shape':list(cut.image.shape),'grid_shape':[len(axis),len(axis),len(strengths)],'grid_points':len(points),
      'starts':config.starts,'max_nfev':config.max_nfev,'warm_jax':True,
      'scalar_field_seconds':scalar_seconds,'vectorized_field_seconds':batch_seconds,
      'field_generation_speedup':scalar_seconds/batch_seconds if batch_seconds else None,
      'field_max_abs_difference_arcsec':max_difference,'scan_seconds_including_baseline':scan_seconds,
      'vectorized_scan_seconds_including_baseline':vectorized_scan_seconds,
      'vectorized_scan_speedup':scan_seconds/vectorized_scan_seconds if vectorized_scan_seconds else None,
      'estimated_grid_seconds_per_point':max(0.,scan_seconds-fit.elapsed_seconds)/len(points),
      'baseline_seconds':fit.elapsed_seconds,'converged_grid_points':sum(bool(r['converged']) for r in result['results']),
      'profile_top_cumulative':stream.getvalue()}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='profile_top_cumulative'},sort_keys=True))
    print(stream.getvalue())
    return 0

if __name__=='__main__':raise SystemExit(main())
