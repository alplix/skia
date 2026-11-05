"""Compare Herculens/JAX and lenstronomy/SciPy fits on seeded synthetic lenses."""
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
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from skia.cutout import Cutout
from skia.fitter import FitConfig, fit_cutout
from skia.simulator import Lens, Observation, Source, gaussian_psf, simulate

_recovery_spec=importlib.util.spec_from_file_location('skia_recovery_bench',_BENCH_PATH/'recovery.py')
_recovery=importlib.util.module_from_spec(_recovery_spec)
_recovery_spec.loader.exec_module(_recovery)
LENS_RANGES=_recovery.LENS_RANGES
SOURCE_RANGES=_recovery.SOURCE_RANGES

LOWER=np.array([.4,1.6,-.45,-.45,-.2,-.2,-.15,-.15,
                np.log(.01),np.log(.03),.6,-.45,-.45,-.4,-.4])
UPPER=np.array([1.6,2.4,.45,.45,.2,.2,.15,.15,
                np.log(1000),np.log(.8),4.,.45,.45,.4,.4])
SCALES=np.array([.12,.08,.06,.06,.025,.025,.02,.02,.4,.2,.25,.08,.08,.06,.06])


def _vector(lens,source):
    return np.array([lens.theta_E,lens.gamma,lens.e1,lens.e2,lens.center_x,lens.center_y,
        lens.gamma1,lens.gamma2,np.log(source.amp),np.log(source.R_sersic),source.n_sersic,
        source.e1,source.e2,source.center_x,source.center_y],dtype=float)


def _physical(vector):
    values=np.asarray(vector,dtype=float).copy();values[8:10]=np.exp(values[8:10])
    lens=dict(zip(('theta_E','gamma','e1','e2','center_x','center_y','gamma1','gamma2'),values[:8].tolist(),strict=True))
    source=dict(zip(('amp','R_sersic','n_sersic','e1','e2','center_x','center_y'),
                    np.array([values[8],values[9],*values[10:15]],dtype=float).tolist(),strict=True))
    return {'lens':lens,'source':source}


class LenstronomyRenderer:
    """Independent lenstronomy EPL/shear/Sersic forward model on a fixed cutout."""
    def __init__(self,cutout: Cutout):
        self.cutout=cutout
        self.lens=LensModel(lens_model_list=['EPL','SHEAR'])
        self.light=LightModel(light_model_list=['SERSIC_ELLIPSE'],
            profile_kwargs_list=[{'sersic_major_axis':True}])
        osamp=cutout.oversample;pad=(cutout.psf_fine.shape[0]//2+osamp-1)//osamp
        size=(cutout.image.shape[0]+2*pad)*osamp
        coord=(np.arange(size,dtype=float)-(size-1)/2)*cutout.pixel_scale/osamp
        self.xx,self.yy=np.meshgrid(coord,coord);self.pad=pad

    def render(self,vector: np.ndarray) -> np.ndarray:
        """Render source electrons/pixel using lenstronomy lens and light profiles."""
        p=np.asarray(vector,dtype=float)
        kwargs_lens=[{'theta_E':p[0],'gamma':p[1],'e1':p[2],'e2':p[3],
                      'center_x':p[4],'center_y':p[5]},
                     {'gamma1':p[6],'gamma2':p[7]}]
        alpha_x,alpha_y=self.lens.alpha(self.xx,self.yy,kwargs_lens)
        kwargs_source=[{'amp':np.exp(p[8]),'R_sersic':np.exp(p[9]),'n_sersic':p[10],
                        'e1':p[11],'e2':p[12],'center_x':p[13],'center_y':p[14]}]
        light=self.light.surface_brightness(self.xx-alpha_x,self.yy-alpha_y,kwargs_source)
        convolved=fftconvolve(light,self.cutout.psf_fine,mode='same')
        osamp=self.cutout.oversample;ny,nx=self.cutout.image.shape
        crop=convolved[self.pad*osamp:(self.pad+ny)*osamp,
                       self.pad*osamp:(self.pad+nx)*osamp]
        image=crop.reshape(ny,osamp,nx,osamp).mean(axis=(1,3))
        return np.maximum(image*self.cutout.pixel_scale**2*self.cutout.exposure,0.)


def fit_lenstronomy(cutout: Cutout,config: FitConfig):
    """Fit using scipy finite-difference Jacobians and lenstronomy profiles."""
    renderer=LenstronomyRenderer(cutout);initial=_vector(Lens(),Source())
    rng=np.random.Generator(np.random.PCG64(config.seed));candidates=[];attempts=[]
    for start in range(config.starts):
        p0=initial.copy() if start==0 else np.clip(initial+rng.normal(size=15)*SCALES,LOWER+1e-8,UPPER-1e-8)
        def residual(p):
            return ((renderer.render(p)-cutout.image)/cutout.noise_map)[cutout.mask]
        result=least_squares(residual,p0,bounds=(LOWER,UPPER),method='trf',x_scale='jac',
            max_nfev=config.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        chi2=float(result.fun@result.fun)
        attempts.append({'start':start,'chi2':chi2,'converged':bool(result.success),
                         'status':int(result.status),'nfev':int(result.nfev),'message':str(result.message)})
        candidates.append((chi2,start,result))
    chi2,start,result=min(candidates,key=lambda item:(item[0],item[1]))
    return {'parameters':_physical(result.x),'chi2':chi2,'converged':bool(result.success),
            'status':int(result.status),'nfev':int(result.nfev),'best_start':start,'attempts':attempts}


def _shear(parameters):
    return float(np.hypot(parameters['lens']['gamma1'],parameters['lens']['gamma2']))


def _axis_ratio(parameters):
    ellipticity=np.hypot(parameters['lens']['e1'],parameters['lens']['e2'])
    return float((1-ellipticity)/(1+ellipticity))


def crosscheck_fit(lens,source,observation,noise_seed,config):
    sim=simulate(lens,source,observation,seed=noise_seed)
    cutout=Cutout(sim.image,sim.noise_map,sim.mask,
        gaussian_psf(observation.pixel_scale/observation.oversample,observation.psf_fwhm),
        observation.pixel_scale,observation.exposure,observation.oversample)
    herculens_fit=fit_cutout(cutout,config)
    lenstronomy_fit=fit_lenstronomy(cutout,config)
    h={'parameters':herculens_fit.parameters,'chi2':herculens_fit.chi2,
       'converged':herculens_fit.converged,'status':herculens_fit.status,
       'nfev':herculens_fit.nfev,'best_start':herculens_fit.best_start}
    l=lenstronomy_fit
    differences={
        'theta_E_arcsec':h['parameters']['lens.theta_E']-l['parameters']['lens']['theta_E'],
        'gamma':h['parameters']['lens.gamma']-l['parameters']['lens']['gamma'],
        'shear_magnitude':float(np.hypot(h['parameters']['lens.gamma1'],h['parameters']['lens.gamma2'])-_shear(l['parameters'])),
        'axis_ratio':float((1-np.hypot(h['parameters']['lens.e1'],h['parameters']['lens.e2']))/(1+np.hypot(h['parameters']['lens.e1'],h['parameters']['lens.e2']))-_axis_ratio(l['parameters'])),
        'chi2':h['chi2']-l['chi2']}
    truth={'lens':asdict(lens),'source':asdict(source)}
    return {'noise_seed':noise_seed,'truth':truth,'herculens':h,'lenstronomy':l,'difference_herculens_minus_lenstronomy':differences}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=5)
    parser.add_argument('--max-nfev',type=int,default=200)
    parser.add_argument('--npix',type=int,default=32)
    parser.add_argument('--output',type=Path,default=Path('docs/lenstronomy-fit-crosscheck'))
    args=parser.parse_args(argv)
    if args.count<1 or args.max_nfev<1 or args.npix<16:parser.error('count/max-nfev must be positive and npix >=16')
    args.output.mkdir(parents=True,exist_ok=True);raw=args.output/'results.jsonl'
    if raw.exists():parser.error('results already exist; select a new output directory')
    observation=Observation(npix=args.npix,oversample=1)
    config=FitConfig(starts=2,seed=731,max_nfev=args.max_nfev)
    protocol={'format':'skia-lenstronomy-fit-crosscheck-v1','count':args.count,
        'parameter_seed':20261103,'noise_seed_base':700000,'observation':asdict(observation),
        'fit_config':asdict(config),'lens_ranges':LENS_RANGES,'source_ranges':SOURCE_RANGES,
        'independent_method':'lenstronomy 1.13.2 EPL/SHEAR/SERSIC_ELLIPSE forward model with scipy.optimize.least_squares finite-difference Jacobian',
        'comparison':'Same input images, noise maps, parameter bounds, start points and stopping settings. Fits are independent implementations; same initial vectors and objective.',
        'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        'git_worktree_dirty':bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip())}
    (args.output/'protocol.json').write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    rng=np.random.Generator(np.random.PCG64(20261103));rows=[]
    with raw.open('x') as handle:
        for index in range(args.count):
            lens=Lens(**{name:float(rng.uniform(*bounds)) for name,bounds in LENS_RANGES.items()})
            source=Source(**{name:float(rng.uniform(*bounds)) for name,bounds in SOURCE_RANGES.items()})
            row=crosscheck_fit(lens,source,observation,700000+index,config)
            row['index']=index;rows.append(row)
            handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
            print(f'lenstronomy fit cross-check {index+1}/{args.count}',flush=True)
    fields=tuple(rows[0]['difference_herculens_minus_lenstronomy'])
    summary={'count':len(rows),'converged':{
        'herculens':sum(row['herculens']['converged'] for row in rows),
        'lenstronomy':sum(row['lenstronomy']['converged'] for row in rows)},
        'median_absolute_parameter_difference':{
            name:float(np.median([abs(row['difference_herculens_minus_lenstronomy'][name]) for row in rows]))
            for name in fields},
        'max_absolute_parameter_difference':{
            name:float(max(abs(row['difference_herculens_minus_lenstronomy'][name]) for row in rows))
            for name in fields}}
    (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('LENSTRONOMY FIT CROSS-CHECK SUMMARY: '+json.dumps(summary,sort_keys=True),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
