"""Cross-check Herculens simulator deflections/light against lenstronomy."""
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
from scipy.signal import fftconvolve
from herculens.LightModel.Profiles.sersic import Sersic as HerculensSersic
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from skia.simulator import Lens, Observation, Source, deflection, gaussian_psf, simulate

_recovery_spec=importlib.util.spec_from_file_location('skia_recovery_bench',Path(__file__).with_name('recovery.py'))
_recovery=importlib.util.module_from_spec(_recovery_spec)
_recovery_spec.loader.exec_module(_recovery)
LENS_RANGES=_recovery.LENS_RANGES
SOURCE_RANGES=_recovery.SOURCE_RANGES


def _kwargs(lens,source):
    mass=[{'theta_E':lens.theta_E,'gamma':lens.gamma,'e1':lens.e1,'e2':lens.e2,
           'center_x':lens.center_x,'center_y':lens.center_y},
          {'gamma1':lens.gamma1,'gamma2':lens.gamma2}]
    light=[{'amp':source.amp,'R_sersic':source.R_sersic,'n_sersic':source.n_sersic,
            'e1':source.e1,'e2':source.e2,'center_x':source.center_x,'center_y':source.center_y}]
    return mass,light


def crosscheck_one(lens,source,observation,seed):
    """Compare native-image/arcsec lens and light fields on one fixed model."""
    simulated=simulate(lens,source,observation,seed=seed)
    osamp=observation.oversample;step=observation.pixel_scale/osamp
    fine_psf=gaussian_psf(step,observation.psf_fwhm)
    pad=(fine_psf.shape[0]//2+osamp-1)//osamp
    size=(observation.npix+2*pad)*osamp
    coord=(np.arange(size,dtype=float)-(size-1)/2)*step
    xx,yy=np.meshgrid(coord,coord)
    kwargs_lens,kwargs_source=_kwargs(lens,source)
    lens_model=LensModel(lens_model_list=['EPL','SHEAR'])
    light_model=LightModel(light_model_list=['SERSIC_ELLIPSE'],
                           profile_kwargs_list=[{'sersic_major_axis':True}])
    alpha_x,alpha_y=lens_model.alpha(xx,yy,kwargs_lens)
    ref_alpha_x,ref_alpha_y=deflection(xx,yy,lens)
    deflection_diff=np.stack((alpha_x-ref_alpha_x,alpha_y-ref_alpha_y),axis=-1)
    source_x=xx-alpha_x;source_y=yy-alpha_y
    alternate_light=light_model.surface_brightness(source_x,source_y,kwargs_source)
    herculens_light=np.asarray(HerculensSersic(radius_major_axis=True).function(
        source_x,source_y,**asdict(source)))
    light_diff=alternate_light-herculens_light
    convolved=fftconvolve(alternate_light,fine_psf,mode='same')
    crop=convolved[pad*osamp:(pad+observation.npix)*osamp,
                   pad*osamp:(pad+observation.npix)*osamp]
    image=crop.reshape(observation.npix,osamp,observation.npix,osamp).mean(axis=(1,3))
    image=np.maximum(image*observation.pixel_scale**2*observation.exposure,0.)
    image_diff=image-simulated.model
    rms=lambda values:float(np.sqrt(np.mean(np.square(values))))
    return {'seed':seed,'lens':asdict(lens),'source':asdict(source),
        'deflection_rms_arcsec':rms(deflection_diff),
        'deflection_max_abs_arcsec':float(np.max(np.abs(deflection_diff))),
        'source_brightness_rms':rms(light_diff),
        'source_brightness_max_abs':float(np.max(np.abs(light_diff))),
        'source_brightness_relative_rms':rms(light_diff)/max(rms(herculens_light),np.finfo(float).tiny),
        'noiseless_image_rms':rms(image_diff),
        'noiseless_image_max_abs':float(np.max(np.abs(image_diff))),
        'noiseless_image_relative_rms':rms(image_diff)/max(rms(simulated.model),np.finfo(float).tiny),
        'noiseless_image_max_fraction_of_peak':float(np.max(np.abs(image_diff))/np.max(simulated.model))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=5)
    parser.add_argument('--output',type=Path,default=Path('docs/lenstronomy-crosscheck'))
    args=parser.parse_args(argv)
    if args.count<1:parser.error('count must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'results.jsonl'
    if raw.exists():parser.error('results already exist; choose a new output directory')
    observation=Observation(npix=40)
    protocol={'format':'skia-lenstronomy-crosscheck-v1','count':args.count,
        'parameter_seed':20261103,'noise_seeds':'500000+index',
        'observation':asdict(observation),'lens_ranges':LENS_RANGES,'source_ranges':SOURCE_RANGES,
        'packages':{'lenstronomy':'1.13.2','reference':'herculens profiles used by skia.simulator'},
        'mass_models':'lenstronomy EPL + SHEAR; source model lenstronomy SERSIC_ELLIPSE with sersic_major_axis=True',
        'comparison':'At identical sampled parameters, compare arcsec deflections, source-plane surface brightness, and convolved/oversampled noiseless detector images. No parameter fit is run in lenstronomy.',
        'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()}
    (args.output/'protocol.json').write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    rng=np.random.Generator(np.random.PCG64(20261103));rows=[]
    with raw.open('x') as handle:
        for index in range(args.count):
            lens=Lens(**{name:float(rng.uniform(*bounds)) for name,bounds in LENS_RANGES.items()})
            source=Source(**{name:float(rng.uniform(*bounds)) for name,bounds in SOURCE_RANGES.items()})
            row=crosscheck_one(lens,source,observation,500000+index)
            rows.append(row);handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');handle.flush()
            print(f'lenstronomy cross-check {index+1}/{args.count}',flush=True)
    metrics={key:{'max':float(max(row[key] for row in rows)),
                  'median':float(np.median([row[key] for row in rows]))}
             for key in ('deflection_rms_arcsec','deflection_max_abs_arcsec',
                         'source_brightness_relative_rms','noiseless_image_relative_rms',
                         'noiseless_image_max_fraction_of_peak')}
    summary={'count':len(rows),'metrics':metrics}
    (args.output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('LENSTRONOMY CROSS-CHECK SUMMARY: '+json.dumps(summary,sort_keys=True),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
