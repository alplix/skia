"""Paired control vs outer-MAD noise and 10%-broader Gaussian PSF fits."""
import argparse, json
from pathlib import Path
import numpy as np
from recovery import LENS_RANGES, SOURCE_RANGES, _json
from skia.cutout import Cutout
from skia.fitter import FitConfig, PARAMETER_NAMES, fit_cutout, parameter_vector, physical_parameters
from skia.realdata import estimate_sky_noise
from skia.simulator import Lens, Source, Observation, gaussian_psf, simulate


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=200);p.add_argument('--output',type=Path,default=Path('docs/review-fix2'));a=p.parse_args(argv)
 if a.count<1:p.error('count must be positive')
 a.output.mkdir(parents=True,exist_ok=True);raw=a.output/'paired.jsonl'
 if raw.exists():p.error('refusing to overwrite prior outcomes')
 rng=np.random.Generator(np.random.PCG64(20261031));obs=Observation(npix=40,psf_fwhm=.16);rows=[]
 with raw.open('x') as h:
  for i in range(a.count):
   lens=Lens(**{k:float(rng.uniform(*v)) for k,v in LENS_RANGES.items()});source=Source(**{k:float(rng.uniform(*v)) for k,v in SOURCE_RANGES.items()})
   sim=simulate(lens,source,obs,seed=100000+i); estimate,mask=estimate_sky_noise(sim.image)
   variants={
    'known_noise_matched_psf':Cutout(sim.image-estimate.sky,sim.noise_map,sim.mask,gaussian_psf(.05,.16),.1,565,2),
    'estimated_noise_psf_plus_10pct':Cutout(sim.image-estimate.sky,np.full(sim.image.shape,estimate.sigma),mask,gaussian_psf(.05,.176),.1,565,2)}
   row={'i':i,'truth':physical_parameters(parameter_vector(lens,source)),'noise_estimate':estimate.__dict__,'fit_psf':{'known_noise_matched_psf':.16,'estimated_noise_psf_plus_10pct':.176},'snr_estimated':float(np.sqrt(np.sum(sim.model[mask]**2)/np.sum(estimate.sigma**2*np.ones(mask.sum())))),'fits':{}}
   for name,cut in variants.items():
    try:
     f=fit_cutout(cut,FitConfig(gamma_prior_sigma=.1));row['fits'][name]={'parameters':f.parameters,'converged':f.converged,'boundary_parameters':f.boundary_parameters,'chi2':f.chi2,'dof':f.dof,'prior_penalty':f.prior_penalty,'elapsed_seconds':f.elapsed_seconds}
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:row['fits'][name]={'parameters':None,'converged':False,'error':str(e)}
   rows.append(row);h.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');h.flush()
   if (i+1)%20==0 or i+1==a.count:print(f"noise/PSF {i+1}/{a.count}",flush=True)
 summary={}
 for method in ('known_noise_matched_psf','estimated_noise_psf_plus_10pct'):
  finite=[r for r in rows if r['fits'][method]['parameters'] is not None];metrics={}
  for j,name in enumerate(PARAMETER_NAMES):
   err=np.array([r['fits'][method]['parameters'][name]-r['truth'][name] for r in finite]);metrics[name]=None if not len(err) else {'bias':float(err.mean()),'scatter':float(err.std()),'rmse':float(np.mean(err**2)**.5)}
  te=np.array([r['fits'][method]['parameters']['lens.theta_E']-r['truth']['lens.theta_E'] for r in finite]);ge=np.array([r['fits'][method]['parameters']['lens.gamma']-r['truth']['lens.gamma'] for r in finite])
  summary[method]={'count':len(rows),'finite':len(finite),'converged':sum(r['fits'][method]['converged'] for r in rows),'boundary':sum(bool(r['fits'][method].get('boundary_parameters')) for r in rows),'gross':int(np.sum((abs(te)>.03)|(abs(ge)>.15))),'theta_E_gross':int(np.sum(abs(te)>.03)),'gamma_gross':int(np.sum(abs(ge)>.15)),'median_chi2_dof':float(np.median([r['fits'][method]['chi2']/r['fits'][method]['dof'] for r in finite])) if finite else None,'parameters':metrics}
 _json(a.output/'summary.json',summary)
 lines=['# Finding 2: estimated noise and PSF mismatch','',f'{len(rows)} paired lenses. Every branch uses the gamma prior sigma=0.1.','', 'The matched control uses oracle expected variance and true simulated PSF. The estimated branch uses the documented fixed outer-ring MAD and a +10% Gaussian PSF FWHM error. Noise and PSF changes are combined; causal contributions are not separable in this run.','', '| Outcome | Known variance / matched PSF | Estimated sky noise / PSF +10% |','|---|---:|---:|']
 for k,label in [('finite','finite fits'),('converged','converged'),('boundary','bound contacts'),('gross','gross recovery flags'),('theta_E_gross','theta_E gross errors'),('gamma_gross','gamma gross errors'),('median_chi2_dof','median image chi2/dof')]:lines.append(f"| {label} | {summary['known_noise_matched_psf'][k]} | {summary['estimated_noise_psf_plus_10pct'][k]} |")
 for name in ('lens.theta_E','lens.gamma','lens.gamma1','lens.gamma2','shear_magnitude'):
  lines+=['',f'## {name} bias / scatter / RMSE','', '| Branch | Bias | Scatter | RMSE |','|---|---:|---:|---:|']
  for method in summary:
   st=summary[method]['parameters'].get(name)
   if name=='shear_magnitude':
    e=[]
    for r in rows:
     pp=r['fits'][method]['parameters']
     if pp:e.append(np.hypot(pp['lens.gamma1'],pp['lens.gamma2'])-np.hypot(r['truth']['lens.gamma1'],r['truth']['lens.gamma2']))
    z=np.array(e);st={'bias':float(z.mean()),'scatter':float(z.std()),'rmse':float(np.mean(z**2)**.5)} if len(z) else None
   if st:lines.append(f"| {method} | {st['bias']:.7g} | {st['scatter']:.7g} | {st['rmse']:.7g} |")
 lines+=['',f"Median estimated image S/N: {np.median([r['snr_estimated'] for r in rows]):.6g}.",'','Raw paired estimates, ring sigmas and masks are in `paired.jsonl`; machine summary is in `summary.json`. Results describe a matched source family and one stated PSF perturbation, not a full PSF/noise uncertainty distribution.','']
 (a.output/'REPORT.md').write_text('\n'.join(lines));print('SUMMARY: '+json.dumps(summary,sort_keys=True),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
