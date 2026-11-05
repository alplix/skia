"""Apply one frozen free-slope and one frozen prior fit to each delivered Q1 cutout."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

from skia.fitter import FitConfig, fit_cutout
from skia.realdata import load_q1_fits


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/q1_cutouts'))
    p.add_argument('--output',type=Path,default=Path('docs/P1_REAL_REPORT.md'))
    args=p.parse_args(argv)
    manifest={}
    with (args.data/'manifest.csv').open(newline='') as f:
      for row in csv.DictReader(f): manifest[row['file']]=row
    files=sorted(args.data.glob('*_VIS.fits'))
    if len(files)!=len(manifest) or len(files)==0: raise SystemExit('FITS/manifest counts differ or are empty')
    rows=[]
    for i,path in enumerate(files):
      row={'file':path.name,'id':path.stem.removesuffix('_VIS'),'fits':{}}
      try:
        cut,noise,meta=load_q1_fits(path)
        row.update({'observation':meta,'shape':list(cut.image.shape)})
        for name,prior in [('free',None),('gamma_prior_sigma_0.1',.1)]:
          try:
            fit=fit_cutout(cut,FitConfig(gamma_prior_sigma=prior))
            row['fits'][name]={'parameters':fit.parameters,'chi2':fit.chi2,'dof':fit.dof,
               'chi2_per_dof':fit.chi2/fit.dof,'converged':fit.converged,'status':fit.status,
               'message':fit.message,'boundary_parameters':fit.boundary_parameters,
               'prior_penalty':fit.prior_penalty,'elapsed_seconds':fit.elapsed_seconds,
               'attempts':fit.attempts}
          except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            row['fits'][name]={'error':str(e),'parameters':None,'converged':False}
      except Exception as e:
        row['error']=f'{type(e).__name__}: {e}'
      row['reference']={k:(float(manifest[row['file']][k]) if manifest[row['file']].get(k) else None)
                         for k in ('ref_einstein_radius_arcsec','ref_ell_comp_0','ref_ell_comp_1','ref_shear_g1','ref_shear_g2')}
      rows.append(row)
      print(f"real fit {i+1}/{len(files)}: {path.name}; failures={sum('error' in r or any(not f.get('converged',False) for f in r.get('fits',{}).values()) for r in rows)}",flush=True)
    output=args.output; output.parent.mkdir(parents=True,exist_ok=True)
    raw=output.with_suffix('.json')
    raw.write_text(json.dumps({'format':'skia-real-q1-fits-v1','rows':rows},indent=2,sort_keys=True,allow_nan=False)+'\n')
    text=['# P1.5 real Q1 cutout fits','',f'{len(files)} local FITS cutouts; every manifest row attempted. Results: `{raw.name}`.','',
      'The adapter subtracts the median of the fixed outer 15% border and estimates constant sky noise as 1.4826*MAD there. It masks nonfinite pixels and values more than 8 sky sigmas below that median; positive source flux is preserved. No gain/RMS/exposure is supplied; noise is sky-only. Fit PSF is a circular Gaussian with FWHM 0.16 arcsec. DATA.md says this is an unverified approximate VIS width. Image units remain native_flux; amplitude is not an electron calibration. No lens-specific choices or noise/PSF tuning.','',
      'Both all-free and gamma~Normal(2,0.1) fits use the same bounded, seeded optimizer. chi2/dof uses only the image-data term; prior penalty is separately recorded. Reference table comparisons are against Euclid model estimates, not truth. Centre is excluded because its coordinate origin is undocumented. Ellipticity comparison uses q=(1-|e|)/(1+|e|); shear comparison uses magnitude.','',
      '| File | Shape | Noise sigma (native/pixel) | PSF FWHM | Free chi2/dof | Free converged | Prior chi2/dof | Prior converged | Bound contacts | Ref theta_E | Fit theta_E free | Ref shear | Fit shear free | Ref q | Fit q free |','|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
      m=r.get('observation',{}); a=r.get('fits',{}).get('free',{}); b=r.get('fits',{}).get('gamma_prior_sigma_0.1',{})
      def fmt(x): return '—' if x is None else f'{x:.4g}'
      def theta(f): return f.get('parameters',{}).get('lens.theta_E') if f else None
      def shear(f):
        pp=f.get('parameters') if f else None
        return np.hypot(pp['lens.gamma1'],pp['lens.gamma2']) if pp else None
      def q(f):
        pp=f.get('parameters') if f else None
        e=np.hypot(pp['lens.e1'],pp['lens.e2']) if pp else None
        return (1-e)/(1+e) if e is not None else None
      ref=r.get('reference',{})
      e=np.hypot(ref['ref_ell_comp_0'],ref['ref_ell_comp_1']) if ref.get('ref_ell_comp_0') is not None and ref.get('ref_ell_comp_1') is not None else None
      refq=(1-e)/(1+e) if e is not None else None
      rs=np.hypot(ref['ref_shear_g1'],ref['ref_shear_g2']) if ref.get('ref_shear_g1') is not None and ref.get('ref_shear_g2') is not None else None
      shape='x'.join(map(str,r.get('shape',[]))) or 'load failure'
      contacts=','.join(a.get('boundary_parameters',[])) or 'none'
      text.append(f"| {r['file']} | {shape} | {fmt(m.get('sigma_native_flux_per_pixel'))} | {fmt(m.get('psf_fwhm_arcsec_assumed'))} | {fmt(a.get('chi2_per_dof'))} | {a.get('converged','failed')} | {fmt(b.get('chi2_per_dof'))} | {b.get('converged','failed')} | {contacts} | {fmt(ref.get('ref_einstein_radius_arcsec'))} | {fmt(theta(a))} | {fmt(rs)} | {fmt(shear(a))} | {fmt(refq)} | {fmt(q(a))} |")
    for method in ('free','gamma_prior_sigma_0.1'):
      valid=[r['fits'].get(method,{}) for r in rows if r['fits'].get(method,{}).get('parameters')]
      text+=['',f'## {method}: summary','',f"Finite: {len(valid)}/{len(rows)}; converged: {sum(x.get('converged',False) for x in valid)}/{len(rows)}; bound contact: {sum(bool(x.get('boundary_parameters')) for x in valid)}/{len(rows)}; median chi2/dof: {fmt(float(np.median([x['chi2_per_dof'] for x in valid])) if valid else None)}."]
      for key,label in [('ref_einstein_radius_arcsec','theta_E'),('shear','shear magnitude'),('axis_ratio','q')]:
        diffs=[]
        for r in rows:
          f=r.get('fits',{}).get(method,{}); pp=f.get('parameters'); ref=r.get('reference',{})
          if not pp: continue
          if key=='ref_einstein_radius_arcsec': x,y=pp['lens.theta_E'],ref.get(key)
          elif key=='shear':
            x=np.hypot(pp['lens.gamma1'],pp['lens.gamma2']); y=np.hypot(ref['ref_shear_g1'],ref['ref_shear_g2']) if ref.get('ref_shear_g1') is not None and ref.get('ref_shear_g2') is not None else None
          else:
            e=np.hypot(pp['lens.e1'],pp['lens.e2']); x=(1-e)/(1+e); e=np.hypot(ref['ref_ell_comp_0'],ref['ref_ell_comp_1']) if ref.get('ref_ell_comp_0') is not None and ref.get('ref_ell_comp_1') is not None else None; y=(1-e)/(1+e) if e is not None else None
          if y is not None: diffs.append(x-y)
        text.append(f"- {label} fit-minus-reference median: {fmt(float(np.median(diffs)) if diffs else None)}; compared {len(diffs)}. Reference estimates are not truth.")
    text+=['','## Limitations and failures','','Fit failures, status messages, estimates and mask/noise metadata are retained per file in the JSON. These fits use a single-Sersic source and omit foreground lens light, and are exploratory, not science validated. No formal likelihood interpretation is possible until native image units and variance are calibrated. Reference comparisons are rough model-to-model checks only.','']
    output.write_text('\n'.join(text))
    return 0
if __name__=='__main__': raise SystemExit(main())
