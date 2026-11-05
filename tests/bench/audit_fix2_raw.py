"""Normalize physical truth values in a prior run without changing fit outputs."""
import json
from pathlib import Path
import numpy as np
from skia.fitter import PARAMETER_NAMES

root=Path(__file__).resolve().parents[2]
folder=root/'docs/review-fix2'
source=folder/'paired.jsonl'
normalized=folder/'paired-physical.jsonl'
rows=[]
with source.open() as handle, normalized.open('x') as out:
 for line in handle:
  row=json.loads(line); packed=row['truth']
  physical={name:float(value) for name,value in zip(PARAMETER_NAMES,packed,strict=True)}
  physical['source.amp']=float(np.exp(physical['source.amp']))
  physical['source.R_sersic']=float(np.exp(physical['source.R_sersic']))
  row['truth_packed_for_optimizer']=packed
  row['truth']=physical
  rows.append(row);out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
summary={}
for method in ('known_noise_matched_psf','estimated_noise_psf_plus_10pct'):
 finite=[r for r in rows if r['fits'][method]['parameters'] is not None];parameter_summary={};metrics={}
 for name in PARAMETER_NAMES:
  errors=np.array([r['fits'][method]['parameters'][name]-r['truth'][name] for r in finite])
  parameter_summary[name]={'bias':float(errors.mean()),'scatter':float(errors.std()),'rmse':float(np.mean(errors**2)**.5)} if len(errors) else None
 te=np.array([r['fits'][method]['parameters']['lens.theta_E']-r['truth']['lens.theta_E'] for r in finite])
 ge=np.array([r['fits'][method]['parameters']['lens.gamma']-r['truth']['lens.gamma'] for r in finite])
 metrics.update({'count':len(rows),'finite':len(finite),'converged':sum(r['fits'][method]['converged'] for r in rows),'boundary':sum(bool(r['fits'][method].get('boundary_parameters')) for r in rows),'gross':int(np.sum((abs(te)>.03)|(abs(ge)>.15))),'theta_E_gross':int(np.sum(abs(te)>.03)),'gamma_gross':int(np.sum(abs(ge)>.15)),'median_chi2_dof':float(np.median([r['fits'][method]['chi2']/r['fits'][method]['dof'] for r in finite]))})
 
 metrics['parameters']=parameter_summary
 summary[method]=metrics

(folder/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(f'AUDIT: normalized {len(rows)} paired rows to physical amp/radius; source truth preserved in paired-physical.jsonl')
