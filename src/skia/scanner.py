"""CPU subhalo proxy scan over angular lensing strengths (not solar masses)."""
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import yaml
from lenstronomy.LensModel.Profiles.tnfw import TNFW

from skia.cutout import Cutout, load_cutout
from skia.fitter import (FitConfig, PARAMETER_NAMES, TWO_SERSIC_PARAMETER_NAMES,
                         SmoothRenderer, fit_cutout, parameter_vector, two_sersic_parameter_vector)
from skia.simulator import Lens, Source


def vector_from_fit(payload: dict) -> np.ndarray:
    """Pack physical fit-JSON parameters for deterministic scan initialization."""
    if not isinstance(payload,dict): raise ValueError('Baseline must be a mapping')
    if payload.get('format')!='skia-fit-v1': raise ValueError('Expected skia-fit-v1 baseline')
    p=payload.get('parameters',{})
    if not isinstance(p,dict): raise ValueError('Baseline parameters must be a mapping')
    p=dict(p)
    required={f'lens.{name}' for name in Lens.__dataclass_fields__}|{f'source.{name}' for name in Source.__dataclass_fields__}
    if not required.issubset(p): raise ValueError(f'Baseline is missing parameters: {sorted(required-set(p))}')
    source2_keys={'source2.amp','source2.R_sersic','source2.n_sersic'}
    present=source2_keys.intersection(p)
    if present and present!=source2_keys: raise ValueError('Two-Sersic baseline has incomplete source2 parameters')
    numeric_keys=required|(source2_keys if present else set())
    for name in numeric_keys:
        value=p[name]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)):
            raise ValueError(f'Baseline parameter {name} must be a finite real number')
        try:number=float(value)
        except (TypeError,ValueError,OverflowError) as error:
            raise ValueError(f'Baseline parameter {name} must be a finite real number') from error
        if not np.isfinite(number):raise ValueError(f'Baseline parameter {name} must be a finite real number')
        p[name]=number
    lens=Lens(**{k.removeprefix('lens.'):v for k,v in p.items() if k.startswith('lens.')})
    source=Source(**{k.removeprefix('source.'):v for k,v in p.items() if k.startswith('source.')})
    if present:
        source2=Source(amp=p['source2.amp'],R_sersic=p['source2.R_sersic'],n_sersic=p['source2.n_sersic'],
                       e1=source.e1,e2=source.e2,center_x=source.center_x,center_y=source.center_y)
        return two_sersic_parameter_vector(lens,source,source2)
    return parameter_vector(lens,source)


def load_grid(path: str | Path) -> dict:
    """Load explicit TNFW angular-coordinate grid; alpha_Rs is arcsec, not Msun."""
    with Path(path).open(encoding='utf-8') as handle: grid=yaml.safe_load(handle)
    return normalize_grid(grid)


def normalize_grid(grid: dict) -> dict:
    """Validate and normalize a decoded TNFW angular grid mapping."""
    if not isinstance(grid,dict) or grid.get('profile')!='TNFW': raise ValueError('grid profile must be TNFW')
    for name in ('Rs_arcsec','r_trunc_arcsec','x_arcsec','y_arcsec','alpha_Rs_arcsec'):
        if name not in grid: raise ValueError(f'missing grid field {name}')
    if any(_contains_boolean(grid[name]) for name in ('Rs_arcsec','r_trunc_arcsec','x_arcsec','y_arcsec','alpha_Rs_arcsec')):
        raise ValueError('TNFW grid values must be numeric; booleans are not accepted')
    try:
        rs=float(grid['Rs_arcsec']);rt=float(grid['r_trunc_arcsec'])
        axes=[np.asarray(grid[k],dtype=float) for k in ('x_arcsec','y_arcsec','alpha_Rs_arcsec')]
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('TNFW grid values must be numeric') from error
    if rs<=0 or rt<=0 or not np.isfinite([rs,rt]).all(): raise ValueError('TNFW angular radii must be finite and positive')
    if any(a.ndim!=1 or not len(a) or not np.isfinite(a).all() for a in axes): raise ValueError('each grid axis must be a nonempty finite list')
    if np.any(axes[2]<0): raise ValueError('alpha_Rs_arcsec must be nonnegative')
    return {'profile':'TNFW','Rs_arcsec':rs,'r_trunc_arcsec':rt,'x_arcsec':axes[0].tolist(),'y_arcsec':axes[1].tolist(),'alpha_Rs_arcsec':axes[2].tolist()}


def _finite_float_values(values, label: str) -> tuple[float,...]:
    """Convert numeric API inputs to finite floats with a stable error type."""
    if any(isinstance(value,(bool,np.bool_)) for value in values):
        raise ValueError(f'{label} must be numeric and finite')
    try:
        converted=tuple(float(value) for value in values)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError(f'{label} must be numeric and finite') from error
    if not np.isfinite(converted).all():
        raise ValueError(f'{label} must be numeric and finite')
    return converted


def _contains_boolean(value) -> bool:
    """Find booleans recursively before NumPy silently casts them to 0/1."""
    if isinstance(value,(bool,np.bool_)):return True
    if isinstance(value,np.ndarray):
        if np.issubdtype(value.dtype,np.bool_):return True
        if value.dtype==object:return any(_contains_boolean(item) for item in value.flat)
        return False
    if isinstance(value,(list,tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def halo_deflection_field(cutout: Cutout, x_arcsec: float, y_arcsec: float,
                          alpha_Rs_arcsec: float, Rs_arcsec: float, r_trunc_arcsec: float) -> np.ndarray:
    """Return padded subpixel TNFW deflection (arcsec) for one fixed grid point."""
    x_arcsec,y_arcsec,alpha_Rs_arcsec,Rs_arcsec,r_trunc_arcsec=_finite_float_values(
        (x_arcsec,y_arcsec,alpha_Rs_arcsec,Rs_arcsec,r_trunc_arcsec),'TNFW parameters')
    if alpha_Rs_arcsec<0 or min(Rs_arcsec,r_trunc_arcsec)<=0:
        raise ValueError('TNFW parameters must be finite; radii positive; strength nonnegative')
    os=cutout.oversample;pad=(cutout.psf_fine.shape[0]//2+os-1)//os;ny,nx=cutout.image.shape
    width,height=(nx+2*pad)*os,(ny+2*pad)*os
    scale=cutout.pixel_scale/os
    ix=np.arange(width,dtype=float)-(width-1)/2;iy=np.arange(height,dtype=float)-(height-1)/2
    xx,yy=np.meshgrid(ix*scale,iy*scale)
    if alpha_Rs_arcsec==0:return np.zeros((height,width,2),dtype=float)
    ax,ay=TNFW().derivatives(xx,yy,Rs=Rs_arcsec,alpha_Rs=alpha_Rs_arcsec,
                              r_trunc=r_trunc_arcsec,center_x=x_arcsec,center_y=y_arcsec)
    field=np.stack((ax,ay),axis=-1)
    if not np.isfinite(field).all():raise ValueError('TNFW produced a nonfinite deflection field')
    return field


def halo_deflection_fields(cutout: Cutout, points: list[tuple[float,float,float]],
                           Rs_arcsec: float, r_trunc_arcsec: float,
                           chunk_size: int=16) -> np.ndarray:
    """Vectorize TNFW angular deflection generation; return (point,y,x,xy) arcsec fields."""
    if type(chunk_size) is not int or chunk_size<1: raise ValueError('chunk_size must be a positive integer')
    Rs_arcsec,r_trunc_arcsec=_finite_float_values((Rs_arcsec,r_trunc_arcsec),'TNFW radii')
    if min(Rs_arcsec,r_trunc_arcsec)<=0:
        raise ValueError('TNFW radii must be finite and positive')
    if not points:
        os=cutout.oversample;pad=(cutout.psf_fine.shape[0]//2+os-1)//os;ny,nx=cutout.image.shape
        return np.empty((0,(ny+2*pad)*os,(nx+2*pad)*os,2),dtype=float)
    if _contains_boolean(points):
        raise ValueError('points must be numeric finite (x_arcsec,y_arcsec,alpha_Rs_arcsec) triples')
    try:
        values=np.asarray(points,dtype=float)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('points must be finite (x_arcsec,y_arcsec,alpha_Rs_arcsec) triples') from error
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise ValueError('points must be finite (x_arcsec,y_arcsec,alpha_Rs_arcsec) triples')
    if np.any(values[:,2]<0):
        raise ValueError('angular strengths must be nonnegative')
    os=cutout.oversample;pad=(cutout.psf_fine.shape[0]//2+os-1)//os;ny,nx=cutout.image.shape
    width,height=(nx+2*pad)*os,(ny+2*pad)*os;scale=cutout.pixel_scale/os
    ix=np.arange(width,dtype=float)-(width-1)/2;iy=np.arange(height,dtype=float)-(height-1)/2
    xx,yy=np.meshgrid(ix*scale,iy*scale);profile=TNFW();batches=[]
    for start in range(0,len(values),chunk_size):
        chunk=values[start:start+chunk_size];cx=chunk[:,0][None,None,:];cy=chunk[:,1][None,None,:]
        strength=chunk[:,2][None,None,:]
        ax,ay=profile.derivatives(xx[...,None],yy[...,None],Rs=Rs_arcsec,alpha_Rs=strength,
                                  r_trunc=r_trunc_arcsec,center_x=cx,center_y=cy)
        batch=np.stack((ax,ay),axis=-1).transpose(2,0,1,3)
        if not np.isfinite(batch).all():raise ValueError('TNFW produced nonfinite vectorized deflections')
        batches.append(batch)
    return np.concatenate(batches,axis=0)


def _checkpoint_fingerprint(cutout: Cutout, baseline: dict, grid: dict, config: FitConfig) -> str:
    digest=hashlib.sha256()
    for array in (cutout.image,cutout.noise_map,cutout.mask,cutout.psf_fine):
        contiguous=np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode());digest.update(repr(contiguous.shape).encode());digest.update(contiguous.tobytes())
    metadata={'pixel_scale':cutout.pixel_scale,'exposure':cutout.exposure,'oversample':cutout.oversample,
              'image_unit':cutout.image_unit,'baseline':baseline,'grid':grid,'config':config.__dict__}
    digest.update(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return digest.hexdigest()


def _save_checkpoint(payload: dict,path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(payload,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _is_finite_number(value) -> bool:
    """Whether value is a JSON number (excluding bool) with a finite value."""
    if type(value) not in (int,float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_checkpoint_rows(saved: dict, points: list[tuple[float,float,float]],
                              parameter_names: tuple[str,...]) -> None:
    """Reject malformed completed results before they enter scan aggregation."""
    rows=saved.get('results')
    if not isinstance(rows,list) or len(rows)>len(points):
        raise ValueError('Checkpoint result list is invalid')
    for row,point in zip(rows,points,strict=False):
        if not isinstance(row,dict):
            raise ValueError('Checkpoint result row must be an object')
        try: row_point=(row['x_arcsec'],row['y_arcsec'],row['alpha_Rs_arcsec'])
        except KeyError as error: raise ValueError('Checkpoint grid-point record is incomplete') from error
        if any(not _is_finite_number(value) for value in row_point):
            raise ValueError('Checkpoint grid-point coordinates are invalid')
        if row_point!=point:
            raise ValueError('Checkpoint grid-point order is invalid')
        numeric=('chi2','delta_chi2','prior_penalty')
        if any(not _is_finite_number(row.get(name)) for name in numeric):
            raise ValueError('Checkpoint result has an invalid numeric value')
        if type(row.get('converged')) is not bool:
            raise ValueError('Checkpoint convergence flag is invalid')
        if not isinstance(row.get('boundary_parameters'),list) or any(
                not isinstance(name,str) or name not in parameter_names
                for name in row['boundary_parameters']):
            raise ValueError('Checkpoint boundary parameter list is invalid')
        parameters=row.get('parameters')
        if not isinstance(parameters,dict) or set(parameters)!=set(parameter_names) or any(
                not isinstance(name,str) or not _is_finite_number(value)
                for name,value in parameters.items()):
            raise ValueError('Checkpoint fitted parameters are invalid')
        if (type(row.get('status')) is not int or type(row.get('nfev')) is not int
                or row['nfev']<0 or not isinstance(row.get('message'),str)):
            raise ValueError('Checkpoint optimizer record is invalid')


def scan(cutout: Cutout, baseline: dict, grid: dict, *,
         checkpoint_path: str|Path|None=None,
         progress_callback: Callable[[int,int],None]|None=None,
         vectorized_fields: bool=False) -> dict:
    """Refit nuisance parameters per grid point; optionally checkpoint/resume.

    Checkpoints are local implementation state. Callback receives completed and
    total grid-point counts, not a BOINC API progress value.
    """
    grid=normalize_grid(grid)
    base_vector=vector_from_fit(baseline)
    optimizer=baseline.get('optimizer',{})
    config=FitConfig(**optimizer) if optimizer else FitConfig()
    parameter_names=TWO_SERSIC_PARAMETER_NAMES if config.source_components==2 else PARAMETER_NAMES
    points=[(float(x),float(y),float(strength)) for x in grid['x_arcsec'] for y in grid['y_arcsec'] for strength in grid['alpha_Rs_arcsec']]
    fingerprint=_checkpoint_fingerprint(cutout,baseline,grid,config)
    checkpoint=Path(checkpoint_path) if checkpoint_path is not None else None
    saved=None
    zero_field_fit=None
    if checkpoint is not None and checkpoint.exists():
        try: saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as error: raise ValueError(f'Invalid scan checkpoint: {error}') from error
        if not isinstance(saved,dict): raise ValueError('Invalid scan checkpoint: expected an object')
        if saved.get('format')!='skia-checkpoint-v1' or saved.get('fingerprint')!=fingerprint:
            raise ValueError('Checkpoint does not match this cutout, baseline, grid, and configuration')
        _validate_checkpoint_rows(saved,points,parameter_names)
        rows=saved['results']
        baseline_chi2=saved.get('baseline_chi2');baseline_converged=saved.get('baseline_converged');baseline_parameters=saved.get('baseline_parameters')
        if (not _is_finite_number(baseline_chi2) or type(baseline_converged) is not bool
                or not isinstance(baseline_parameters,dict) or set(baseline_parameters)!=set(parameter_names) or any(
                    not isinstance(name,str) or not _is_finite_number(value)
                    for name,value in baseline_parameters.items())):
            raise ValueError('Checkpoint baseline record is invalid')
    else:
        rows=[]
        base=fit_cutout(cutout,config,initial_vector=base_vector)
        baseline_chi2=base.chi2;baseline_converged=base.converged;baseline_parameters=base.parameters
        zero_field_fit=base
        if checkpoint is not None:
            saved={'format':'skia-checkpoint-v1','fingerprint':fingerprint,'baseline_chi2':baseline_chi2,
                   'baseline_converged':baseline_converged,'baseline_parameters':baseline_parameters,'results':rows}
            _save_checkpoint(saved,checkpoint)
    # A grid point whose field is all +0.0 (alpha_Rs_arcsec 0, a third of every task) is the baseline problem again:
    # the same data, start vector, configuration and seed, and the baseline's field is jnp.zeros of the same shape.
    # The fit is deterministic (docs/P3_DETERMINISM.md), so its result is the baseline's bit for bit and is reused.
    # After a resume the baseline fit is not in memory: the first zero-field point is fitted and reused after that.
    remaining=points[len(rows):]
    def complete_point(index,point,field):
        nonlocal zero_field_fit
        x,y,strength=point
        if _is_zero_field(field):
            if zero_field_fit is None:
                zero_field_fit=fit_cutout(cutout,config,initial_vector=base_vector,deflection_field=field)
            result=zero_field_fit
        else:
            result=fit_cutout(cutout,config,initial_vector=base_vector,deflection_field=field)
        rows.append({'x_arcsec':float(x),'y_arcsec':float(y),'alpha_Rs_arcsec':float(strength),
                     'chi2':result.chi2,'delta_chi2':baseline_chi2-result.chi2,'converged':result.converged,
                     'boundary_parameters':list(result.boundary_parameters),'parameters':result.parameters,
                     'prior_penalty':result.prior_penalty,'status':result.status,'nfev':result.nfev,
                     'message':result.message})
        if checkpoint is not None:
            saved.update(results=rows)
            _save_checkpoint(saved,checkpoint)
        if progress_callback is not None: progress_callback(index+1,len(points))
    if vectorized_fields:
      for offset in range(0,len(remaining),16):
        block=remaining[offset:offset+16]
        fields=halo_deflection_fields(cutout,block,grid['Rs_arcsec'],grid['r_trunc_arcsec'])
        for index,(point,field) in enumerate(zip(block,fields,strict=True),start=len(rows)):
            complete_point(index,point,field)
    else:
      for index,(x,y,strength) in enumerate(remaining,start=len(rows)):
        field=halo_deflection_field(cutout,x,y,strength,grid['Rs_arcsec'],grid['r_trunc_arcsec'])
        complete_point(index,(x,y,strength),field)
    best=max(rows,key=lambda r:(r['delta_chi2'],-r['x_arcsec'],-r['y_arcsec'],-r['alpha_Rs_arcsec']))
    return {'format':'skia-scan-v1','profile':'spherical-TNFW','strength_parameter':'alpha_Rs_arcsec',
            'mass_unit':'not-calibrated','baseline_chi2':baseline_chi2,'baseline_converged':baseline_converged,
            'baseline_parameters':baseline_parameters,'grid':grid,'best':best,'results':rows}


def _is_zero_field(field) -> bool:
    """Whether every deflection value is +0.0 (bit pattern 0), the field the baseline fit uses."""
    values=np.ascontiguousarray(field)
    return values.dtype==np.float64 and not values.view(np.uint64).any()


def save_scan(result: dict,path: str|Path) -> None:
    """Write deterministic fixed-precision text; timing is intentionally excluded."""
    rows=['# skia-scan-v1 profile=spherical-TNFW mass=not-calibrated','x_arcsec y_arcsec alpha_Rs_arcsec delta_chi2 chi2 converged']
    for r in result['results']:
        rows.append(' '.join(f"{r[k]:.10f}" for k in ('x_arcsec','y_arcsec','alpha_Rs_arcsec','delta_chi2','chi2'))+f" {int(r['converged'])}")
    Path(path).write_text('\n'.join(rows)+'\n',encoding='utf-8')
