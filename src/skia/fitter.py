"""Bounded CPU/JAX least-squares fitting of an EPL+shear lens and Sersic source.

This is a Gaussian pixel likelihood with a fixed supplied noise map. It is not
Bayesian pixelated-source inference and makes no subhalo detection claim.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
import numpy as np
from scipy.optimize import least_squares
from herculens.MassModel.Profiles.epl import EPL
from herculens.MassModel.Profiles.shear import Shear
from herculens.LightModel.Profiles.sersic import Sersic

from skia.cutout import Cutout
from skia.simulator import Lens, Source

PARAMETER_NAMES = (
    'lens.theta_E', 'lens.gamma', 'lens.e1', 'lens.e2', 'lens.center_x', 'lens.center_y',
    'lens.gamma1', 'lens.gamma2', 'source.amp', 'source.R_sersic', 'source.n_sersic',
    'source.e1', 'source.e2', 'source.center_x', 'source.center_y',
)
TWO_SERSIC_PARAMETER_NAMES = PARAMETER_NAMES + ('source2.amp','source2.R_sersic','source2.n_sersic')
# Fixed search domain, shared by all lenses; no per-lens tuning or truth inputs.
LOWER = np.array([0.4, 1.6, -.45, -.45, -.2, -.2, -.15, -.15,
                  np.log(.01), np.log(.03), .6, -.45, -.45, -.4, -.4])
UPPER = np.array([1.6, 2.4, .45, .45, .2, .2, .15, .15,
                  np.log(1000), np.log(.8), 4, .45, .45, .4, .4])
LOWER_TWO_SERSIC=np.concatenate((LOWER,[np.log(.001),np.log(.01),.6]))
UPPER_TWO_SERSIC=np.concatenate((UPPER,[np.log(1000),np.log(.8),4.]))


def parameter_vector(lens: Lens, source: Source) -> np.ndarray:
    """Pack physical profiles in fixed order; log-transform positive amp and radius."""
    values = {**{'lens.'+k: v for k, v in asdict(lens).items()},
              **{'source.'+k: v for k, v in asdict(source).items()}}
    p = np.array([values[name] for name in PARAMETER_NAMES], dtype=float)
    if p[8] <= 0 or p[9] <= 0:
        raise ValueError('Fitted source amplitude and radius must be positive')
    p[8:10] = np.log(p[8:10])
    return p


def two_sersic_parameter_vector(lens: Lens, source: Source, source2: Source) -> np.ndarray:
    """Pack two shared-geometry Sersic components; component 2 adds amp, radius, index."""
    first=parameter_vector(lens,source)
    if source2.amp<=0: raise ValueError('Second Sersic amplitude must be positive')
    second=np.array([np.log(source2.amp),np.log(source2.R_sersic),source2.n_sersic])
    return np.concatenate((first,second))


def physical_parameters(vector: np.ndarray) -> dict[str, float]:
    """Unpack 15/18-vector parameters (angles arcsec, amp e-/s/arcsec²)."""
    p = np.array(vector, dtype=float, copy=True)
    if p.shape not in ((15,),(18,)) or not np.isfinite(p).all():
        raise ValueError('Expected a finite vector of 15 or 18 parameters')
    p[8:10] = np.exp(p[8:10])
    names=PARAMETER_NAMES
    if p.size==18:
        p[15:17]=np.exp(p[15:17]);names=TWO_SERSIC_PARAMETER_NAMES
    if not np.isfinite(p).all():
        raise ValueError('Physical parameters overflowed')
    return dict(zip(names, p.tolist(), strict=True))


def linearized_parameter_errors(jacobian: np.ndarray, parameters: dict[str,float],
                                chi2: float, dof: int) -> tuple[dict[str,float|None],int]:
    """Estimate local 1-sigma errors from weighted Jacobian; amplitudes/radii use delta method.

    Returns all-None errors when the residual Jacobian is rank deficient. This
    is a local Gaussian approximation, not a posterior interval.
    """
    jac=np.asarray(jacobian,dtype=float)
    if jac.ndim!=2 or jac.shape[1]!=len(parameters) or not np.isfinite(jac).all():
        raise ValueError('Jacobian must be finite with one column per fitted parameter')
    if not np.isfinite([chi2,dof]).all() or chi2<0 or dof<=0:
        raise ValueError('chi2 must be finite/nonnegative and dof positive')
    _,singular,vh=np.linalg.svd(jac,full_matrices=False)
    tolerance=np.finfo(float).eps*max(jac.shape)*(singular[0] if singular.size else 0.)
    rank=int(np.count_nonzero(singular>tolerance))
    if rank<jac.shape[1]: return {name:None for name in parameters},rank
    covariance=(vh.T/(singular**2))@vh
    covariance*=chi2/dof
    diagonal=np.maximum(np.diag(covariance),0.)
    errors={}
    log_parameters={'source.amp','source.R_sersic','source2.amp','source2.R_sersic'}
    for index,(name,value) in enumerate(parameters.items()):
        error=float(np.sqrt(diagonal[index]))
        if name in log_parameters: error*=value
        errors[name]=error
    return errors,rank


@lru_cache(maxsize=8)
def _compiled_renderer(ny: int, nx: int, oversample: int, psf_size: int):
    """Cache only geometry-specific functions, never lens truth or observed pixels."""
    os = oversample
    pad = (psf_size//2 + os - 1)//os
    width, height = (nx + 2*pad)*os, (ny + 2*pad)*os
    index_x = jnp.arange(width, dtype=jnp.float64) - (width-1)/2
    index_y = jnp.arange(height, dtype=jnp.float64) - (height-1)/2
    xx, yy = jnp.meshgrid(index_x, index_y)
    epl, shear, sersic = EPL(), Shear(), Sersic(radius_major_axis=True)

    def render(p, psf, pixel_scale, exposure, halo_deflection):
        x, y = xx*pixel_scale/os, yy*pixel_scale/os
        ax, ay = epl.derivatives(x, y, theta_E=p[0], gamma=p[1], e1=p[2], e2=p[3],
                                 center_x=p[4], center_y=p[5])
        sx, sy = shear.derivatives(x, y, gamma1=p[6], gamma2=p[7])
        source_x=x-ax-sx-halo_deflection[..., 0]
        source_y=y-ay-sy-halo_deflection[..., 1]
        light = sersic.function(source_x,source_y,amp=jnp.exp(p[8]), R_sersic=jnp.exp(p[9]),
                                n_sersic=p[10], e1=p[11], e2=p[12], center_x=p[13], center_y=p[14])
        if p.shape[0]==18:
            light=light+sersic.function(source_x,source_y,amp=jnp.exp(p[15]),R_sersic=jnp.exp(p[16]),
                n_sersic=p[17],e1=p[11],e2=p[12],center_x=p[13],center_y=p[14])
        blurred = fftconvolve(light, psf, mode='same')
        crop = blurred[pad*os:(pad+ny)*os, pad*os:(pad+nx)*os]
        image = crop.reshape(ny, os, nx, os).mean(axis=(1, 3))*pixel_scale**2*exposure
        return jnp.maximum(image, 0)

    return jax.jit(render), jax.jit(jax.jacfwd(render, argnums=0))


class SmoothRenderer:
    """Differentiable smooth forward model for a fixed detector geometry/PSF."""

    def __init__(self, cutout: Cutout):
        self.cutout = cutout
        self._render, self._jac = _compiled_renderer(*cutout.image.shape, cutout.oversample,
                                                     cutout.psf_fine.shape[0])
        self._args = (jnp.asarray(cutout.psf_fine), cutout.pixel_scale, cutout.exposure)

    def _deflection_arg(self, deflection_field: np.ndarray | None):
        os=self.cutout.oversample; pad=(self.cutout.psf_fine.shape[0]//2+os-1)//os
        ny,nx=self.cutout.image.shape; shape=((ny+2*pad)*os,(nx+2*pad)*os,2)
        if deflection_field is None: return jnp.zeros(shape,dtype=jnp.float64)
        field=np.asarray(deflection_field,dtype=np.float64)
        if field.shape!=shape or not np.isfinite(field).all(): raise ValueError(f'deflection_field must be finite with shape {shape}')
        return jnp.asarray(field)

    def evaluate(self, vector: np.ndarray, deflection_field: np.ndarray | None = None) -> np.ndarray:
        """Return expected smooth source electrons per pixel for a packed 15-vector."""
        return np.asarray(self._render(vector, *self._args,self._deflection_arg(deflection_field)))

    def jacobian(self, vector: np.ndarray, deflection_field: np.ndarray | None = None) -> np.ndarray:
        """Return d(image electrons)/d(packed parameter), shape (ny, nx, 15)."""
        return np.asarray(self._jac(vector, *self._args,self._deflection_arg(deflection_field)))


@dataclass(frozen=True)
class FitConfig:
    """Fixed deterministic search policy; max_nfev is per starting point."""
    starts: int = 2
    seed: int = 731
    max_nfev: int = 200
    gamma_prior_sigma: float | None = None
    source_components: int = 1

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.starts, self.seed, self.max_nfev)):
            raise ValueError('Fit configuration requires integer values')
        if self.starts < 1 or self.seed < 0 or self.max_nfev < 1:
            raise ValueError('Require starts/max_nfev >= 1 and seed >= 0')
        if self.gamma_prior_sigma is not None and (not np.isfinite(self.gamma_prior_sigma) or self.gamma_prior_sigma <= 0):
            raise ValueError('gamma_prior_sigma must be finite and positive')
        if type(self.source_components) is not int or self.source_components not in (1,2):
            raise ValueError('source_components must be 1 or 2')


@dataclass(frozen=True)
class FitResult:
    """Best finite fit; elapsed_seconds is measured separately from deterministic output."""
    parameters: dict[str, float]
    chi2: float
    prior_penalty: float
    image_unit: str
    dof: int
    converged: bool
    status: int
    message: str
    nfev: int
    njev: int
    best_start: int
    boundary_parameters: tuple[str, ...]
    attempts: tuple[dict, ...]
    config: FitConfig
    parameter_errors: dict[str,float|None]
    covariance_rank: int
    elapsed_seconds: float


def fit_cutout(cutout: Cutout, config: FitConfig | None = None, *,
               initial_vector: np.ndarray | None = None,
               deflection_field: np.ndarray | None = None) -> FitResult:
    """Fit 15 or 18 smooth parameters without truth, returning failures explicitly.

    Use the fixed simulator-default initial profiles, then seed-fixed perturbations.
    The lowest finite chi2 is selected even if its optimizer did not converge;
    convergence and bound contacts are reported rather than hidden.
    """
    cfg = config or FitConfig()
    started = perf_counter()
    renderer = SmoothRenderer(cutout)
    field=renderer._deflection_arg(deflection_field)
    usable = cutout.mask
    data, noise = cutout.image[usable], cutout.noise_map[usable]

    def residual(p):
        image_residual = (renderer.evaluate(p,field)[usable] - data)/noise
        if cfg.gamma_prior_sigma is None:
            return image_residual
        return np.append(image_residual, (p[1]-2.0)/cfg.gamma_prior_sigma)

    def jacobian(p):
        image_jacobian = renderer.jacobian(p,field)[usable]/noise[:, None]
        if cfg.gamma_prior_sigma is None:
            return image_jacobian
        return np.vstack((image_jacobian, np.eye(1, image_jacobian.shape[1], 1)[0]/cfg.gamma_prior_sigma))

    if initial_vector is None:
        initial=parameter_vector(Lens(),Source())
        if cfg.source_components==2:
            initial=two_sersic_parameter_vector(Lens(),Source(),Source(amp=3.,R_sersic=.06,n_sersic=1.))
    else: initial=np.asarray(initial_vector,dtype=float)
    names=TWO_SERSIC_PARAMETER_NAMES if cfg.source_components==2 else PARAMETER_NAMES
    lower,upper=(LOWER_TWO_SERSIC,UPPER_TWO_SERSIC) if cfg.source_components==2 else (LOWER,UPPER)
    if initial.shape!=(len(names),) or not np.isfinite(initial).all(): raise ValueError(f'initial_vector must be finite with {len(names)} values')
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    scales = np.array([.12, .08, .06, .06, .025, .025, .02, .02,
                       .4, .2, .25, .08, .08, .06, .06])
    if cfg.source_components==2: scales=np.concatenate((scales,[.5,.25,.25]))
    attempts, candidates = [], []
    for start in range(cfg.starts):
        p0 = initial.copy() if start == 0 else np.clip(initial+rng.normal(size=len(names))*scales, lower+1e-8, upper-1e-8)
        try:
            result = least_squares(residual, p0, jac=jacobian, bounds=(lower, upper),
                                   method='trf', x_scale='jac', max_nfev=cfg.max_nfev,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8)
            objective = float(result.fun @ result.fun)
            data_residual = (renderer.evaluate(result.x,field)[usable] - data)/noise
            chi2 = float(data_residual @ data_residual)
            penalty = objective - chi2
            if not np.isfinite(objective) or not np.isfinite(result.x).all():
                raise ValueError('Optimizer returned nonfinite values')
            attempts.append({'start': start, 'chi2_data': chi2, 'objective': objective, 'converged': bool(result.success),
                             'status': int(result.status), 'nfev': int(result.nfev),
                             'message': str(result.message)})
            candidates.append((objective, chi2, penalty, start, result))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempts.append({'start': start, 'chi2': None, 'converged': False,
                             'status': -1, 'nfev': None, 'message': str(error)})
    if not candidates:
        raise ValueError('All optimizer starts failed: ' + '; '.join(a['message'] for a in attempts))
    _, chi2, penalty, best, result = min(candidates, key=lambda item: (item[0], item[3]))
    near_bound = (np.minimum(result.x-lower, upper-result.x)/(upper-lower) < 1e-4)
    physical=physical_parameters(result.x)
    errors,rank=linearized_parameter_errors(result.jac,physical,chi2,int(usable.sum())-len(names))
    return FitResult(physical_parameters(result.x), chi2, penalty, cutout.image_unit, int(usable.sum())-len(names),
                     bool(result.success), int(result.status), str(result.message),
                     int(result.nfev), int(result.njev or 0), best,
                     tuple(name for name, contact in zip(names, near_bound, strict=True) if contact),
                     tuple(attempts), cfg, errors, rank, perf_counter()-started)


def save_fit(result: FitResult, path: str | Path) -> None:
    """Write sorted versioned JSON with 10 decimals; omit nondeterministic wall time."""
    model_name='EPL+SHEAR/SERSIC' if result.config.source_components==1 else 'EPL+SHEAR/TWO_SERSIC_SHARED_GEOMETRY'
    payload = {'format': 'skia-fit-v1', 'model': model_name,
               'likelihood': 'fixed-noise-gaussian', 'parameters': result.parameters,
               'parameter_errors_1sigma': result.parameter_errors,
               'parameter_covariance_rank': result.covariance_rank,
               'image_unit': result.image_unit,
               'source_amplitude_unit': f'{result.image_unit}/s/arcsec^2',
               'prior_penalty': result.prior_penalty,
               'chi2': result.chi2, 'prior_penalty': result.prior_penalty,
               'dof': result.dof, 'chi2_per_dof': result.chi2/result.dof,
               'converged': result.converged, 'status': result.status, 'message': result.message,
               'nfev': result.nfev, 'njev': result.njev, 'best_start': result.best_start,
               'boundary_parameters': result.boundary_parameters, 'attempts': result.attempts,
               'optimizer': asdict(result.config), 'precision_decimals': 10}
    Path(path).write_text(_fixed_json(payload)+'\n', encoding='utf-8')


def _fixed_json(value, level: int = 0) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError('Cannot serialize nonfinite fit output')
        return f'{0.0 if abs(value) < .5e-10 else value:.10f}'
    if isinstance(value, dict):
        items = [json.dumps(key)+': '+_fixed_json(value[key], level+1) for key in sorted(value)]
        return '{\n' + ',\n'.join('  '*(level+1)+item for item in items) + '\n'+'  '*level+'}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_fixed_json(x, level) for x in value) + ']'
    return json.dumps(value, allow_nan=False)
