"""Single-plane angular lens simulation in electrons, with independent detector noise.

x increases with column and y with row (no sky WCS); origin is the image centre.
Ellipticity is (1-q)/(1+q) * (cos(2 phi), sin(2 phi)). Angles are arcsec.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve
from herculens.MassModel.Profiles.epl import EPL
from herculens.MassModel.Profiles.shear import Shear
from herculens.LightModel.Profiles.sersic import Sersic
from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe

Array = NDArray[np.float64]


def _finite(values: tuple[float, ...]) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError("All numeric parameters must be finite")


def _ellipticity(e1: float, e2: float) -> None:
    if np.hypot(e1, e2) >= 0.8:
        raise ValueError("Ellipticity modulus must be below 0.8")


@dataclass(frozen=True)
class Lens:
    """SPLE/EPL: theta_E in arcsec, 3D density slope gamma, dimensionless shear."""
    theta_E: float = 1.0
    gamma: float = 2.0
    e1: float = 0.12
    e2: float = 0.04
    center_x: float = 0.0
    center_y: float = 0.0
    gamma1: float = 0.02
    gamma2: float = -0.01

    def __post_init__(self) -> None:
        _finite(tuple(asdict(self).values()))
        _ellipticity(self.e1, self.e2)
        if self.theta_E < 0 or not 1 < self.gamma < 3:
            raise ValueError("Require theta_E >= 0 and 1 < gamma < 3")
        if np.hypot(self.gamma1, self.gamma2) >= 1:
            raise ValueError("Shear modulus must be below one")


@dataclass(frozen=True)
class Source:
    """Sersic: amp is e-/s/arcsec² at the major-axis half-light radius R_sersic."""
    amp: float = 10.0
    R_sersic: float = 0.18
    n_sersic: float = 1.2
    e1: float = 0.08
    e2: float = -0.03
    center_x: float = 0.08
    center_y: float = 0.04

    def __post_init__(self) -> None:
        _finite(tuple(asdict(self).values()))
        _ellipticity(self.e1, self.e2)
        if self.amp < 0 or self.R_sersic <= 0 or not 0.5 <= self.n_sersic <= 8:
            raise ValueError("Require amp >= 0, R_sersic > 0 and 0.5 <= n_sersic <= 8")


@dataclass(frozen=True)
class Subhalo:
    """Spherical TNFW or cored pseudo-Jaffe; angular radii/deflections in arcsec.

    TNFW uses Rs, alpha_Rs (untruncated NFW normalization), r_trunc.
    PJAFFE uses sigma0 (central convergence), Ra (core), Rs (outer radius).
    No cosmology or mass-in-Msun conversion is assumed.
    """
    profile: Literal["TNFW", "PJAFFE"] = "TNFW"
    center_x: float = 0.8
    center_y: float = 0.2
    Rs: float = 0.2
    alpha_Rs: float = 0.015
    r_trunc: float = 0.8
    sigma0: float = 1.0
    Ra: float = 0.01

    def __post_init__(self) -> None:
        _finite(tuple(v for k, v in asdict(self).items() if k != "profile"))
        if self.profile not in ("TNFW", "PJAFFE"):
            raise ValueError("Supported subhaloes: TNFW, PJAFFE")
        if min(self.Rs, self.r_trunc, self.Ra) <= 0 or self.Ra >= self.Rs:
            raise ValueError("Require 0 < Ra < Rs and r_trunc > 0")
        if min(self.alpha_Rs, self.sigma0) < 0:
            raise ValueError("Subhalo normalization must be nonnegative")


@dataclass(frozen=True)
class Observation:
    """Square detector: arcsec/pixel, seconds, sky e-/s/pixel, read noise e-/pixel."""
    npix: int = 100
    pixel_scale: float = 0.1
    exposure: float = 565.0
    sky_rate: float = 0.1
    read_noise: float = 4.5
    psf_fwhm: float = 0.18
    oversample: int = 2

    def __post_init__(self) -> None:
        _finite(tuple(asdict(self).values()))
        if type(self.npix) is not int or type(self.oversample) is not int:
            raise ValueError("npix and oversample must be integers")
        if self.npix < 2 or self.oversample < 1:
            raise ValueError("Require npix >= 2 and oversample >= 1")
        if min(self.pixel_scale, self.exposure, self.psf_fwhm) <= 0:
            raise ValueError("Pixel scale, exposure and PSF FWHM must be positive")
        if min(self.sky_rate, self.read_noise) < 0:
            raise ValueError("Noise parameters must be nonnegative")


@dataclass(frozen=True)
class Simulation:
    """Arrays in electrons (noise_map is expected per-pixel standard deviation).

    image is sky-subtracted, model is noiseless source electrons after PSF,
    psf is a unit-sum detector-grid Gaussian kernel, mask=True means usable.
    """
    image: Array
    model: Array
    noise_map: Array
    psf: Array
    mask: NDArray[np.bool_]
    truth: dict = field(default_factory=dict)


def deflection(x: Array, y: Array, lens: Lens, subhalo: Subhalo | None = None) -> tuple[Array, Array]:
    """Return alpha_x, alpha_y in arcsec at image-plane x,y in arcsec."""
    kwargs = asdict(lens)
    g1, g2 = kwargs.pop("gamma1"), kwargs.pop("gamma2")
    if lens.theta_E == 0:
        ax, ay = np.zeros_like(x), np.zeros_like(y)
    else:
        ax, ay = EPL().derivatives(x, y, **kwargs)
    sx, sy = Shear().derivatives(x, y, gamma1=g1, gamma2=g2)
    ax, ay = np.asarray(ax + sx), np.asarray(ay + sy)
    if subhalo is not None:
        h = subhalo
        if h.profile == "TNFW":
            dx, dy = TNFW().derivatives(x, y, Rs=h.Rs, alpha_Rs=h.alpha_Rs,
                                        r_trunc=h.r_trunc, center_x=h.center_x, center_y=h.center_y)
        else:
            dx, dy = PseudoJaffe().derivatives(x, y, sigma0=h.sigma0, Ra=h.Ra,
                                              Rs=h.Rs, center_x=h.center_x, center_y=h.center_y)
        ax, ay = ax + dx, ay + dy
    return ax, ay


def gaussian_psf(pixel_scale: float, fwhm: float) -> Array:
    """Unit-sum Gaussian sampled at pixel centres; both arguments in arcsec."""
    _finite((pixel_scale, fwhm))
    if min(pixel_scale, fwhm) <= 0:
        raise ValueError("PSF scale and FWHM must be positive")
    sigma = fwhm / np.sqrt(8 * np.log(2)) / pixel_scale
    radius = max(1, int(np.ceil(5 * sigma)))
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def simulate(lens: Lens | None = None, source: Source | None = None,
             observation: Observation | None = None, subhalo: Subhalo | None = None,
             *, seed: int = 0) -> Simulation:
    """Render one single-plane lens; seeded Poisson counts plus Gaussian read noise.

    Ray trace at subpixel centres; convolve at subpixel resolution before pixel
    integration. Pad by the PSF support to include light outside the cutout.
    This is a VIS-like synthetic observation, not an instrument calibration.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    lens, source, obs = lens or Lens(), source or Source(), observation or Observation()
    os = obs.oversample
    step = obs.pixel_scale / os
    fine_psf = gaussian_psf(step, obs.psf_fwhm)
    pad = (fine_psf.shape[0] // 2 + os - 1) // os
    n = (obs.npix + 2 * pad) * os
    coord = (np.arange(n, dtype=float) - (n - 1) / 2) * step
    x, y = np.meshgrid(coord, coord)
    ax, ay = deflection(x, y, lens, subhalo)
    light = np.asarray(Sersic(radius_major_axis=True).function(x-ax, y-ay, **asdict(source)))
    blurred = fftconvolve(light, fine_psf, mode="same")
    start, stop = pad * os, (pad + obs.npix) * os
    integrated = blurred[start:stop, start:stop].reshape(obs.npix, os, obs.npix, os).mean(axis=(1, 3))
    model = np.maximum(integrated * obs.pixel_scale**2 * obs.exposure, 0)
    if not np.all(np.isfinite(model)):
        raise ValueError("Nonfinite rendered image")
    sky = obs.sky_rate * obs.exposure
    expectation = model + sky
    rng = np.random.Generator(np.random.PCG64(seed))
    image = rng.poisson(expectation).astype(float) + rng.normal(0, obs.read_noise, model.shape) - sky
    noise_map = np.sqrt(expectation + obs.read_noise**2)
    truth = {"format": "skia-simulation-v1", "seed": seed, "lens": asdict(lens),
             "source": asdict(source), "observation": asdict(obs),
             "subhalo": asdict(subhalo) if subhalo else None}
    return Simulation(image, model, noise_map, gaussian_psf(obs.pixel_scale, obs.psf_fwhm),
                      np.ones(model.shape, dtype=bool), truth)


def save_simulation(result: Simulation, path: str | Path) -> None:
    """Save arrays and versioned JSON truth in a pickle-free synthetic NPZ bundle.

    This intermediate simulation format is not a BOINC result format.
    """
    obs = Observation(**result.truth["observation"])
    metadata = {"format": "skia-cutout-v1", "image_units": "electron",
                "pixel_scale": obs.pixel_scale, "exposure": obs.exposure,
                "oversample": obs.oversample}
    fine_psf = gaussian_psf(obs.pixel_scale/obs.oversample, obs.psf_fwhm)
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, image=result.image, model=result.model,
                            noise_map=result.noise_map, psf=result.psf, mask=result.mask,
                            psf_fine=fine_psf, observation_json=json.dumps(metadata, sort_keys=True),
                            truth_json=json.dumps(result.truth, sort_keys=True, allow_nan=False))
