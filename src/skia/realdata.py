"""Reproducible adapter for the delivered Q1 background-subtracted VIS FITS cutouts."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from skia.cutout import Cutout
from skia.simulator import gaussian_psf


@dataclass(frozen=True)
class NoiseEstimate:
    """Robust sky-only estimate in native flux units per image pixel."""
    sky: float
    sigma: float
    ring_pixels: int
    fit_pixels: int
    rejected_pixels: int


def estimate_sky_noise(image: np.ndarray, ring_fraction: float = .15) -> tuple[NoiseEstimate, np.ndarray]:
    """Estimate median sky and MAD sigma from a fixed outer border; mask >8 sigma.

    No gain/exposure calibration exists in Q1 cutouts, so this deliberately uses
    a sky-only, spatially constant noise estimate in native image units. Keep
    positive source flux; mask only extreme negative pixels as well as NaNs.
    """
    values = np.asarray(image, dtype=float)
    if values.ndim != 2 or min(values.shape) < 20:
        raise ValueError('Expected a two-dimensional FITS image with at least 20 pixels per side')
    if not 0 < ring_fraction < .5:
        raise ValueError('ring_fraction must be in (0, 0.5)')
    height, width = values.shape
    ey = max(2, int(np.ceil(height*ring_fraction)))
    ex = max(2, int(np.ceil(width*ring_fraction)))
    yy, xx = np.indices(values.shape)
    ring = (yy < ey) | (yy >= height-ey) | (xx < ex) | (xx >= width-ex)
    usable_ring = ring & np.isfinite(values)
    ring_values = values[usable_ring]
    if ring_values.size < 100:
        raise ValueError('Insufficient finite outer-ring pixels for noise estimate')
    sky = float(np.median(ring_values))
    sigma = float(1.482602218505602*np.median(np.abs(ring_values-sky)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Outer-ring MAD noise estimate is zero or nonfinite')
    mask = np.isfinite(values) & (values >= sky-8*sigma)
    if mask.sum() <= 15:
        raise ValueError('Fewer than 16 usable pixels after the fixed outlier rule')
    return NoiseEstimate(sky, sigma, int(ring_values.size), int(mask.sum()), int(values.size-mask.sum())), mask


def load_q1_fits(path: str | Path, psf_fwhm: float = .16) -> tuple[Cutout, NoiseEstimate, dict]:
    """Read primary image, subtract robust sky and attach the stated Gaussian PSF.

    Pixel scale comes from the FITS celestial CD matrix; image values remain in
    native_flux units since the mosaic has no BUNIT/exposure or gain calibration.
    """
    with fits.open(path, memmap=False) as hdus:
        if hdus[0].data is None or hdus[0].data.ndim != 2:
            raise ValueError('Expected 2D primary image')
        data = np.array(hdus[0].data, dtype=float)
        header = hdus[0].header.copy()
    if 'CD1_1' not in header or 'CD1_2' not in header or 'CD2_1' not in header or 'CD2_2' not in header:
        raise ValueError('FITS header must provide the complete celestial CD matrix')
    cd = np.array([[header['CD1_1'], header['CD1_2']], [header['CD2_1'], header['CD2_2']]], dtype=float)
    scale = float(np.sqrt(abs(np.linalg.det(cd)))*3600)
    if not np.isfinite(scale) or not .05 <= scale <= .2:
        raise ValueError(f'Implausible pixel scale from FITS WCS: {scale}')
    estimate, mask = estimate_sky_noise(data)
    sky_subtracted = np.where(np.isfinite(data), data-estimate.sky, 0.0)
    noise = np.full(data.shape, estimate.sigma, dtype=float)
    kernel = gaussian_psf(scale/2, psf_fwhm)
    cutout = Cutout(sky_subtracted, noise, mask, kernel, scale, 1.0, 2, 'native_flux')
    metadata = {'file': Path(path).name, 'pixel_scale_arcsec': scale, 'psf_model': 'circular Gaussian',
                'psf_fwhm_arcsec_assumed': psf_fwhm, 'psf_fwhm_provenance': '0.16 arcsec approximation in docs/DATA.md; not verified per cutout',
                'noise_model': 'constant sky-only MAD from outer 15% border; gain/exposure unknown',
                'mask_rule': 'finite and not more than 8 estimated sky sigmas below outer-border median; preserve positive source flux',
                'sky_native_flux_per_pixel': estimate.sky, 'sigma_native_flux_per_pixel': estimate.sigma,
                'ring_pixels': estimate.ring_pixels, 'usable_pixels': estimate.fit_pixels,
                'rejected_pixels': estimate.rejected_pixels, 'magzero': header.get('MAGZERO'),
                'bunit': header.get('BUNIT'), 'exposure_header': header.get('EXPTIME')}
    return cutout, estimate, metadata
