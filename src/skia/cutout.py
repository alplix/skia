"""Truth-independent sky-subtracted imaging inputs for a single lens plane."""
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Cutout:
    """Square image/noise in electrons, angular scale in arcsec/pixel, exposure in s.

    True mask values are used. psf_fine is unit-sum and sampled at
    pixel_scale/oversample; its odd-sized centre is the zero-offset location.
    """
    image: NDArray[np.float64]
    noise_map: NDArray[np.float64]
    mask: NDArray[np.bool_]
    psf_fine: NDArray[np.float64]
    pixel_scale: float
    exposure: float
    oversample: int
    image_unit: str = 'electron'

    def __post_init__(self) -> None:
        for name in ('image', 'noise_map', 'mask', 'psf_fine'):
            try:
                source=np.asarray(getattr(self,name))
            except (TypeError,ValueError,OverflowError) as error:
                raise ValueError(f'{name} must be an array') from error
            if name!='mask' and (not np.issubdtype(source.dtype,np.number)
                    or np.issubdtype(source.dtype,np.complexfloating)):
                raise ValueError(f'{name} must contain real numeric values')
            try:
                array=np.array(source,copy=True) if name=='mask' else np.array(source,dtype=np.float64,copy=True)
            except (TypeError,ValueError,OverflowError) as error:
                raise ValueError(f'{name} has an invalid array dtype') from error
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.image.ndim != 2 or min(self.image.shape) < 2:
            raise ValueError('image must be a two-dimensional cutout')
        if self.noise_map.shape != self.image.shape or self.mask.shape != self.image.shape:
            raise ValueError('image, noise_map and mask must have the same shape')
        if self.mask.dtype != np.bool_ or np.count_nonzero(self.mask) <= 15:
            raise ValueError('A boolean mask with more than 15 usable pixels is required')
        if not np.isfinite(self.image[self.mask]).all():
            raise ValueError('Usable image pixels must be finite')
        if not np.isfinite(self.noise_map[self.mask]).all() or np.any(self.noise_map[self.mask] <= 0):
            raise ValueError('Usable noise standard deviations must be finite and positive')
        p = self.psf_fine
        if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] % 2 != 1:
            raise ValueError('psf_fine must be an odd-sized square array')
        if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-10, rtol=1e-10):
            raise ValueError('psf_fine must be nonnegative, finite and unit-sum')
        if type(self.oversample) is not int or not 1 <= self.oversample <= 8:
            raise ValueError('oversample must be an integer in [1, 8]')
        if not np.isfinite([self.pixel_scale, self.exposure]).all() or min(self.pixel_scale, self.exposure) <= 0:
            raise ValueError('pixel_scale and exposure must be finite and positive')
        if not isinstance(self.image_unit, str) or not self.image_unit:
            raise ValueError('image_unit must be a nonempty string')


def load_cutout(path: str | Path) -> Cutout:
    """Read skia-cutout-v1 NPZ without consulting model, truth or source parameters."""
    with np.load(path, allow_pickle=False) as bundle:
        required = {'image', 'noise_map', 'mask', 'psf_fine', 'observation_json'}
        missing = required - set(bundle.files)
        if missing:
            raise ValueError('Missing cutout fields: ' + ', '.join(sorted(missing)))
        meta = json.loads(str(bundle['observation_json']))
        if not isinstance(meta, dict) or meta.get('format') != 'skia-cutout-v1':
            raise ValueError('Expected skia-cutout-v1 observation metadata')
        if meta.get('image_units') not in ('electron', 'native_flux'):
            raise ValueError('Unsupported image units; expected electron or native_flux')
        try:
            return Cutout(bundle['image'], bundle['noise_map'], bundle['mask'], bundle['psf_fine'],
                          meta['pixel_scale'], meta['exposure'], meta['oversample'], meta['image_units'])
        except KeyError as error:
            raise ValueError(f'Missing observation metadata: {error.args[0]}') from error


def save_cutout(cutout: Cutout, path: str | Path | BinaryIO) -> None:
    """Write canonical `skia-cutout-v1` NPZ arrays without simulator truth.

    Image and noise units follow ``cutout.image_unit``; pixel scale is arcsec
    per pixel, exposure seconds, and ``psf_fine`` is unit-sum at the declared
    oversampled pixel scale.
    """
    metadata={'format':'skia-cutout-v1','image_units':cutout.image_unit,
              'pixel_scale':float(cutout.pixel_scale),'exposure':float(cutout.exposure),
              'oversample':int(cutout.oversample)}
    payload=BytesIO()
    np.savez_compressed(payload,image=cutout.image,noise_map=cutout.noise_map,mask=cutout.mask,
                        psf_fine=cutout.psf_fine,
                        observation_json=json.dumps(metadata,sort_keys=True,separators=(',',':')))
    if hasattr(path,'write'):
        path.write(payload.getvalue())
    else:
        Path(path).write_bytes(payload.getvalue())
