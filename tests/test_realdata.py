import numpy as np
import pytest
from astropy.io import fits

from skia.realdata import estimate_sky_noise, load_q1_fits


def test_outer_ring_mad_mask_is_deterministic():
    rng=np.random.Generator(np.random.PCG64(19))
    image=rng.normal(.003,.002,(64,64))
    image[20:44,20:44]+=.02
    image[0,0]=-1
    image[2,2]=1
    image[1,1]=np.nan
    a,mask=estimate_sky_noise(image)
    b,other=estimate_sky_noise(image)
    assert a==b and a.sigma==pytest.approx(.002,rel=.08)
    assert a.sky==pytest.approx(.003,abs=.0005)
    assert a.rejected_pixels>=2 and not mask[0,0] and not mask[1,1] and mask[2,2]
    np.testing.assert_array_equal(mask,other)


def test_outer_ring_estimator_rejects_bad_inputs():
    with pytest.raises(ValueError): estimate_sky_noise(np.ones((10,10)))
    with pytest.raises(ValueError): estimate_sky_noise(np.full((32,32),np.nan))
    with pytest.raises(ValueError): estimate_sky_noise(np.ones((32,32)),.5)
    sparse=np.full((40,40),np.nan)
    sparse[0,0]=0.0
    with pytest.raises(ValueError,match='Insufficient finite outer-ring'):
        estimate_sky_noise(sparse)


def test_q1_fits_adapter_reads_wcs_and_marks_flux_native(tmp_path):
    rng=np.random.Generator(np.random.PCG64(20))
    data=rng.normal(.001,.003,(80,81))
    data[25:55,25:56]+=.02
    header=fits.Header({'MAGZERO':24.6,'CD1_1':-2.777777777778e-5,'CD1_2':0.,
                        'CD2_1':0.,'CD2_2':2.777777777778e-5})
    path=tmp_path/'sample.fits'
    fits.writeto(path,data,header)
    cut,noise,meta=load_q1_fits(path)
    assert cut.image.shape==data.shape and cut.image_unit=='native_flux'
    assert cut.pixel_scale==pytest.approx(.1,rel=1e-9)
    assert cut.exposure==1 and cut.oversample==2
    assert cut.noise_map[cut.mask].std()<1e-12
    assert meta['magzero']==24.6 and meta['bunit'] is None
    assert meta['psf_fwhm_arcsec_assumed']==.16
    assert noise.fit_pixels==int(cut.mask.sum())


def test_q1_adapter_requires_wcs_and_plausible_scale(tmp_path):
    path=tmp_path/'bad.fits'; fits.writeto(path,np.ones((32,32)))
    with pytest.raises(ValueError,match='CD matrix'): load_q1_fits(path)
    header=fits.Header({'CD1_1':-.01,'CD1_2':0.,'CD2_1':0.,'CD2_2':.01})
    fits.writeto(path,np.ones((32,32)),header,overwrite=True)
    with pytest.raises(ValueError,match='Implausible'): load_q1_fits(path)


def test_q1_adapter_rejects_non_image_primary_and_invalid_psf(tmp_path):
    path=tmp_path/'not-image.fits'
    fits.writeto(path,np.ones(32))
    with pytest.raises(ValueError,match='Expected 2D primary image'):
        load_q1_fits(path)
    data=np.random.default_rng(3).normal(0,.002,(40,40))
    header=fits.Header({'CD1_1':-2.777777777778e-5,'CD1_2':0.,
                        'CD2_1':0.,'CD2_2':2.777777777778e-5})
    fits.writeto(path,data,header,overwrite=True)
    with pytest.raises(ValueError):
        load_q1_fits(path,psf_fwhm=-.16)
