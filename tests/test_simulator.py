from dataclasses import replace
import json
import numpy as np
import pytest
from skia.simulator import (Lens, Source, Subhalo, Observation, Simulation,
                            deflection, gaussian_psf, simulate, save_simulation)


@pytest.mark.parametrize('cls,kwargs', [
    (Lens, {'gamma': 3}), (Lens, {'theta_E': -1}), (Lens, {'e1': 0.9}),
    (Lens, {'gamma1': 1}), (Source, {'amp': -1}), (Source, {'n_sersic': 0}),
    (Source, {'R_sersic': 0}), (Source, {'e2': 0.9}),
    (Subhalo, {'profile': 'NFW'}), (Subhalo, {'Ra': 1}),
    (Subhalo, {'alpha_Rs': -1}), (Observation, {'npix': 2.5}),
    (Observation, {'oversample': 0}), (Observation, {'read_noise': -1}),
    (Observation, {'pixel_scale': 0}), (Observation, {'exposure': float('nan')}),
])
def test_reject_invalid_config(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_circular_isothermal_deflection_and_shear():
    x, y = np.array([0.3, -0.4, 0.9]), np.array([0.4, 0.3, -0.2])
    lens = Lens(e1=0, e2=0, theta_E=1.3, gamma=2, gamma1=0.03, gamma2=-0.02)
    ax, ay = deflection(x, y, lens)
    r = np.hypot(x, y)
    np.testing.assert_allclose(ax, lens.theta_E*x/r + lens.gamma1*x + lens.gamma2*y, rtol=1e-9)
    np.testing.assert_allclose(ay, lens.theta_E*y/r + lens.gamma2*x - lens.gamma1*y, rtol=1e-9)


def test_no_lens_deflection():
    x, y = np.array([0.2, 1.0]), np.array([0.1, -0.5])
    ax, ay = deflection(x, y, Lens(theta_E=0, gamma1=0, gamma2=0))
    np.testing.assert_array_equal(ax, 0*x)
    np.testing.assert_array_equal(ay, 0*y)


@pytest.mark.parametrize('profile', ['TNFW', 'PJAFFE'])
def test_subhalo_radial_symmetry_and_zero_strength(profile):
    lens = Lens(theta_E=0, gamma1=0, gamma2=0)
    halo = Subhalo(profile=profile, center_x=0, center_y=0)
    x, y = np.array([0.13, -0.13]), np.array([0.21, -0.21])
    ax, ay = deflection(x, y, lens, halo)
    assert np.all(np.isfinite(ax)) and ax[0] > 0
    np.testing.assert_allclose(ax, -ax[::-1])
    np.testing.assert_allclose(ay, -ay[::-1])
    np.testing.assert_allclose(ax*y, ay*x)
    ax, ay = deflection(x, y, lens, replace(halo, alpha_Rs=0, sigma0=0))
    np.testing.assert_array_equal(ax, x*0)
    np.testing.assert_array_equal(ay, y*0)


def test_psf_normalization_symmetry_and_width():
    p = gaussian_psf(0.02, 0.18)
    assert np.all(p >= 0)
    assert p.sum() == pytest.approx(1)
    np.testing.assert_array_equal(p, p[::-1, ::-1])
    c = np.arange(p.shape[0]) - p.shape[0]//2
    width = np.sqrt(np.sum(p * c[None, :]**2))*0.02
    assert width == pytest.approx(0.18 / np.sqrt(8*np.log(2)), rel=1e-4)
    with pytest.raises(ValueError):
        gaussian_psf(0, 1)


@pytest.mark.parametrize('profile', [None, 'TNFW', 'PJAFFE'])
def test_seed_reproducibility_and_output(profile):
    obs = Observation(npix=24)
    halo = Subhalo(profile=profile) if profile else None
    a = simulate(observation=obs, subhalo=halo, seed=42)
    b = simulate(observation=obs, subhalo=halo, seed=42)
    c = simulate(observation=obs, subhalo=halo, seed=43)
    assert isinstance(a, Simulation)
    assert a.image.shape == (24, 24)
    for key in ('image', 'model', 'noise_map', 'psf', 'mask'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        assert np.isfinite(getattr(a, key)).all()
    assert not np.array_equal(a.image, c.image)
    np.testing.assert_array_equal(a.model, c.model)
    assert a.truth['seed'] == 42 and a.mask.all()
    np.testing.assert_allclose(a.noise_map**2, a.model + obs.sky_rate*obs.exposure + obs.read_noise**2)
    if halo:
        smooth = simulate(observation=obs, seed=42)
        assert np.max(np.abs(a.model - smooth.model)) > 0


def test_noise_mean_and_variance_in_blank_field():
    obs = Observation(npix=256)
    a = simulate(source=Source(amp=0), observation=obs, seed=12)
    variance = obs.sky_rate*obs.exposure + obs.read_noise**2
    assert abs(a.image.mean()) < 5*np.sqrt(variance/a.image.size)
    # 5-sigma variance sampling bound, with margin for the Poisson fourth moment.
    assert abs(a.image.var()/variance - 1) < 5*np.sqrt(2/a.image.size) + 0.005
    np.testing.assert_array_equal(a.model, 0*a.model)


def test_unlensed_flux_exposure_and_pixel_integration():
    lens = Lens(theta_E=0, gamma1=0, gamma2=0)
    source = Source(n_sersic=0.5, R_sersic=0.25, e1=0, e2=0, center_x=0, center_y=0)
    obs = Observation(npix=48, oversample=3)
    a = simulate(lens, source, obs)
    b = simulate(lens, source, replace(obs, exposure=2*obs.exposure))
    np.testing.assert_allclose(b.model, 2*a.model)
    # n=0.5 is a Gaussian; compare the analytic integral using upstream's b_n approximation.
    bn = 1.9992*source.n_sersic - 0.3271
    total = source.amp*np.exp(bn)*np.pi*source.R_sersic**2/bn*obs.exposure
    assert a.model.sum() == pytest.approx(total, rel=2e-5)
    fine = simulate(lens, source, replace(obs, pixel_scale=0.05, npix=96))
    assert fine.model.sum() == pytest.approx(a.model.sum(), rel=2e-5)


def test_pickle_free_save_preserves_truth(tmp_path):
    a = simulate(observation=Observation(npix=12), subhalo=Subhalo(profile='PJAFFE'), seed=5)
    path = tmp_path / 'cutout.npz'
    save_simulation(a, path)
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved['truth_json'])) == a.truth
        for key in ('image', 'model', 'noise_map', 'psf', 'mask'):
            np.testing.assert_array_equal(saved[key], getattr(a, key))


@pytest.mark.parametrize('seed', [-1, 1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        simulate(seed=seed)


def test_default_field_and_nonisothermal_odd_grid():
    default = simulate(seed=9)
    assert default.model.shape == (100, 100)
    assert default.truth['observation']['pixel_scale'] == 0.1
    odd = simulate(lens=Lens(gamma=2.2), observation=Observation(npix=25, oversample=1), seed=9)
    assert np.isfinite(odd.model).all() and odd.model.sum() > 0


def test_reproducibility_in_separate_processes(tmp_path):
    import os
    import subprocess
    import sys
    code = (
        'from skia.simulator import Observation, simulate, save_simulation; '
        'import sys; '
        'save_simulation(simulate(observation=Observation(npix=12), seed=28), sys.argv[1])'
    )
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true')
    paths = [tmp_path/'a.npz', tmp_path/'b.npz']
    for path in paths:
        subprocess.run([sys.executable, '-c', code, str(path)], env=env, check=True, capture_output=True, timeout=60)
    with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
