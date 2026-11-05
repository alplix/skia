from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from skia.cli import main
from skia.cutout import Cutout, load_cutout
from skia.fitter import (FitConfig, SmoothRenderer, fit_cutout, parameter_vector,
                         physical_parameters, save_fit, two_sersic_parameter_vector,
                         linearized_parameter_errors)
from skia.simulator import Lens, Source, Observation, simulate, save_simulation


@pytest.fixture(scope='module')
def sample(tmp_path_factory):
    directory = tmp_path_factory.mktemp('fitter')
    lens = Lens(theta_E=.96, gamma=2.06, e2=-.025)
    source = Source(amp=12, center_x=.06)
    sim = simulate(lens, source, Observation(npix=32), seed=123)
    path = directory / 'lens.npz'
    save_simulation(sim, path)
    return lens, source, sim, path, load_cutout(path)


def test_parameter_roundtrip_and_validation(sample):
    lens, source, *_ = sample
    p = parameter_vector(lens, source)
    values = physical_parameters(p)
    assert values['lens.gamma'] == lens.gamma
    assert values['source.amp'] == pytest.approx(source.amp)
    assert values['source.R_sersic'] == pytest.approx(source.R_sersic)
    with pytest.raises(ValueError):
        physical_parameters(np.zeros(14))
    with pytest.raises(ValueError):
        parameter_vector(lens, Source(amp=0))


def test_linearized_errors_known_linear_jacobian_and_rank_loss():
    names={'a':2.,'source.amp':10.}
    errors,rank=linearized_parameter_errors(np.eye(2),names,chi2=2.,dof=2)
    assert rank==2 and errors=={'a':1.,'source.amp':10.}
    errors,rank=linearized_parameter_errors(np.array([[1.,1.],[2.,2.]]),names,2.,2)
    assert rank==1 and errors=={'a':None,'source.amp':None}


def test_renderer_matches_simulator_and_jacobian(sample):
    lens, source, sim, _, cutout = sample
    renderer = SmoothRenderer(cutout)
    p = parameter_vector(lens, source)
    np.testing.assert_allclose(renderer.evaluate(p), sim.model, atol=2e-10, rtol=1e-10)
    jac = renderer.jacobian(p)
    assert jac.shape == (*sim.model.shape, 15)
    assert np.isfinite(jac).all()
    # Central finite differences for every parameter, including log flux/radius.
    for i in range(15):
        d = np.zeros(15)
        d[i] = 1e-5
        numeric = (renderer.evaluate(p+d)-renderer.evaluate(p-d))/(2e-5)
        assert np.linalg.norm(jac[:, :, i]-numeric)/np.linalg.norm(numeric) < 2e-5


def test_two_sersic_shared_geometry_render_and_fit(sample):
    lens,source,_,_,cutout=sample
    source2=Source(amp=3.,R_sersic=.06,n_sersic=2.2,e1=source.e1,e2=source.e2,
                   center_x=source.center_x,center_y=source.center_y)
    vector=two_sersic_parameter_vector(lens,source,source2)
    physical=physical_parameters(vector)
    assert physical['source2.amp']==pytest.approx(3.)
    rendered=SmoothRenderer(cutout).evaluate(vector)
    synthetic=replace(cutout,image=rendered)
    fit=fit_cutout(synthetic,FitConfig(starts=1,max_nfev=30,source_components=2),initial_vector=vector)
    assert fit.converged and fit.chi2<1e-10 and fit.dof==int(cutout.mask.sum())-18
    assert fit.parameters['source2.R_sersic']==pytest.approx(.06)
    from skia.scanner import vector_from_fit
    from dataclasses import asdict
    np.testing.assert_allclose(vector_from_fit({'format':'skia-fit-v1','parameters':fit.parameters,
                                                'optimizer':asdict(fit.config)}),vector,atol=1e-10)


def test_load_ignores_all_truth_and_model(sample, tmp_path):
    *_, path, cutout = sample
    with np.load(path, allow_pickle=False) as saved:
        fields = {key: saved[key] for key in saved.files if key not in ('model', 'truth_json')}
    fields['truth_json'] = 'not even valid json'
    fields['model'] = np.full(cutout.image.shape, np.nan)
    edited = tmp_path/'blind.npz'
    np.savez(edited, **fields)
    blind = load_cutout(edited)
    a = fit_cutout(cutout, FitConfig(starts=1))
    b = fit_cutout(blind, FitConfig(starts=1))
    assert a.parameters == b.parameters and a.chi2 == b.chi2


def test_noiseless_recovery_all_free_parameters(sample):
    lens, source, sim, _, cutout = sample
    noiseless = replace(cutout, image=sim.model)
    fit = fit_cutout(noiseless, FitConfig(starts=1))
    assert fit.converged and fit.chi2 < 1e-10
    truth = physical_parameters(parameter_vector(lens, source))
    for name, value in truth.items():
        assert fit.parameters[name] == pytest.approx(value, abs=2e-5)


def test_masked_bad_pixels_do_not_enter_fit(sample):
    *_, cutout = sample
    mask = cutout.mask.copy()
    mask[0, :] = False
    image = cutout.image.copy()
    noise = cutout.noise_map.copy()
    image[0, :] = np.nan
    noise[0, :] = -1
    a = fit_cutout(replace(cutout, mask=mask), FitConfig(starts=1))
    b = fit_cutout(replace(cutout, image=image, noise_map=noise, mask=mask), FitConfig(starts=1))
    assert a.parameters == b.parameters and a.dof == mask.sum()-15


def test_exhaustion_is_reported_and_fixed_json_ignores_time(sample, tmp_path):
    *_, cutout = sample
    fit = fit_cutout(cutout, FitConfig(starts=1, max_nfev=1))
    assert not fit.converged and fit.status == 0
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    save_fit(fit, a)
    save_fit(replace(fit, elapsed_seconds=999), b)
    assert a.read_bytes() == b.read_bytes()
    payload = json.loads(a.read_text())
    assert payload['format'] == 'skia-fit-v1'
    assert payload['converged'] is False
    assert 'elapsed_seconds' not in payload
    assert 'parameter_errors_1sigma' in payload
    assert '0.0000000000' in a.read_text()


@pytest.mark.parametrize('changes', [
    {'pixel_scale': -1}, {'exposure': float('inf')}, {'oversample': 0},
    {'oversample': 1.5}, {'mask': np.zeros((32, 32), dtype=bool)},
    {'mask': np.ones((32, 32), dtype=int)}, {'noise_map': np.zeros((32, 32))},
    {'psf_fine': np.zeros((3, 3))}, {'psf_fine': np.ones((2, 2))/4},
    {'noise_map': np.ones((2, 2))}, {'image': np.ones((3, 5))},
])
def test_bad_cutout_rejected(sample, changes):
    with pytest.raises(ValueError):
        replace(sample[-1], **changes)


@pytest.mark.parametrize('kwargs', [{'starts': 0}, {'seed': -1}, {'max_nfev': 0}, {'starts': 1.5}])
def test_bad_fit_config(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)


def test_loader_rejects_legacy_and_bad_units(tmp_path, sample):
    path = tmp_path/'bad.npz'
    np.savez(path, image=np.ones((10, 10)))
    with pytest.raises(ValueError, match='Missing cutout fields'):
        load_cutout(path)
    with np.load(sample[-2], allow_pickle=False) as saved:
        fields = {k: saved[k] for k in saved.files}
    meta = json.loads(str(fields['observation_json']))
    meta['image_units'] = 'adu'
    fields['observation_json'] = json.dumps(meta)
    np.savez(path, **fields)
    with pytest.raises(ValueError, match='Unsupported image units'):
        load_cutout(path)


def test_cli_success_failure_and_timing(sample, tmp_path, capsys):
    path = sample[-2]
    output = tmp_path/'fit.json'
    assert main([str(path), '-o', str(output), '--starts', '1']) == 0
    text = capsys.readouterr().err
    assert 'elapsed_seconds=' in text and 'converged=True' in text
    assert json.loads(output.read_text())['converged']
    assert main([str(path), '-o', str(output), '--starts', '1', '--gamma-prior-sigma', '.1']) == 0
    assert json.loads(output.read_text())['optimizer']['gamma_prior_sigma'] == .1
    assert main([str(path), '-o', str(output), '--max-nfev', '1', '--starts', '1']) == 3
    assert not json.loads(output.read_text())['converged']
    assert main([str(tmp_path/'missing.npz'), '-o', str(output)]) == 2


def test_installed_cli_is_deterministic_across_processes(sample, tmp_path):
    executable = Path(sys.executable).parent/'skia-fit'
    paths = [tmp_path/'a.json', tmp_path/'b.json']
    for path in paths:
        result = subprocess.run([str(executable), str(sample[-2]), '-o', str(path), '--starts', '1'],
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        assert 'elapsed_seconds=' in result.stderr
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_all_starts_failure_is_explicit(sample, monkeypatch):
    import skia.fitter as fitter

    def fail(*args, **kwargs):
        raise ValueError('deliberate numerical error')

    monkeypatch.setattr(fitter, 'least_squares', fail)
    with pytest.raises(ValueError, match='All optimizer starts failed'):
        fit_cutout(sample[-1], FitConfig(starts=2))


def test_output_rejects_nonfinite_values(sample, tmp_path):
    fit = fit_cutout(sample[-1], FitConfig(starts=1, max_nfev=1))
    with pytest.raises(ValueError, match='nonfinite'):
        save_fit(replace(fit, chi2=float('nan')), tmp_path/'bad.json')


def test_gamma_prior_changes_optimizer_but_reports_data_chi_square(sample):
    lens, source, _, _, cutout = sample
    free = fit_cutout(cutout, FitConfig(starts=1))
    prior = fit_cutout(cutout, FitConfig(starts=1, gamma_prior_sigma=.1))
    assert abs(prior.parameters['lens.gamma']-2) <= abs(free.parameters['lens.gamma']-2)
    assert prior.prior_penalty == pytest.approx(((prior.parameters['lens.gamma']-2)/.1)**2)
    assert prior.chi2 == pytest.approx(free.chi2, abs=20)
    assert prior.attempts[0]['objective'] == pytest.approx(prior.chi2+prior.prior_penalty)
