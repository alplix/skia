import importlib.util
from pathlib import Path

BENCH=Path(__file__).parent/'bench'
spec=importlib.util.spec_from_file_location('lenstronomy_crosscheck_bench',BENCH/'crosscheck_lenstronomy.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
crosscheck_one=module.crosscheck_one
fit_module_spec=importlib.util.spec_from_file_location('lenstronomy_fit_crosscheck_bench',BENCH/'lenstronomy_fit_crosscheck.py')
fit_module=importlib.util.module_from_spec(fit_module_spec)
fit_module_spec.loader.exec_module(fit_module)
from skia.simulator import Lens, Observation, Source
from skia.fitter import FitConfig


def test_lenstronomy_mass_light_and_image_match_skia_forward_model():
    result=crosscheck_one(Lens(),Source(),Observation(npix=20,oversample=1),seed=71)
    assert result['deflection_max_abs_arcsec']<1e-12
    assert result['source_brightness_relative_rms']<1e-12
    assert result['noiseless_image_relative_rms']<1e-12


def test_lenstronomy_independent_fit_matches_skia_fit_on_tiny_simulation():
    result=fit_module.crosscheck_fit(Lens(),Source(),Observation(npix=16,oversample=1),
        noise_seed=71,config=FitConfig(starts=1,seed=731,max_nfev=20))
    differences=result['difference_herculens_minus_lenstronomy']
    assert abs(differences['chi2'])<1e-5
    assert abs(differences['theta_E_arcsec'])<1e-5
    assert abs(differences['gamma'])<1e-5
