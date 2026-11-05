from io import BytesIO

import pytest
from skia.cutout import Cutout,load_cutout,save_cutout
from skia.simulator import Observation,save_simulation,simulate

import numpy as np


def test_save_cutout_round_trip_is_deterministic_and_truth_free(tmp_path):
    source=tmp_path/'simulation.npz'
    save_simulation(simulate(observation=Observation(npix=20),seed=41),source)
    cutout=load_cutout(source)
    first=tmp_path/'first.npz';second=tmp_path/'second.npz'
    save_cutout(cutout,first);save_cutout(cutout,second)
    assert first.read_bytes()==second.read_bytes()
    restored=load_cutout(first)
    np.testing.assert_array_equal(restored.image,cutout.image)
    np.testing.assert_array_equal(restored.noise_map,cutout.noise_map)
    np.testing.assert_array_equal(restored.mask,cutout.mask)
    np.testing.assert_array_equal(restored.psf_fine,cutout.psf_fine)
    assert restored.image_unit==cutout.image_unit


def test_save_cutout_accepts_binary_file_object():
    simulation=simulate(observation=Observation(npix=20,oversample=1),seed=42)
    from skia.cutout import Cutout
    # Save simulation arrays through the public adapter after constructing its worker view.
    worker=Cutout(simulation.image,simulation.noise_map,simulation.mask,simulation.psf,.1,565.,1)
    stream=BytesIO()
    save_cutout(worker,stream)
    stream.seek(0)
    with np.load(stream,allow_pickle=False) as archive:
        assert set(archive.files)=={'image','noise_map','mask','psf_fine','observation_json'}
        assert 'truth_json' not in archive.files


def test_load_cutout_rejects_missing_arrays_and_bad_metadata(tmp_path):
    missing=tmp_path/'missing.npz'
    np.savez(missing,image=np.zeros((20,20)))
    with pytest.raises(ValueError,match='Missing cutout fields'):
        load_cutout(missing)
    bad=tmp_path/'bad-version.npz'
    np.savez(bad,image=np.zeros((20,20)),noise_map=np.ones((20,20)),mask=np.ones((20,20),dtype=bool),
        psf_fine=np.ones((1,1)),observation_json='{"format":"skia-cutout-v0"}')
    with pytest.raises(ValueError,match='Expected skia-cutout-v1'):
        load_cutout(bad)


@pytest.mark.parametrize('bad_values',[
    np.full((20,20),'pixel'),
    np.ones((20,20),dtype=complex)*(1+2j),
])
def test_cutout_rejects_non_real_numeric_image_arrays(bad_values):
    simulation=simulate(observation=Observation(npix=20),seed=41)
    with pytest.raises(ValueError,match='image must contain real numeric values'):
        Cutout(bad_values,simulation.noise_map,simulation.mask,simulation.psf,
            .1,565.,2)


def test_load_cutout_rejects_non_numeric_npz_image(tmp_path):
    path=tmp_path/'bad-image.npz'
    np.savez(path,image=np.full((20,20),'pixel'),noise_map=np.ones((20,20)),
        mask=np.ones((20,20),dtype=bool),psf_fine=np.ones((1,1)),
        observation_json='{"format":"skia-cutout-v1","image_units":"electron",'
            '"pixel_scale":0.1,"exposure":1,"oversample":1}')
    with pytest.raises(ValueError,match='image must contain real numeric values'):
        load_cutout(path)
