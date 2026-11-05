import json
from io import BytesIO
import zipfile

import numpy as np
import pytest
import yaml

from skia.cutout import load_cutout
from skia.fitter import FitConfig, fit_cutout, save_fit
from skia.scanner import scan
from skia.simulator import Observation, save_simulation, simulate
from skia.task_bundle import create_task_bundle, read_task_bundle


def _inputs(tmp_path):
    sim=tmp_path/'cutout.npz'
    save_simulation(simulate(observation=Observation(npix=20),seed=99),sim)
    cutout=load_cutout(sim)
    fit=fit_cutout(cutout,FitConfig(starts=1,max_nfev=40,gamma_prior_sigma=.1))
    baseline=tmp_path/'baseline.json';save_fit(fit,baseline)
    grid=tmp_path/'grid.yaml'
    grid.write_text(yaml.safe_dump({'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,
      'x_arcsec':[.2],'y_arcsec':[.1],'alpha_Rs_arcsec':[0.,.01]}))
    return sim,baseline,grid


def test_bundle_is_deterministic_and_hash_checked(tmp_path):
    cut,baseline,grid=_inputs(tmp_path)
    one=tmp_path/'one.zip';two=tmp_path/'two.zip'
    create_task_bundle(cut,baseline,grid,one);create_task_bundle(cut,baseline,grid,two)
    assert one.read_bytes()==two.read_bytes()
    image,fit,axes=read_task_bundle(one)
    np.testing.assert_array_equal(image.image,load_cutout(cut).image)
    assert fit['format']=='skia-fit-v1' and axes['alpha_Rs_arcsec']==[0.,.01]
    with zipfile.ZipFile(one) as archive:
        with np.load(BytesIO(archive.read('cutout.npz')),allow_pickle=False) as member:
            assert set(member.files)=={'image','noise_map','mask','psf_fine','observation_json'}
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(one) as src:
        members={name:src.read(name) for name in src.namelist()}
    manifest=json.loads(members['manifest.json']);manifest['files']['baseline.json']='0'*64
    with zipfile.ZipFile(bad,'w') as dst:
        for name,value in members.items():dst.writestr(name,json.dumps(manifest).encode() if name=='manifest.json' else value)
    with pytest.raises(ValueError):read_task_bundle(tmp_path/'bad.zip')


def test_bundle_rejects_non_object_json(tmp_path):
    cut,baseline,grid=_inputs(tmp_path)
    malformed=tmp_path/'malformed.json';malformed.write_text('[]')
    with pytest.raises(ValueError,match='must contain an object'):
        create_task_bundle(cut,malformed,grid,tmp_path/'bad-input.zip')
    valid=tmp_path/'valid.zip';create_task_bundle(cut,baseline,grid,valid)
    invalid=tmp_path/'invalid-manifest.zip'
    with zipfile.ZipFile(valid) as source:
        members={name:source.read(name) for name in source.namelist()}
    members['manifest.json']=b'[]'
    with zipfile.ZipFile(invalid,'w') as target:
        for name,value in members.items():target.writestr(name,value)
    with pytest.raises(ValueError,match='must contain an object'):
        read_task_bundle(invalid)


def test_scan_checkpoint_resumes_and_rejects_other_grid(tmp_path):
    cut_path,baseline_path,grid_path=_inputs(tmp_path)
    cut=load_cutout(cut_path);baseline=json.loads(baseline_path.read_text());grid=yaml.safe_load(grid_path.read_text())
    checkpoint=tmp_path/'checkpoint.json';progress=[]
    def interrupt(done,total):
        progress.append((done,total))
        if done==1:raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError):scan(cut,baseline,grid,checkpoint_path=checkpoint,progress_callback=interrupt)
    assert len(json.loads(checkpoint.read_text())['results'])==1
    resumed_progress=[]
    resumed=scan(cut,baseline,grid,checkpoint_path=checkpoint,progress_callback=lambda d,t:resumed_progress.append((d,t)))
    direct=scan(cut,baseline,grid)
    assert resumed_progress==[(2,2)]
    assert resumed['results']==direct['results']
    changed={**grid,'alpha_Rs_arcsec':[0.,.02]}
    with pytest.raises(ValueError,match='does not match'):
        scan(cut,baseline,changed,checkpoint_path=checkpoint)


def test_scan_rejects_malformed_checkpoint_schema(tmp_path):
    cut_path,baseline_path,grid_path=_inputs(tmp_path)
    checkpoint=tmp_path/'checkpoint.json';checkpoint.write_text('[]')
    with pytest.raises(ValueError,match='expected an object'):
        scan(load_cutout(cut_path),json.loads(baseline_path.read_text()),yaml.safe_load(grid_path.read_text()),checkpoint_path=checkpoint)


@pytest.mark.parametrize(('field','value','message'),[
    ('chi2','invalid','numeric value'),
    ('chi2',10**400,'numeric value'),
    ('converged',1,'convergence flag'),
    ('parameters',{'lens.theta_E':float('nan')},'fitted parameters'),
    ('parameters',{'lens.theta_E':1.0},'fitted parameters'),
    ('boundary_parameters',['unknown.parameter'],'boundary parameter list'),
    ('nfev',-1,'optimizer record'),
])
def test_scan_rejects_malformed_checkpoint_result_rows(tmp_path,field,value,message):
    cut_path,baseline_path,grid_path=_inputs(tmp_path)
    cut=load_cutout(cut_path);baseline=json.loads(baseline_path.read_text());grid=yaml.safe_load(grid_path.read_text())
    checkpoint=tmp_path/'checkpoint.json'
    def interrupt(done,total):
        if done==1: raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError):
        scan(cut,baseline,grid,checkpoint_path=checkpoint,progress_callback=interrupt)
    payload=json.loads(checkpoint.read_text())
    payload['results'][0][field]=value
    checkpoint.write_text(json.dumps(payload,allow_nan=True))
    with pytest.raises(ValueError,match=message):
        scan(cut,baseline,grid,checkpoint_path=checkpoint)


def test_scan_rejects_incomplete_checkpoint_baseline_parameters(tmp_path):
    cut_path,baseline_path,grid_path=_inputs(tmp_path)
    checkpoint=tmp_path/'checkpoint.json'
    cut=load_cutout(cut_path);baseline=json.loads(baseline_path.read_text());grid=yaml.safe_load(grid_path.read_text())
    def interrupt(done,total):
        if done==1:raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError):scan(cut,baseline,grid,checkpoint_path=checkpoint,progress_callback=interrupt)
    payload=json.loads(checkpoint.read_text());payload['baseline_parameters']={}
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='Checkpoint baseline record is invalid'):
        scan(cut,baseline,grid,checkpoint_path=checkpoint)


def test_app_cli_emits_versioned_result_and_progress(tmp_path,capsys):
    cut,baseline,grid=_inputs(tmp_path)
    bundle=tmp_path/'task.zip';create_task_bundle(cut,baseline,grid,bundle)
    from skia.app_cli import main
    out=tmp_path/'result.txt'
    assert main([str(bundle),'-o',str(out)])==0
    output=out.read_text()
    assert output.startswith('# skia-scan-v1')
    second=tmp_path/'result-second.txt'
    assert main([str(bundle),'-o',str(second)])==0
    assert second.read_bytes()==out.read_bytes()
    assert 'PROGRESS 2 2' in capsys.readouterr().err


def test_bundle_cli_success_and_validation_error(tmp_path,capsys):
    cut,baseline,grid=_inputs(tmp_path)
    from skia.bundle_cli import main
    output=tmp_path/'task.zip'
    assert main([str(cut),str(baseline),str(grid),'-o',str(output)])==0
    assert output.is_file()
    bad=tmp_path/'bad.json';bad.write_text('[]')
    assert main([str(cut),str(bad),str(grid),'-o',str(tmp_path/'bad.zip')])==2
    assert 'must contain an object' in capsys.readouterr().err
