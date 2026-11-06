import json
import numpy as np
import pytest
import yaml

from skia.cli import main as fit_main
from skia.cutout import load_cutout
from skia.fitter import fit_cutout,FitConfig,save_fit,parameter_vector,physical_parameters
from skia.scanner import halo_deflection_field,halo_deflection_fields,load_grid,normalize_grid,scan,save_scan,vector_from_fit
from skia.simulator import Lens,Source,simulate,Observation,Subhalo,save_simulation


def test_angular_grid_validation(tmp_path):
    path=tmp_path/'grid.yaml';path.write_text(yaml.safe_dump({'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,'x_arcsec':[0.],'y_arcsec':[0.],'alpha_Rs_arcsec':[0.,.01]}))
    grid=load_grid(path);assert grid['alpha_Rs_arcsec']==[0.,.01]
    path.write_text('profile: TNFW\nRs_arcsec: -1\nr_trunc_arcsec: .8\nx_arcsec: [0]\ny_arcsec: [0]\nalpha_Rs_arcsec: [0]\n')
    with pytest.raises(ValueError):load_grid(path)
    path.write_text('profile: NFW')
    with pytest.raises(ValueError):load_grid(path)


@pytest.mark.parametrize('changes',[
    {'Rs_arcsec':None},
    {'x_arcsec':{'unexpected':'mapping'}},
    {'alpha_Rs_arcsec':['not-a-number']},
    {'Rs_arcsec':True},
    {'x_arcsec':[False]},
    {'alpha_Rs_arcsec':[True]},
])
def test_normalize_grid_rejects_non_numeric_values(changes):
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,
          'x_arcsec':[0.],'y_arcsec':[0.],'alpha_Rs_arcsec':[0.]}
    grid.update(changes)
    with pytest.raises(ValueError,match='numeric'):
        normalize_grid(grid)


@pytest.mark.parametrize('invalid',[True,10**400,float('nan'),'1'])
def test_vector_from_fit_rejects_non_json_numeric_parameter_values(invalid):
    baseline={'format':'skia-fit-v1','parameters':physical_parameters(parameter_vector(Lens(),Source()))}
    baseline['parameters']['lens.theta_E']=invalid
    with pytest.raises(ValueError,match='finite real number'):
        vector_from_fit(baseline)


def test_vector_from_fit_does_not_mutate_baseline_payload():
    baseline={'format':'skia-fit-v1','parameters':physical_parameters(parameter_vector(Lens(),Source()))}
    baseline['parameters']['lens.theta_E']=1
    vector_from_fit(baseline)
    assert type(baseline['parameters']['lens.theta_E']) is int


def test_tnfw_angular_field_zero_and_finite(tmp_path):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
      s=simulate(observation=Observation(npix=20),seed=8);p=d+'/c.npz';save_simulation(s,p);cut=load_cutout(p)
      zero=halo_deflection_field(cut,0,0,0,.2,.8)
      pad=(cut.psf_fine.shape[0]//2+cut.oversample-1)//cut.oversample
      assert zero.shape==((cut.image.shape[0]+2*pad)*cut.oversample,(cut.image.shape[1]+2*pad)*cut.oversample,2)
      np.testing.assert_array_equal(zero,0)
      field=halo_deflection_field(cut,.2,-.1,.02,.2,.8)
      assert np.isfinite(field).all() and np.max(abs(field))>0
      batch=halo_deflection_fields(cut,[(0.,0.,0.),(.2,-.1,.02)],.2,.8,chunk_size=1)
      empty=halo_deflection_fields(cut,[],.2,.8)
      assert empty.shape==(0,*zero.shape)
      with pytest.raises(ValueError,match='radii must be finite and positive'):
          halo_deflection_fields(cut,[],-.2,.8)
      with pytest.raises(ValueError,match='numeric and finite'):
          halo_deflection_fields(cut,[],.2,float('nan'))
      np.testing.assert_allclose(batch[0],zero,atol=0,rtol=0)
      np.testing.assert_allclose(batch[1],field,atol=1e-14,rtol=1e-12)
      with pytest.raises(ValueError):halo_deflection_field(cut,0,0,-1,.2,.8)
      with pytest.raises(ValueError,match='numeric and finite'):
          halo_deflection_field(cut,0,0,0,None,.8)
      with pytest.raises(ValueError,match='numeric and finite'):
          halo_deflection_field(cut,True,0,0,.2,.8)
      with pytest.raises(ValueError,match='numeric and finite'):
          halo_deflection_fields(cut,[],'bad',.8)
      with pytest.raises(ValueError,match='points must be finite'):
          halo_deflection_fields(cut,[{'bad':'point'}],.2,.8)
      with pytest.raises(ValueError,match='points must be numeric finite'):
          halo_deflection_fields(cut,[(False,0.,0.)],.2,.8)


def test_scan_refits_baseline_writes_fixed_format(tmp_path):
    s=simulate(observation=Observation(npix=20),subhalo=Subhalo(center_x=.4,center_y=.2),seed=12)
    cp=tmp_path/'c.npz';save_simulation(s,cp);cut=load_cutout(cp)
    baseline=fit_cutout(cut,FitConfig(starts=1,max_nfev=80));fp=tmp_path/'fit.json';save_fit(baseline,fp)
    payload=json.loads(fp.read_text());v=vector_from_fit(payload)
    assert np.isfinite(v).all()
    with pytest.raises(ValueError,match='missing parameters'):
        vector_from_fit({'format':'skia-fit-v1','parameters':{'lens.theta_E':1.0}})
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,'x_arcsec':[.4], 'y_arcsec':[.2], 'alpha_Rs_arcsec':[0.,.02]}
    result=scan(cut,payload,grid)
    vectorized=scan(cut,payload,grid,vectorized_fields=True)
    assert result['format']=='skia-scan-v1' and result['mass_unit']=='not-calibrated'
    assert len(result['results'])==2
    assert vectorized['results']==result['results']
    zero=next(r for r in result['results'] if r['alpha_Rs_arcsec']==0)
    assert abs(zero['delta_chi2'])<1e-8
    assert result['best']['delta_chi2']>=0
    with pytest.raises(ValueError,match='missing grid field'):
        scan(cut,payload,{'profile':'TNFW'})
    out=tmp_path/'scan.txt';save_scan(result,out)
    assert out.read_text().startswith('# skia-scan-v1')
    lines=out.read_text().splitlines();assert len(lines)==4 and all(len(x.split())==6 for x in lines[1:])


def test_scan_cli_invalid_baseline(tmp_path,capsys):
    import yaml
    fp=tmp_path/'fit.json';fp.write_text('{"format":"bad"}')
    cp=tmp_path/'c.npz';save_simulation(simulate(observation=Observation(npix=20),seed=2),cp)
    grid=tmp_path/'grid.yaml';grid.write_text(yaml.safe_dump({'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,'x_arcsec':[0],'y_arcsec':[0],'alpha_Rs_arcsec':[0]}))
    from skia.scan_cli import main
    assert main([str(fp),str(cp),'--grid',str(grid),'-o',str(tmp_path/'out')])==2
    fp.write_text('[]')
    assert main([str(fp),str(cp),'--grid',str(grid),'-o',str(tmp_path/'out')])==2
    assert 'Baseline must be a mapping' in capsys.readouterr().err


def test_zero_field_points_reuse_the_baseline_fit_bit_for_bit(tmp_path,monkeypatch):
    # alpha_Rs 0 gives an all-zero field, the baseline problem again: scan() reuses the baseline fit for it (SKI-003).
    # The rows must equal those of fitting every point, also when a resume has lost the baseline fit from memory.
    import skia.scanner as scanner
    s=simulate(observation=Observation(npix=20),subhalo=Subhalo(center_x=.4,center_y=.2),seed=12)
    cp=tmp_path/'c.npz';save_simulation(s,cp);cut=load_cutout(cp)
    fp=tmp_path/'fit.json';save_fit(fit_cutout(cut,FitConfig(starts=2,max_nfev=80)),fp);payload=json.loads(fp.read_text())
    grid={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,'x_arcsec':[.4],'y_arcsec':[.2,-.2],'alpha_Rs_arcsec':[0.,.02]}
    calls=[];real_fit=scanner.fit_cutout
    monkeypatch.setattr(scanner,'fit_cutout',lambda *a,**k:calls.append(1) or real_fit(*a,**k))
    reused=scan(cut,payload,grid)
    assert len(calls)==1+2  # the baseline and the two points with alpha_Rs 0.02
    with monkeypatch.context() as m:
        m.setattr(scanner,'_is_zero_field',lambda field:False)
        fitted=scan(cut,payload,grid)
    assert reused['results']==fitted['results'] and reused['baseline_chi2']==fitted['baseline_chi2']
    assert [r['delta_chi2'] for r in reused['results'] if r['alpha_Rs_arcsec']==0]==[0.,0.]
    checkpoint=tmp_path/'scan.checkpoint.json';scan(cut,payload,grid,checkpoint_path=checkpoint)
    saved=json.loads(checkpoint.read_text());saved['results']=saved['results'][:1];checkpoint.write_text(json.dumps(saved))
    resumed=scan(cut,payload,grid,checkpoint_path=checkpoint)
    assert resumed['results']==reused['results']
    np.testing.assert_array_equal(halo_deflection_field(cut,.4,.2,0.,.2,.8).view(np.uint64),0)
