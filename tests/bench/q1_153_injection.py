"""Tile-held-out injection study with real Q1 residuals and matched clean nulls."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from skia.cutout import Cutout
from skia.fitter import FitConfig, SmoothRenderer, parameter_vector
from skia.realdata import load_q1_fits
from skia.scanner import halo_deflection_field, scan
from skia.simulator import Lens, Source

THRESHOLDS=(2.,5.,10.)
INJECTIONS=(.02,.05)
GRID={'profile':'TNFW','Rs_arcsec':.2,'r_trunc_arcsec':.8,
      'x_arcsec':[-.3,.3],'y_arcsec':[-.3,.3],'alpha_Rs_arcsec':[0.,.02,.05]}
CONFIG=FitConfig(starts=1,seed=731,max_nfev=80,gamma_prior_sigma=.1)
SPLIT_SALT='skia-q1-153-v1:'


def _split(tile_index: str) -> str:
    """Assign complete Euclid tile groups to a fixed 80/20 split by hash."""
    bucket=int(hashlib.sha256((SPLIT_SALT+str(tile_index)).encode()).hexdigest(),16)%5
    return 'calibration' if bucket else 'evaluation'


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(('skia-q1-null-v1:'+name).encode()).digest()[:8],'big')


def _vector(parameters: dict) -> np.ndarray:
    lens=Lens(**{k.removeprefix('lens.'):v for k,v in parameters.items() if k.startswith('lens.')})
    source=Source(**{k.removeprefix('source.'):v for k,v in parameters.items() if k.startswith('source.')})
    return parameter_vector(lens,source)


def _measure(cutout: Cutout, baseline: dict) -> dict:
    result=scan(cutout,baseline,GRID)
    return {'max_delta_chi2':float(result['best']['delta_chi2']),
            'best':{k:result['best'][k] for k in ('x_arcsec','y_arcsec','alpha_Rs_arcsec')},
            'converged_points':sum(bool(row['converged']) for row in result['results']),
            'grid_points':len(result['results']),
            'points':[{'x_arcsec':row['x_arcsec'],'y_arcsec':row['y_arcsec'],
                       'alpha_Rs_arcsec':row['alpha_Rs_arcsec'],'delta_chi2':row['delta_chi2'],
                       'chi2':row['chi2'],'converged':row['converged'],'status':row['status'],
                       'nfev':row['nfev'],'boundary_parameters':row['boundary_parameters']}
                      for row in result['results']]}


def _matched_clean_null(cutout: Cutout, vector: np.ndarray, filename: str) -> Cutout:
    """Make a subhalo-free smooth lens with this cutout's assumed PSF/noise map."""
    model=SmoothRenderer(cutout).evaluate(vector)
    rng=np.random.Generator(np.random.PCG64(_seed(filename)))
    image=model+rng.normal(0.,cutout.noise_map)
    return replace(cutout,image=image)


def _injected_real(cutout: Cutout, vector: np.ndarray, renderer: SmoothRenderer,
                   smooth: np.ndarray, alpha: float) -> Cutout:
    field=halo_deflection_field(cutout,.3,.3,alpha,.2,.8)
    with_halo=renderer.evaluate(vector,field)
    return replace(cutout,image=cutout.image+with_halo-smooth)


def _b5_real_null(row: dict) -> dict:
    """Reuse the identical fixed-grid candidate scan already completed in B5."""
    result=row['scan']
    return {'max_delta_chi2':float(result['best_delta_chi2']),'best':result['best'],
            'converged_points':int(result['converged_points']),
            'grid_points':int(result['grid_points']),
            'failed_points':result['unconverged_points'],
            'source_run':row.get('_source_run','unspecified')}


def _appendable_rows(path: Path) -> tuple[dict[str,dict],int]:
    if not path.exists():return {},0
    raw=path.read_bytes();complete=raw.rfind(b'\n')+1
    if complete<len(raw):path.write_bytes(raw[:complete])
    rows={}
    for line in raw[:complete].splitlines():
        if not line:continue
        row=json.loads(line)
        if row['file'] in rows:raise ValueError(f'duplicate result row for {row["file"]}')
        rows[row['file']]=row
    return rows,len(rows)


def _summarize(rows: list[dict], output: Path) -> dict:
    train=[row for row in rows if row['split']=='calibration']
    test=[row for row in rows if row['split']=='evaluation']
    train_null=np.asarray([row['matched_clean_null']['max_delta_chi2'] for row in train])
    if len(train_null)<20 or not test:raise ValueError('insufficient calibration/evaluation cutouts')
    calibrated=float(np.quantile(train_null,.95,method='higher'))
    snr_edges=np.quantile([row['integrated_snr'] for row in train],[1/3,2/3]).tolist()
    def counts(group, getter):
        values=np.asarray([getter(row) for row in group],dtype=float)
        return {'n':len(values),'median':float(np.median(values)) if len(values) else None,
                'counts_at_fixed_delta_chi2':{str(t):int(np.sum(values>=t)) for t in THRESHOLDS},
                'calibrated_threshold':calibrated,
                'count_at_calibrated_threshold':int(np.sum(values>=calibrated))}
    summary={
      'calibration_clean_null':counts(train,lambda row:row['matched_clean_null']['max_delta_chi2']),
      'evaluation_clean_null':counts(test,lambda row:row['matched_clean_null']['max_delta_chi2']),
      'evaluation_q1_unmodified_candidates':counts(test,lambda row:row['real_null']['max_delta_chi2']),
      'evaluation_q1_injections':{},
      'all_q1_unmodified_candidates':counts(rows,lambda row:row['real_null']['max_delta_chi2']),
      'calibrated_threshold_definition':'empirical 95th percentile using method=higher of calibration-tile matched clean-null maximum delta-chi2; selected before evaluating held-out tiles',
      'split':{'calibration_cutouts':len(train),'evaluation_cutouts':len(test),
               'calibration_tiles':len({row['tile_index'] for row in train}),
               'evaluation_tiles':len({row['tile_index'] for row in test}),
               'tile_overlap':len({row['tile_index'] for row in train}&{row['tile_index'] for row in test})},
      'snr_stratification':{'edges_from_calibration_integrated_snr':snr_edges,
                            'evaluation_bins':{}}}
    for label,group in (
        ('low',[row for row in test if row['integrated_snr']<=snr_edges[0]]),
        ('middle',[row for row in test if snr_edges[0]<row['integrated_snr']<=snr_edges[1]]),
        ('high',[row for row in test if row['integrated_snr']>snr_edges[1]])):
        summary['snr_stratification']['evaluation_bins'][label]={
          'n':len(group),
          'q1_candidate_null':counts(group,lambda row:row['real_null']['max_delta_chi2']),
          'injections':{f'{alpha:.2f}':counts(group,lambda row:row['injections'][f'{alpha:.2f}']['max_delta_chi2']) for alpha in INJECTIONS}}
    for alpha in INJECTIONS:
        key=f'{alpha:.2f}'
        summary['evaluation_q1_injections'][key]=counts(test,lambda row:row['injections'][key]['max_delta_chi2'])
    if summary['split']['tile_overlap']:
        raise ValueError('tile leakage between calibration and evaluation split')
    output.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/p2-153-injection-results.jsonl')
    parser.add_argument('--b5-input',default='docs/b5-153-real-robustness.jsonl')
    parser.add_argument('--files-list',help='newline-separated FITS filenames; defaults to every manifested cutout')
    args=parser.parse_args(argv)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    b5_rows={json.loads(line)['file']:json.loads(line) for line in Path(args.b5_input).read_text().splitlines() if line}
    data_dir=Path('data/q1_cutouts')
    if args.files_list:
        names=[line.strip() for line in Path(args.files_list).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(names)!=len(set(names)):parser.error('files-list contains duplicate names')
        if any(Path(name).name!=name or not name.endswith('_VIS.fits') for name in names):
            parser.error('files-list entries must be FITS basenames')
        files=sorted(data_dir/name for name in names)
    else:
        # Keep the historical 153-cutout default reproducible after the dataset grows.
        files=sorted(data_dir/name for name in b5_rows)
    with Path('data/q1_cutouts/manifest.csv').open(encoding='utf-8',newline='') as handle:
        manifest={row['file']:row for row in csv.DictReader(handle)}
    selected={path.name for path in files}
    if not selected or not selected.issubset(manifest) or set(b5_rows)!=selected:
        parser.error('selected FITS files need complete matching B5 and manifest rows')
    if any(not path.is_file() for path in files):parser.error('selected cutout file is missing')
    source_run=str(args.b5_input)
    existing,_=_appendable_rows(output)
    if not set(existing).issubset({path.name for path in files}):raise ValueError('output contains unknown cutout rows')
    with output.open('a',encoding='utf-8') as stream:
      for number,path in enumerate(files,1):
        if path.name in existing:
            print(f'Q1 injection {number}/{len(files)} already complete: {path.name}',flush=True);continue
        meta=manifest[path.name];tile=str(meta['tile_index']);split=_split(tile)
        base_row=dict(b5_rows[path.name]);base_row['_source_run']=source_run
        if 'baseline' not in base_row:raise ValueError(f'B5 baseline unavailable for {path.name}')
        cutout,_,_=load_q1_fits(path)
        parameters=base_row['baseline']['parameters'];vector=_vector(parameters)
        baseline={'format':'skia-fit-v1','parameters':parameters,'optimizer':asdict(CONFIG)}
        renderer=SmoothRenderer(cutout);smooth=renderer.evaluate(vector)
        clean=_matched_clean_null(cutout,vector,path.name)
        item={'file':path.name,'tile_index':int(tile),'split':split,
              'integrated_snr':float(np.maximum(cutout.image[cutout.mask],0.).sum()/np.sqrt(np.square(cutout.noise_map[cutout.mask]).sum())),
              'matched_clean_null':_measure(clean,baseline),
              'real_null':_b5_real_null(base_row),'injections':{}}
        if split=='evaluation':
            for alpha in INJECTIONS:
                changed=_injected_real(cutout,vector,renderer,smooth,alpha)
                item['injections'][f'{alpha:.2f}']=_measure(changed,baseline)
        stream.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');stream.flush()
        existing[path.name]=item
        print(f'Q1 injection {number}/{len(files)} ({split}) {path.name}',flush=True)
    rows=[existing.get(path.name) for path in files]
    if any(row is None for row in rows):raise RuntimeError('some injection rows are missing; rerun to resume')
    summary=_summarize(rows,output.with_suffix('.summary.json'))
    print(json.dumps(summary,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
