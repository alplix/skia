"""Deterministic ZIP task bundle format for the provisional local app."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import yaml

from skia.cutout import Cutout, load_cutout, save_cutout
from skia.scanner import normalize_grid, vector_from_fit

FILES=('baseline.json','cutout.npz','grid.yaml')
MAX_FILE_BYTES=128*1024*1024


def _json_mapping(value: bytes, label: str) -> dict:
    """Decode a JSON object and normalize malformed input to a user-facing error."""
    try:
        decoded=json.loads(value)
    except (UnicodeDecodeError,json.JSONDecodeError) as error:
        raise ValueError(f'Invalid {label} JSON: {error}') from error
    if not isinstance(decoded,dict): raise ValueError(f'{label} JSON must contain an object')
    return decoded


def _read_sources(cutout_path: str|Path,baseline_path: str|Path,grid_path: str|Path) -> dict[str,bytes]:
    data={'cutout.npz':Path(cutout_path).read_bytes(),
          'baseline.json':Path(baseline_path).read_bytes(),
          'grid.yaml':Path(grid_path).read_bytes()}
    if any(len(value)>MAX_FILE_BYTES for value in data.values()): raise ValueError('Task bundle member exceeds size limit')
    baseline=_json_mapping(data['baseline.json'],'baseline')
    if baseline.get('format')!='skia-fit-v1': raise ValueError('Expected skia-fit-v1 baseline')
    vector_from_fit(baseline)
    grid=normalize_grid(yaml.safe_load(data['grid.yaml']))
    # Canonicalize the cutout so simulator-only model/truth arrays never enter
    # a task bundle and archive members contain only the worker input contract.
    parsed_cutout=load_cutout(BytesIO(data['cutout.npz']))
    canonical=BytesIO();save_cutout(parsed_cutout,canonical)
    data['cutout.npz']=canonical.getvalue()
    if len(data['cutout.npz'])>MAX_FILE_BYTES: raise ValueError('Canonical cutout exceeds size limit')
    data['baseline.json']=(json.dumps(baseline,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    data['grid.yaml']=(yaml.safe_dump(grid,sort_keys=True).encode())
    return data


def create_task_bundle(cutout_path: str|Path,baseline_path: str|Path,grid_path: str|Path,output: str|Path) -> None:
    """Create a reproducible ZIP with image/noise/PSF/mask, fit and angular grid."""
    data=_read_sources(cutout_path,baseline_path,grid_path)
    manifest={'format':'skia-task-v1','files':{name:sha256(data[name]).hexdigest() for name in FILES}}
    members={'manifest.json':(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode(),**data}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
        for name in ('manifest.json',*FILES):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system=3;info.external_attr=0o600<<16;info.compress_type=zipfile.ZIP_STORED
            archive.writestr(info,members[name])


def read_task_bundle(path: str|Path) -> tuple[Cutout,dict,dict]:
    """Validate hashes and schema, returning cutout, baseline and normalized grid."""
    try:
        with zipfile.ZipFile(path,'r') as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(('manifest.json',*FILES)):
                raise ValueError('Task bundle must contain exactly manifest.json, baseline.json, cutout.npz, grid.yaml')
            if any(archive.getinfo(name).file_size>MAX_FILE_BYTES for name in names): raise ValueError('Task bundle member exceeds size limit')
            members={name:archive.read(name) for name in names}
    except zipfile.BadZipFile as error:
        raise ValueError(f'Invalid task ZIP bundle: {error}') from error
    manifest=_json_mapping(members['manifest.json'],'manifest')
    if manifest.get('format')!='skia-task-v1' or not isinstance(manifest.get('files'),dict):
        raise ValueError('Expected skia-task-v1 manifest')
    if set(manifest['files'])!=set(FILES): raise ValueError('Task manifest file list is invalid')
    for name in FILES:
        if sha256(members[name]).hexdigest()!=manifest['files'][name]: raise ValueError(f'Task bundle digest mismatch: {name}')
    baseline=_json_mapping(members['baseline.json'],'baseline')
    if baseline.get('format')!='skia-fit-v1': raise ValueError('Expected skia-fit-v1 baseline')
    vector_from_fit(baseline)
    cutout=load_cutout(BytesIO(members['cutout.npz']))
    grid=normalize_grid(yaml.safe_load(members['grid.yaml']))
    return cutout,baseline,grid
