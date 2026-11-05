"""Record installed versions, sizes and actual license files for reviewer inspection."""
from importlib.metadata import distributions
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
license_overrides = {'pyxdg': 'LGPL-2.0 (development only)', 'cycler': 'BSD-3-Clause', 'kiwisolver': 'BSD-3-Clause', 'objax': 'Apache-2.0', 'scipy': 'BSD-3-Clause + bundled notices', 'scikit-image': 'BSD-3-Clause + bundled notices', 'python-dateutil': 'Apache-2.0 OR BSD-3-Clause', 'parameterized': 'BSD-2-Clause', 'matplotlib': 'Matplotlib license + font notices'}
rows = ["# Dependency licenses", "", "Local development environment audit, 2026-10-31. These are the actual installed distribution license files, not a release clearance. Full notices are retained in `licenses/`. Python itself is PSF-licensed (see the interpreter distribution).", "", "No volunteer executable is built or shipped here. pyxdg (LGPL-2.0), Astropy WCSLIB (LGPL-3.0), Eigen notices in ml_dtypes (MPL-2.0), and any other restricted components or bundled notices must be excluded or resolved before P3; a development environment is not a deployable payload. In particular, permissive top-level licenses do not clear all bundled native libraries/fonts. Cross-platform wheels need a fresh audit.", "", "| Distribution | Version | Installed MiB | License metadata (see full files) | License files |", "|---|---|---:|---|---|"]
for dist in sorted(distributions(), key=lambda d: d.metadata['Name'].lower()):
    name = dist.metadata['Name']
    if name.lower() == 'skia':
        continue
    folder = root / "docs" / "licenses" / name
    if folder.exists() and name != "tensorboard-data-server":
        shutil.rmtree(folder)
    files = dist.files or []
    size = sum(dist.locate_file(f).stat().st_size for f in files if dist.locate_file(f).is_file())
    notices = []
    for f in files:
        if any(x in Path(str(f)).name.lower() for x in ('license', 'licence', 'copying', 'notice')) and dist.locate_file(f).is_file():
            content = dist.locate_file(f).read_bytes()
            target = root / 'docs' / 'licenses' / name / str(f).replace('../', '').replace('/', '__')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            notices.append(f"[{target.name}](licenses/{name}/{target.name})")
    if name == 'tensorboard-data-server':
        notices.append('[upstream Apache-2.0 license](licenses/tensorboard-data-server/UPSTREAM_LICENSE); wheel omits notice, see SOURCE.txt')
    lic = license_overrides.get(name) or dist.metadata.get('License-Expression') or dist.metadata.get('License') or 'see files'
    lic = lic.splitlines()[0][:120].replace('|', '/')
    rows.append(f"| {name} | {dist.version} | {size / 2**20:.3f} | {lic} | {', '.join(notices) or '**MISSING: review required**'} |")
(root / 'docs' / 'LICENSES.md').write_text('\n'.join(rows) + '\n')
