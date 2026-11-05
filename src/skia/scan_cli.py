"""Run a TNFW angular-strength grid scan; no cosmological mass conversion is assumed."""
import argparse
import json
import sys
from pathlib import Path
import yaml
from skia.cutout import load_cutout
from skia.scanner import load_grid, scan, save_scan


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit_json');parser.add_argument('cutout_npz');parser.add_argument('--grid',required=True);parser.add_argument('-o','--output',required=True)
    parser.add_argument('--vectorized-fields',action='store_true',help='batch TNFW field generation (experimental; may be slower)')
    a=parser.parse_args(argv)
    try:
        baseline=json.loads(Path(a.fit_json).read_text(encoding='utf-8'))
        grid=load_grid(a.grid);result=scan(load_cutout(a.cutout_npz),baseline,grid,vectorized_fields=a.vectorized_fields);save_scan(result,a.output)
    except (ValueError,OSError,EOFError,KeyError,TypeError,yaml.YAMLError) as error:
        print(f'skia-scan: {error}',file=sys.stderr);return 2
    best=result['best'];print(f"grid_points={len(result['results'])} best_delta_chi2={best['delta_chi2']:.6f} mass=not-calibrated",file=sys.stderr)
    return 0

if __name__=='__main__':raise SystemExit(main())
