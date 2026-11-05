"""Run a provisional Skia task bundle with local checkpoint/resume."""
import argparse
import sys
from pathlib import Path
import zipfile

import yaml

from skia.scanner import scan, save_scan
from skia.task_bundle import read_task_bundle


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_bundle')
    parser.add_argument('-o','--output',required=True)
    parser.add_argument('--checkpoint',help='checkpoint path; defaults to OUTPUT.checkpoint.json')
    args=parser.parse_args(argv)
    checkpoint=Path(args.checkpoint) if args.checkpoint else Path(args.output+'.checkpoint.json')
    try:
        cutout,baseline,grid=read_task_bundle(args.task_bundle)
        def progress(done,total): print(f'PROGRESS {done} {total}',file=sys.stderr,flush=True)
        result=scan(cutout,baseline,grid,checkpoint_path=checkpoint,progress_callback=progress)
        save_scan(result,args.output)
    except (ValueError,OSError,EOFError,KeyError,TypeError,zipfile.BadZipFile,yaml.YAMLError) as error:
        print(f'skia-app: {error}',file=sys.stderr);return 2
    best=result['best']
    print(f"grid_points={len(result['results'])} best_delta_chi2={best['delta_chi2']:.6f} mass=not-calibrated",file=sys.stderr)
    return 0


if __name__=='__main__': raise SystemExit(main())
