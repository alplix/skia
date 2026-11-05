"""Create a deterministic skia-task-v1 ZIP from fit, cutout and grid files."""
import argparse
import sys
import yaml

from skia.task_bundle import create_task_bundle


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cutout_npz');parser.add_argument('baseline_json');parser.add_argument('grid_yaml');parser.add_argument('-o','--output',required=True)
    args=parser.parse_args(argv)
    try: create_task_bundle(args.cutout_npz,args.baseline_json,args.grid_yaml,args.output)
    except (ValueError,OSError,EOFError,KeyError,TypeError,yaml.YAMLError) as error:
        print(f'skia-bundle: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
