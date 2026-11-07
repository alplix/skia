"""Run a provisional Skia task bundle with local checkpoint/resume."""
import argparse
import os
import sys
from pathlib import Path
import zipfile

import yaml


def _single_core_xla() -> None:
    """Keep XLA's CPU thread pools on one core.

    XLA sizes its Eigen pool from the CPU affinity mask at backend start and ignores the thread-count
    environment variables. Pin to the current core while the backend is created, then restore the
    mask for every thread, so the pool stays small but is not pinned to one core.
    """
    if not hasattr(os, 'sched_setaffinity') or os.environ.get('SKIA_ALL_CORES'): return
    try:
        full = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {os.sched_getcpu() if hasattr(os, 'sched_getcpu') else min(full)})
    except OSError: return
    try:
        import jax
        jax.devices()
        import jax.numpy as jnp
        jnp.zeros(1).block_until_ready()
    finally:
        # Pool threads inherited the one-core mask; give every thread the full mask back so the OS can
        # spread tasks over all cores (otherwise every task stays on the core it started on).
        try:
            for tid in os.listdir('/proc/self/task'):
                try: os.sched_setaffinity(int(tid), full)
                except (OSError, ValueError): pass
        except OSError: pass
        try: os.sched_setaffinity(0, full)
        except OSError: pass


def _select_boinc_gpu() -> None:
    """GPU build only (SKIA_GPU=1): use the device BOINC assigned to this slot.

    The client writes <gpu_device_num> into init_data.xml in the slot directory; restrict CUDA to that
    device and let JAX allocate GPU memory on demand so several tasks can share one card.
    """
    if not os.environ.get('SKIA_GPU'): return
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if 'CUDA_VISIBLE_DEVICES' in os.environ: return
    try:
        import re
        text = Path('init_data.xml').read_text(errors='replace')
        match = re.search(r'<gpu_device_num>\s*(\d+)\s*</gpu_device_num>', text)
        if match: os.environ['CUDA_VISIBLE_DEVICES'] = match.group(1)
    except OSError: pass


_select_boinc_gpu()
_single_core_xla()

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
