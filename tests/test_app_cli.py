"""skia-app gives XLA one CPU thread per pool and leaves every thread free to run on any core (KB SKI-002).

The test starts a fresh interpreter: XLA sizes its pools once, when the backend starts."""
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.path.isdir('/proc/self/task'),reason='reads thread names and CPU masks from Linux /proc')
def test_app_cli_xla_threads_are_one_per_pool_and_not_pinned():
    # A thread left on one core shares it with every other task left there; BOINC runs one task per core.
    code=("import os, collections, skia.app_cli, jax.numpy as jnp\n"
          "jnp.ones(3).sum().block_until_ready()\n"
          "full=os.sched_getaffinity(0); tids=os.listdir('/proc/self/task')\n"
          "names=collections.Counter(open(f'/proc/self/task/{t}/comm').read().strip() for t in tids)\n"
          "print(names['tf_XLAEigen'], names['tf_XLAExecute'], all(os.sched_getaffinity(int(t))==full for t in tids))")
    env={k:v for k,v in os.environ.items() if k not in ('PJRT_NPROC','NPROC','SKIA_ALL_CORES')}
    out=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,check=True).stdout.strip()
    assert out=='1 1 True'
