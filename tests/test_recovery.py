"""Check study accounting, immutable outputs and its executable reproduction path."""
from pathlib import Path
import json
import runpy
import subprocess
import sys

import numpy as np
import pytest

from skia.fitter import PARAMETER_NAMES

SCRIPT = Path(__file__).parent/'bench/recovery.py'


def test_recovery_metrics_include_failures_and_bound_contacts():
    namespace = runpy.run_path(str(SCRIPT), run_name='recovery_test')
    truth = dict.fromkeys(PARAMETER_NAMES, 0.0)
    common = {'truth': truth, 'chi2': 100, 'dof': 100, 'elapsed_seconds': 1}
    rows = [
        dict(common, parameters=dict.fromkeys(PARAMETER_NAMES, 1.0), converged=True,
             boundary_parameters=[], gross_recovery_failure=True),
        dict(common, parameters=dict.fromkeys(PARAMETER_NAMES, 3.0), converged=False,
             boundary_parameters=['lens.gamma'], gross_recovery_failure=True),
        dict(common, parameters=None, converged=False, boundary_parameters=[], gross_recovery_failure=True),
    ]
    stats = namespace['_metrics'](rows)
    assert stats['count'] == 3 and stats['finite_count'] == 2
    assert stats['hard_failures'] == 1 and stats['optimization_failures'] == 2
    assert stats['boundary_count'] == 1 and stats['gross_recovery_failures'] == 3
    assert stats['all_finite']['parameters']['lens.gamma']['bias'] == 2
    assert stats['all_finite']['parameters']['lens.gamma']['scatter'] == 1
    assert stats['converged_interior']['count'] == 1
    assert stats['converged_interior']['parameters']['lens.gamma']['bias'] == 1
    all_bad = namespace['_metrics']([rows[-1]])
    assert all_bad['all_finite']['parameters']['lens.gamma'] is None
    assert all_bad['timing_median_seconds'] is None


def test_recovery_script_smoke_and_refuse_overwrite(tmp_path):
    command = [sys.executable, str(SCRIPT), '--count', '2', '--bench-count', '1', '--output', str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'RECOVERY SUMMARY: 2 attempted' in result.stdout
    summary = json.loads((tmp_path/'P1_summary.json').read_text())
    assert summary['count'] == 2
    assert len((tmp_path/'P1_recovery.jsonl').read_text().splitlines()) == 2
    assert 'SMOKE TEST ONLY' in (tmp_path/'P1_REPORT.md').read_text()
    assert (tmp_path/'P1_recovery.png').stat().st_size > 0
    protocol = json.loads((tmp_path/'P1_protocol.json').read_text())
    assert protocol['fit_config']['starts'] == 2
    assert protocol['parameter_seed'] == 20261031
    assert len(protocol['source_sha256']['src/skia/fitter.py']) == 64
    first = (tmp_path/'P1_recovery.jsonl').read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and 'already exist' in result.stderr
    assert first == (tmp_path/'P1_recovery.jsonl').read_bytes()


def test_constrained_recovery_script_smoke(tmp_path):
    script=Path(__file__).parent/'bench/constrained_recovery.py'
    result=subprocess.run([sys.executable,str(script),'--count','2','--output',str(tmp_path)],
                          capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    assert len((tmp_path/'paired.jsonl').read_text().splitlines())==2
    report=(tmp_path/'REPORT.md').read_text()
    assert 'Normal(2, 0.1)' in report
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['free']['finite']==2 and summary['gamma_prior_sigma_0.1']['finite']==2


def test_estimated_noise_psf_protocol_smoke(tmp_path):
    script=Path(__file__).parent/'bench/estimated_noise_psf.py'
    result=subprocess.run([sys.executable,str(script),'--count','2','--output',str(tmp_path)],
                          capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    assert len((tmp_path/'paired.jsonl').read_text().splitlines())==2
    report=(tmp_path/'REPORT.md').read_text()
    assert '+10%' in report and 'combined' in report
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['known_noise_matched_psf']['finite']==2
    assert summary['estimated_noise_psf_plus_10pct']['finite']==2
