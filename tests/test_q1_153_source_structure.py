import runpy

import pytest


def test_source_complexity_comparison_reports_raw_and_fractional_gain():
    compare=runpy.run_path('tests/bench/q1_153_source_structure.py',run_name='q1_source_test')['_compare']
    result=compare(100.,70.,20.)
    assert result['source_complexity_delta_chi2']==30.
    assert result['tnfw_delta_chi2']==20.
    assert result['source_complexity_percent_of_one_sersic_chi2']==30.
    assert result['tnfw_percent_of_one_sersic_chi2']==20.
    assert result['source_complexity_gain_larger'] is True


def test_source_complexity_comparison_rejects_invalid_baseline():
    compare=runpy.run_path('tests/bench/q1_153_source_structure.py',run_name='q1_source_test')['_compare']
    with pytest.raises(ValueError,match='finite and baseline positive'):
        compare(0.,1.,0.)
