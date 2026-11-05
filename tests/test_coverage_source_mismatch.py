import importlib.util
from pathlib import Path

import pytest

BENCH=Path(__file__).parent/'bench'
spec=importlib.util.spec_from_file_location('coverage_source_mismatch_bench',BENCH/'coverage_source_mismatch.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_source_mismatch_coverage_summary_reports_raw_and_cross_validated_groups():
    rows=[]
    for index in range(10):
        parameters={name:float(index==0) * 3. for name in module.PARAMETERS}
        rows.append({'index':index,'truth':{name:0. for name in module.PARAMETERS},
            'parameters':parameters,'parameter_errors':{name:1. for name in module.PARAMETERS},
            'converged':True,'boundary_parameters':[],
            'source_truth':{'source1':{'amp':2.,'R_sersic':.2},'source2':{'amp':.5}}})
    rows.append({'index':10,'truth':None,'parameters':None,'parameter_errors':None,
        'converged':False,'boundary_parameters':[]})
    summary=module.summarize(rows,folds=5)
    assert summary['attempted']==11 and summary['finite']==10 and summary['hard_failures']==1
    theta=summary['groups']['all_finite']['parameters']['lens.theta_E']
    assert theta['raw_68']['inside']==9
    assert theta['raw_95']['inside']==9
    assert theta['cross_validated_68']['available']==10
    assert theta['cross_validated_95']['available']==10
    reference=[{'index':row['index'],
        'truth':{**row['truth'],'source.amp':2.,'source.R_sersic':.2},
        'parameters':{name:0. for name in module.PARAMETERS},
        'parameter_errors':row['parameter_errors']} for row in rows if row['parameters'] is not None]
    paired=module.compare_matched_reference(rows,reference)['lens.theta_E']
    assert paired['paired_count']==10
    assert paired['mismatch_68_inside']==9 and paired['matched_68_inside']==10
    assert paired['matched_only_68']==1
    with pytest.raises(ValueError,match='folds'):
        module.summarize(rows,folds=1)
