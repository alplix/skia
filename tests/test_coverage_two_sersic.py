import importlib.util
import sys
from pathlib import Path

BENCH=Path(__file__).parent/'bench'
sys.path.insert(0,str(BENCH))
spec=importlib.util.spec_from_file_location('coverage_two_sersic_bench',BENCH/'coverage_two_sersic.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.remove(str(BENCH))
cross_validated_coverage=module.cross_validated_coverage
summarize=module.summarize
from skia.fitter import TWO_SERSIC_PARAMETER_NAMES


def test_two_sersic_coverage_split_and_heldout_scaling():
    rows=[]
    for index,z in enumerate((0.,1.,2.,3.)):
        rows.append({'index':index,'parameters':{name:z for name in TWO_SERSIC_PARAMETER_NAMES},
            'truth':{name:0. for name in TWO_SERSIC_PARAMETER_NAMES},
            'parameter_errors':{name:1. for name in TWO_SERSIC_PARAMETER_NAMES},
            'covariance_rank':len(TWO_SERSIC_PARAMETER_NAMES),'converged':True,
            'boundary_parameters':[]})
    summary=summarize(rows)
    parameter=summary['all_finite']['parameters']['lens.theta_E']
    assert summary['all_finite']['calibration_rows']==2
    assert summary['all_finite']['evaluation_rows']==2
    assert parameter['raw_evaluation_68']['inside']==1
    assert parameter['raw_evaluation_95']['inside']==1
    assert parameter['calibration_scale_68']==2.
    assert parameter['scaled_evaluation_68']['inside']==1
    crossval=cross_validated_coverage(rows,folds=2)
    cv_parameter=crossval['groups']['all_finite']['parameters']['lens.theta_E']
    assert cv_parameter['heldout_count']==4
    assert cv_parameter['raw_68']['inside']==2
