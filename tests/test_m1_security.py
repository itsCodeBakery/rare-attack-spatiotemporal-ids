"""Runner fail-closed invariants without importing any official test data."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Runner imports the project's existing E6 implementation; when used in a
# cloned research repository these imports resolve to the actual E6 modules.


def test_training_only_path_rejection():
    from scripts.run_m1_context_risk import secure_file
    for name in ("UNSW_NB15_testing-set.csv", "test.csv",
                 "UNSW_NB15_training-set.csv/../test.csv"):
        with pytest.raises((ValueError,FileNotFoundError)):
            secure_file(name)


def test_group_disjoint_assertion_catches_shared_duplicates():
    import numpy as np
    from scripts.run_m1_context_risk import assert_separation
    g=np.array([0,1,1,2,3,4])
    assert_separation(g,np.array([0,1]),np.array([3,4]))
    with pytest.raises(AssertionError):
        assert_separation(g,np.array([0,1]),np.array([2,3]))
    with pytest.raises(AssertionError):
        assert_separation(g,np.array([0,1]),np.array([1,3]))


def test_missing_class_fails_closed():
    import numpy as np
    from scripts.run_m1_context_risk import fold_zero
    labels=np.repeat(np.arange(9),20)
    with pytest.raises(ValueError):
        fold_zero(labels,np.arange(len(labels)),5,2031)
