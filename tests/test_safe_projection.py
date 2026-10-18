"""Numerical guard and deterministic/replayable PCA checks."""
import numpy as np
import pytest
from rareids.safe_projection import robust_pca_2d

def test_all_constant_embeddings_plot_without_stochastic_failure():
    x=np.ones((250,128),dtype=np.float32)*10
    coords,meta=robust_pca_2d(x)
    assert coords.shape==(250,2)
    assert np.isfinite(coords).all()
    assert (coords==0).all()
    assert meta["status"]=="zero_variance"

def test_pca_deterministic_for_nonconstant_values():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(155,42))
    a,d1=robust_pca_2d(x)
    b,d2=robust_pca_2d(x)
    assert np.allclose(a,b)
    assert d1==d2
    assert d1["status"]=="ok"

def test_nonfinite_values_fail_early():
    x=np.ones((20,3))
    x[0,0]=np.nan
    with pytest.raises(ValueError,match="Non-finite"):
        robust_pca_2d(x)
