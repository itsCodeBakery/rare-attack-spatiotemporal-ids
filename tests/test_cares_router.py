"""Unit tests for proposed probability-conserving classification router."""
import json
from pathlib import Path
import numpy as np
from rareids.pair_probability_router import pair_adjust

def test_router_conserves_probabilities():
    rng=np.random.default_rng(42)
    p=rng.dirichlet(np.ones(10),size=100)
    q=rng.uniform(0,1,size=100)
    for mode in ("selective","unrestricted","unchanged"):
        corrected,weight=pair_adjust(p,2,3,q,mode=mode)
        np.testing.assert_allclose(corrected.sum(axis=1),p.sum(axis=1),atol=1e-12)
        np.testing.assert_allclose(corrected[:,2]+corrected[:,3],p[:,2]+p[:,3],atol=1e-12)
        np.testing.assert_allclose(corrected[:,[0,1,4,5,6,7,8,9]],p[:,[0,1,4,5,6,7,8,9]],atol=1e-12)
        assert np.all((weight>=0)&(weight<=1))

def test_router_selection_and_protocol():
    p=np.zeros((3,10))
    p[0,[2,3]]=[.2,.8]
    p[1,[2,3]]=[.001,.001];p[1,6]=.998
    p[2,[2,3]]=[.5,.5]
    corrected,g=pair_adjust(p,2,3,np.array([.99,.99,.5]))
    assert g[0]>0 and g[1]==0 and g[2]==0
    c=json.loads(Path("configs/cares_pair_prototype_seed42.json").read_text())
    assert c["official_test"]=="FORBIDDEN"
    assert c["parent_macro_f1_internal"]==0.6366908581786823
    assert len(c["parent_model_sha256"])==64
    assert len(c["parent_probabilities_sha256"])==64
    assert len(c["pairs"])==3
