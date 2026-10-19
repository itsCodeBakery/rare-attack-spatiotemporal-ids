"""Pure synthetic M1 checks; no benchmark datasets and no test-set access."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from rareids.m1_risk_gate import (mix, check_probabilities, fit_gate,
    calibrate, numeric_scale, history_gate_features, feature_disagreement,
    reports, CLASSES)


def probabilities(n=120,seed=123):
    rng=np.random.default_rng(seed)
    return rng.dirichlet(np.ones(10),size=n),rng.dirichlet(np.ones(10),size=n)


def test_simplex_convex_and_alpha_zero():
    a,b=probabilities()
    q=mix(a,b,np.linspace(0,1,len(a)),.75)
    assert np.allclose(q.sum(1),1)
    assert np.min(q)>=0
    assert np.allclose(mix(a,b,np.ones(len(a)),0),a)
    assert np.allclose(mix(a,b,np.ones(len(a)),1),b)
    with pytest.raises(ValueError):mix(a,b,np.ones(len(a))*1.1,.5)
    with pytest.raises(ValueError):mix(a,b,np.ones(len(a)),1.2)
    with pytest.raises(ValueError):check_probabilities(a*2)


def test_gate_optimization_deterministic_and_finite():
    a,b=probabilities(300)
    y=np.tile(np.arange(10),30)
    inputs=np.hstack([feature_disagreement(a,b),np.ones((len(y),3))])
    gate=fit_gate(inputs,y,a,b)
    pred=gate.predict(inputs)
    gate2=fit_gate(inputs,y,a,b)
    assert gate.train_success
    assert np.all((pred>0)&(pred<1))
    assert np.allclose(pred,gate2.predict(inputs))
    q=mix(a,b,pred,.5)
    assert np.isfinite(q).all()


def test_calibration_feasibility_alpha_fallback_without_relaxation():
    n=200
    y=np.tile(np.arange(10),20)
    correct=np.eye(10)[y]*.99+.001
    correct/=correct.sum(1,keepdims=True)
    other=np.roll(correct,1,axis=1)
    config={"candidate_alpha":[0,.25,.5,.75,1],"constraints":{
        "min_accuracy_vs_context_delta":-0.01,
        "max_benign_false_positive_increase_vs_target":0.005,
        "min_recall_delta_vs_target":{"Analysis":-0.02,"Backdoor":-0.02,"DoS":-0.02}}}
    outcome=calibrate(y,correct,other,np.ones(n),config)
    assert outcome["status"]=="FEASIBLE"
    assert outcome["alpha"]==0
    assert len(outcome["grid"])==5
    # No feasible alpha: context perfect, target consistently predicts wrong class.
    impossible=calibrate(y,other,correct,np.zeros(n),config)
    assert impossible["status"]=="INFEASIBLE"
    assert impossible["alpha"]==0
    assert all(not r["feasible"] for r in impossible["grid"])


def test_no_label_or_id_feature_dependency():
    rng=np.random.default_rng(7)
    n=30
    frame=pd.DataFrame({f"lag{k}__num{j}":rng.uniform(0,20,n)
                        for k in (4,3,2,1,0) for j in range(39)})
    for k in (4,3,2,1,0):
        for c in ("proto","service","state"):
            frame[f"lag{k}__{c}"]=rng.choice(["A","B"],n)
    a,b=probabilities(n)
    numeric=[f"num{j}" for j in range(39)]
    scaler=numeric_scale(frame[[f"lag0__{j}" for j in numeric]].rename(
        columns={f"lag0__{j}":j for j in numeric}),numeric)
    values=history_gate_features(frame,a,b,numeric,["proto","service","state"],scaler)
    frame["attack_cat"]="Worms"
    frame["label"]=1
    frame["id"]=np.arange(n)
    frame["timestamp"]=np.arange(n)*10
    assert np.allclose(values,history_gate_features(frame,a,b,numeric,["proto","service","state"],scaler))
    assert values.shape==(n,11)
    assert np.isfinite(values).all()


def test_no_test_path_and_group_overlap_utilities():
    # The script-level no-test path and group-disjoint tests run separately.
    from sklearn.model_selection import StratifiedGroupKFold
    y=np.repeat(np.arange(10),12)
    groups=np.arange(len(y))
    splits=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2032).split(
        np.zeros(len(y)),y,groups))
    assert len(splits)==4
    for tr,va in splits:
        assert not (set(groups[tr])&set(groups[va]))
    assert set(np.concatenate([va for _,va in splits]).tolist())==set(range(len(y)))


def test_classwise_metrics_are_from_confusion():
    a,_=probabilities(200)
    y=np.tile(np.arange(10),20)
    result=reports(y,a)
    assert sum(result["supports"])==200
    assert np.isclose(result["macro_f1"],np.mean(list(result["f1"].values())))
    assert "Normal" in CLASSES
