"""Synthetic G4 calibration integrity, threshold edge cases, and abstention."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/e7_g4_risk_calibration.py'
spec=importlib.util.spec_from_file_location('g4risk',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_small_validation_class_fails_closed():
    y=np.array([0,0,1,1],dtype='uint8')
    p=np.array([0.1,0.2,0.8,0.9])
    selection=m.choose_band(y,p)
    assert selection['alarm_high']==m.NEVER_ALARM
    assert selection['auto_benign_low']==m.NEVER_BENIGN
    assert selection['alarm_bound_feasible'] is False
    assert selection['benign_bound_feasible'] is False
    out=m.selective_metrics(y,p,np.array(['BENIGN','BENIGN','Bot','DDoS']),selection)
    assert out['decided_rows']==0
    assert out['selective_error'] is None
    assert out['coverage']==0

def test_reliable_separation_enables_band_and_no_overlap():
    benign=np.repeat([0.02,0.1],40000)
    attack=np.repeat([0.8,0.95],40000)
    y=np.concatenate([np.zeros(len(benign)),np.ones(len(attack))]).astype('uint8')
    p=np.concatenate([benign,attack])
    selection=m.choose_band(y,p)
    assert selection['alarm_bound_feasible']
    assert selection['benign_bound_feasible']
    assert selection['auto_benign_low']<selection['alarm_high']
    assert selection['val_empirical_alarm_fpr_plus_dkw']<=0.01
    assert selection['val_empirical_auto_benign_fnr_plus_dkw']<=0.05
    labels=np.array(['BENIGN']*len(benign)+['DDoS']*len(attack))
    out=m.selective_metrics(y,p,labels,selection)
    assert out['false_alarms']==0
    assert out['attack_misses_declared_benign']==0
    assert out['selective_error']==0
    assert out['unseen_family_diagnostics']['DDoS']['support']==len(attack)

def test_val_only_selection_independent_of_test_labels():
    y=np.array([0]*50000+[1]*20000,dtype='uint8')
    p=np.array([0.1]*50000+[0.8]*20000,dtype=float)
    choice=m.choose_band(y,p)
    test_scores=np.array([0.1,0.8,0.5],dtype=float)
    a=m.selective_metrics(np.array([0,1,1]),test_scores,
        np.array(['BENIGN','Attack X','Attack Y']),choice)
    b=m.selective_metrics(np.array([1,0,1]),test_scores,
        np.array(['Attack X','BENIGN','Attack Y']),choice)
    assert choice['alarm_high']==m.choose_band(y,p)['alarm_high']
    assert a['unseen_family_diagnostics']!=b['unseen_family_diagnostics']

def test_bad_probabilities_and_single_class_halt():
    with pytest.raises(AssertionError):
        m.choose_band(np.array([0,0]),np.array([0.1,0.4]))
    with pytest.raises(AssertionError):
        m.choose_band(np.array([0,1]),np.array([float('nan'),0.4]))
    with pytest.raises(AssertionError):
        m.choose_band(np.array([0,1]),np.array([0.2,1.4]))

def test_fpr_selected_without_test_based_tuning():
    rng=np.random.default_rng(42)
    y=np.concatenate([np.zeros(10000),np.ones(2000)]).astype('uint8')
    p=np.concatenate([rng.uniform(0,0.3,10000),rng.uniform(0.3,0.9,2000)])
    selection=m.choose_band(y,p)
    assert selection['alarm_high'] in set(m.GRID.tolist())|{m.NEVER_ALARM}
    assert selection['auto_benign_low'] in set(m.GRID.tolist())|{m.NEVER_BENIGN}
    assert selection['confidence_tuning'].endswith('NO cross-day guarantee')
