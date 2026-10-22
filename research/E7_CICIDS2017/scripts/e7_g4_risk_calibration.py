"""E7 G4 exploratory day-shift risk/abstention audit. No fitting and no source edits.

Calibrate ONLY on the source-heldout validation day, NEVER on already-inspected
G3 test outcomes. DKW bands are validation-only, i.i.d.-conditional diagnostics,
not guaranteed cross-day risk bounds. Original G3 models remain unchanged.
"""
import math
import numpy as np

GRID=np.linspace(0.0,1.0,201,dtype=np.float64)
NEVER_ALARM=1.000001
NEVER_BENIGN=-0.000001
DELTA=0.05
MAX_FPR=0.01
MAX_FNR=0.05

def check_binary(y,p):
    y=np.asarray(y,dtype=np.uint8)
    p=np.asarray(p,dtype=np.float64)
    assert len(y)==len(p) and len(y)>0 and set(np.unique(y))=={0,1}
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return y,p

def dkw_epsilon(n,delta=DELTA):
    assert n>0 and 0<delta<1
    return float(math.sqrt(math.log(2.0/delta)/(2*n)))

def choose_band(y_valid,p_valid):
    """Validation-only selection with simultaneous empirical-CDF deviation bound.

    Maximize recall at fixed FPR conservative bound; on ties use largest high.
    For auto-benign choose the largest low (strictly below high) satisfying an
    attack FNR bound. Empty feasible sets fail closed with finite sentinels.
    """
    y,p=check_binary(y_valid,p_valid)
    benign=np.sort(p[y==0])
    attacks=np.sort(p[y==1])
    nb,na=len(benign),len(attacks)
    assert nb>0 and na>0
    eps_b=dkw_epsilon(nb)
    eps_a=dkw_epsilon(na)
    candidates=np.append(GRID,NEVER_ALARM)
    fpr_counts=nb-np.searchsorted(benign,candidates,side='left')
    tpr_counts=na-np.searchsorted(attacks,candidates,side='left')
    fpr_emp=fpr_counts/nb
    accepted=np.flatnonzero((fpr_emp+eps_b)<=MAX_FPR)
    if len(accepted):
        max_tp=tpr_counts[accepted].max()
        tied=accepted[tpr_counts[accepted]==max_tp]
        idx=int(tied[-1])   # largest threshold for ties
        high=float(candidates[idx])
        alarm_feasible=True
    else:
        high=NEVER_ALARM
        alarm_feasible=False
    low_candidates=np.append(np.array([NEVER_BENIGN]),GRID)
    fn_counts=np.searchsorted(attacks,low_candidates,side='right')
    lows_ok=np.flatnonzero((fn_counts/na+eps_a<=MAX_FNR)&(low_candidates<high))
    if len(lows_ok):
        low=float(low_candidates[int(lows_ok[-1])])
        benign_feasible=True
    else:
        low=NEVER_BENIGN
        benign_feasible=False
    assert low<high
    selection={
        'alarm_high':high,'auto_benign_low':low,
        'validation_benign_rows':int(nb),'validation_attack_rows':int(na),
        'fpr_dkw_epsilon':eps_b,'fnr_dkw_epsilon':eps_a,
        'alarm_bound_feasible':bool(alarm_feasible),
        'benign_bound_feasible':bool(benign_feasible),
        'val_empirical_alarm_fpr':float((benign>=high).mean()),
        'val_empirical_auto_benign_fnr':float((attacks<=low).mean()),
        'val_empirical_alarm_fpr_plus_dkw':float((benign>=high).mean()+eps_b),
        'val_empirical_auto_benign_fnr_plus_dkw':float((attacks<=low).mean()+eps_a),
        'validation_auto_coverage':float(((p<=low)|(p>=high)).mean()),
        'max_alarm_fpr':MAX_FPR,'max_auto_benign_fnr':MAX_FNR,
        'confidence_tuning':'DKW simultaneous CDF band on this validation day; NO cross-day guarantee',
        'selection_policy':'val-only maximize TP under FPR+Dkw cap; largest-high tie; largest allowed low'
    }
    return selection

def selective_metrics(y,p,labels,selection):
    y,p=check_binary(y,p)
    labels=np.asarray(labels,dtype=str)
    assert len(labels)==len(y)
    hi=selection['alarm_high'];lo=selection['auto_benign_low']
    assert lo<hi
    alert=p>=hi; normal=p<=lo
    assert not (alert&normal).any()
    accept=alert|normal; reject=~accept
    counted=int(accept.sum())
    wrong=int(np.count_nonzero(alert&(y==0))+np.count_nonzero(normal&(y==1)))
    nb=int((y==0).sum()); na=int((y==1).sum())
    flagged=int(np.count_nonzero(alert&(y==1)))
    false_alarm=int(np.count_nonzero(alert&(y==0)))
    missed=int(np.count_nonzero(normal&(y==1)))
    by_family={}
    for subtype in sorted(set(labels)-{'BENIGN'}):
        mask=labels==subtype
        support=int(mask.sum())
        assert support>0
        by_family[subtype]={
            'support':support,
            'alert_recall_all_attack_rows':float(alert[mask].mean()),
            'defer_fraction':float(reject[mask].mean()),
            'auto_benign_miss_fraction':float(normal[mask].mean()),
            'low_support':bool(support<100)
        }
    return {
        'rows':int(len(y)),'decided_rows':counted,
        'coverage':float(accept.mean()),'abstention_rate':float(reject.mean()),
        'selective_error':float(wrong/counted) if counted else None,
        'selective_accuracy':float(1-wrong/counted) if counted else None,
        'auto_attack_alerts':int(alert.sum()),'auto_benign_decisions':int(normal.sum()),
        'false_alarms':false_alarm,'attacks_alerted':flagged,
        'attack_misses_declared_benign':missed,
        'test_benign_fpr':float(false_alarm/nb),
        'test_attack_alert_recall':float(flagged/na),
        'test_attack_miss_fraction':float(missed/na),
        'test_benign_deferred_fraction':float(np.count_nonzero(reject&(y==0))/nb),
        'test_attack_deferred_fraction':float(np.count_nonzero(reject&(y==1))/na),
        'test_coverage_below_0_50':bool(accept.mean()<0.5),
        'unseen_family_diagnostics':by_family
    }
