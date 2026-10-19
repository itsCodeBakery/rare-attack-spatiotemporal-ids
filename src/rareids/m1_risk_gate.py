"""Pure, testable M1 gate and risk-calibration utilities; no dataset access.

A differentiable two-expert probability mixture, not a claim of novel MoE.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import confusion_matrix

N_CLASSES = 10
CLASSES = ("Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
           "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms")
FEATURE_NAMES = ("entropy_target", "entropy_context", "margin_target",
                 "margin_context", "maxprob_target", "maxprob_context",
                 "l1_probability_disagreement", "argmax_disagreement",
                 "train_scaled_mean_numerical_drift",
                 "train_scaled_max_numerical_drift",
                 "categorical_history_mismatch_count")


def check_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != N_CLASSES or not np.isfinite(p).all():
        raise ValueError("Requires finite N x 10 expert probabilities")
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9) or not np.allclose(
            p.sum(1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Probability simplex violated")
    return np.clip(p, 0, 1)


def mix(p_target, p_context, gate, alpha):
    t, c = check_probabilities(p_target), check_probabilities(p_context)
    g = np.asarray(gate, dtype=float).reshape(-1)
    if t.shape != c.shape or g.shape != (len(t),) or not np.isfinite(g).all():
        raise ValueError("Mixture shapes or gate values invalid")
    if not (0 <= alpha <= 1) or np.any(g < 0) or np.any(g > 1):
        raise ValueError("Gate and alpha must belong to [0,1]")
    q = t + (alpha * g)[:, None] * (c - t)
    return check_probabilities(q)


def feature_disagreement(pt, pc):
    a, b = check_probabilities(pt), check_probabilities(pc)
    if a.shape != b.shape:
        raise ValueError("Expert shapes differ")
    def parts(p):
        ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
        two = np.partition(p, -2, axis=1)[:, -2:]
        return ent, two[:, 1] - two[:, 0], two[:, 1]
    ae, am, ac = parts(a)
    be, bm, bc = parts(b)
    return np.column_stack((ae, be, am, bm, ac, bc,
        np.abs(a-b).sum(1), (a.argmax(1)!=b.argmax(1)).astype(float)))


def numeric_scale(frame, raw_numeric):
    """Training-only robust scale for each of the 39 numeric source columns."""
    values = np.column_stack([np.asarray(frame[x], dtype=float) for x in raw_numeric])
    med = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    p25 = np.nanpercentile(values, 25, axis=0)
    p75 = np.nanpercentile(values, 75, axis=0)
    scale = p75 - p25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    med[~np.isfinite(med)] = 0.0
    return med, scale


def history_gate_features(raw_window, pt, pc, numeric_cols, categorical_cols, scale):
    """No label, CSV row index, timestamp or session identifiers are inputs."""
    core = feature_disagreement(pt, pc)
    med, robust = scale
    if len(numeric_cols) != 39 or len(categorical_cols) != 3:
        raise ValueError("M1 frozen 39-numeric/3-categorical schema mismatch")
    target = np.column_stack([np.asarray(raw_window[f"lag0__{x}"], dtype=float)
                              for x in numeric_cols])
    historical = np.stack([
        np.column_stack([np.asarray(raw_window[f"lag{k}__{x}"], dtype=float)
                         for x in numeric_cols]) for k in (4,3,2,1)], axis=1)
    if len(med) != 39 or len(robust) != 39:
        raise ValueError("Training-derived numerical scaler dimension wrong")
    delta = np.abs((historical-target[:,None,:])/robust[None,None,:])
    delta[~np.isfinite(delta)] = np.nan
    # Do not allow outlier infinities to poison the optimizer.
    delta = np.clip(np.nan_to_num(delta,nan=0.0),0.0,100.0)
    avg = delta.mean(axis=(1,2))
    maximum = delta.max(axis=(1,2))
    catdiff = np.zeros(len(raw_window), dtype=float)
    for x in categorical_cols:
        last=raw_window[f"lag0__{x}"].astype("string").fillna("__MISSING__").to_numpy()
        for k in (4,3,2,1):
            past=raw_window[f"lag{k}__{x}"].astype("string").fillna("__MISSING__").to_numpy()
            catdiff += (past != last)
    result=np.column_stack((core,avg,maximum,catdiff))
    if result.shape != (len(raw_window),11) or not np.isfinite(result).all():
        raise ValueError("Invalid M1 label-blind gate features")
    return result


def class_weights(y):
    counts = np.bincount(np.asarray(y,dtype=int),minlength=10)
    if len(counts)!=10 or np.any(counts==0):
        raise ValueError("Gate-fit subset requires all ten classes")
    weights=np.minimum(12.0,np.sqrt(len(y)/(10.0*counts)))
    per_row=weights[np.asarray(y,dtype=int)]
    return per_row/per_row.mean()


@dataclass
class Gate:
    coefficient: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    used: np.ndarray
    train_success: bool

    def predict(self, features):
        x=np.asarray(features,float)[:,self.used]
        x=np.clip((x-self.mean)/self.std,-12,12)
        return expit(x@self.coefficient[:-1]+self.coefficient[-1])


def fit_gate(features, y, pt, pc, used=None, lam=0.001):
    y=np.asarray(y,dtype=int)
    if len(y)!=len(features) or np.any((y<0)|(y>=10)):
        raise ValueError("Bad labels for gate fit")
    a,b=check_probabilities(pt),check_probabilities(pc)
    if a.shape != b.shape or len(a)!=len(y):
        raise ValueError("Gate fit shapes mismatch")
    if used is None:used=np.arange(features.shape[1])
    used=np.asarray(used,dtype=int)
    raw=np.asarray(features,dtype=float)[:,used]
    if not np.isfinite(raw).all():raise ValueError("Nonfinite gate feature")
    mean=raw.mean(0);std=raw.std(0);std=np.where(std<1e-8,1.0,std)
    xx=np.clip((raw-mean)/std,-12,12)
    wt=class_weights(y)
    iy=np.arange(len(y))
    at=a[iy,y];difference=b[iy,y]-at
    def objective(theta):
        gg=expit(xx@theta[:-1]+theta[-1])
        qs=np.maximum(at+gg*difference,1e-12)
        loss=float(np.mean(-wt*np.log(qs))+lam*np.dot(theta[:-1],theta[:-1]))
        derivative=-wt*difference*gg*(1-gg)/qs/len(y)
        grad=np.r_[xx.T@derivative+2*lam*theta[:-1],derivative.sum()]
        return loss,grad
    sol=minimize(objective,np.zeros(len(used)+1),jac=True,method="L-BFGS-B",
                 options={"maxiter":500,"ftol":1e-10})
    if not sol.success or not np.isfinite(sol.x).all():
        raise RuntimeError("Gate optimization failed; no silent fallback")
    return Gate(sol.x,mean,std,used,True)


def reports(y,p):
    p=check_probabilities(p)
    y=np.asarray(y,dtype=int)
    if len(y)!=len(p) or np.any((y<0)|(y>=10)):
        raise ValueError("Invalid metric labels")
    preds=p.argmax(1)
    cm=confusion_matrix(y,preds,labels=np.arange(10))
    supports=cm.sum(1); numpred=cm.sum(0); tp=np.diag(cm)
    recall=np.divide(tp,supports,out=np.zeros(10,float),where=supports>0)
    prec=np.divide(tp,numpred,out=np.zeros(10,float),where=numpred>0)
    denom=supports+numpred
    f1=np.divide(2*tp,denom,out=np.zeros(10,float),where=denom>0)
    normal=CLASSES.index("Normal")
    if supports[normal]==0:raise ValueError("Normal support absent")
    return {"n":len(y),"accuracy":float(tp.sum()/len(y)),
            "macro_f1":float(f1.mean()),"weighted_f1":float(np.dot(f1,supports)/len(y)),
            "benign_fpr":float(1-recall[normal]),
            "supports":supports.astype(int).tolist(),
            "recall":dict(zip(CLASSES,map(float,recall))),
            "precision":dict(zip(CLASSES,map(float,prec))),
            "f1":dict(zip(CLASSES,map(float,f1))),
            "brier":float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))),
            "confusion":cm.astype(int).tolist()}


def calibrate(y,pt,pc,gate,config):
    """Honors preregistered grid and NEVER adapts guardrail tolerances."""
    y=np.asarray(y,dtype=int)
    if set(y.tolist()) != set(range(10)):
        raise ValueError("Ten-class calibration support required")
    base_t=reports(y,pt);base_c=reports(y,pc)
    constraint=config["constraints"]
    admissible=[]; rows=[]
    for alpha in config["candidate_alpha"]:
        score=reports(y,mix(pt,pc,gate,alpha))
        okay=(score["accuracy"] >= base_c["accuracy"]+constraint["min_accuracy_vs_context_delta"]-1e-12
              and score["benign_fpr"] <= base_t["benign_fpr"]+constraint["max_benign_false_positive_increase_vs_target"]+1e-12
              and all(score["recall"][c] >= base_t["recall"][c]+floor-1e-12
                  for c,floor in constraint["min_recall_delta_vs_target"].items()))
        rows.append({"alpha":float(alpha),"feasible":bool(okay),"accuracy":score["accuracy"],
                     "macro_f1":score["macro_f1"],"benign_fpr":score["benign_fpr"],
                     **{f"recall_{n}":score["recall"][n] for n in ("Analysis","Backdoor","DoS")}})
        if okay:admissible.append((score["macro_f1"],-float(alpha)))
    if not admissible:
        return {"status":"INFEASIBLE","alpha":0.0,"grid":rows,
                "selection":"target_only_fallback_NEGATIVE_RESULT"}
    highest=max(admissible)
    return {"status":"FEASIBLE","alpha":-highest[1],"grid":rows,
            "selection":"highest_feasible_calibration_macro_f1_then_smallest_alpha"}
