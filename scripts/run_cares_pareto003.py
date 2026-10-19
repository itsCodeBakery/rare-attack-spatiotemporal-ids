"""CARES-PARETO-003: class-pair-specific risk routing, no training/test leakage.

Learn safe target-specific switching from frozen train OOF LightGBM predictions.
Only change labels when standard and rare models disagree; use independent
OOF calibration/audit subsets to choose and verify thresholds before frozen
development validation. No labeled validation is used for gate learning.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support

CLASSES=("Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Normal","Reconnaissance","Shellcode","Worms")
THRESHOLDS=(0.0,0.02,0.05,0.10,0.15,0.20,0.30,0.40)
MIN_ROUTE=20
MAX_STEPS=10

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4194304),b""):h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def evaluate(y,yp):
    cm=confusion_matrix(y,yp,labels=np.arange(10))
    p,r,f,s=precision_recall_fscore_support(y,yp,labels=np.arange(10),zero_division=0)
    return {"accuracy":float(accuracy_score(y,yp)),
            "macro_f1":float(f1_score(y,yp,average="macro",labels=np.arange(10),zero_division=0)),
            "weighted_f1":float(f1_score(y,yp,average="weighted",labels=np.arange(10),zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,yp)),
            "dos_recall":float(r[2]),"analysis_recall":float(r[0]),
            "backdoor_recall":float(r[1]),"dos_tp":int(cm[2,2]),
            "analysis_tp":int(cm[0,0]),"backdoor_tp":int(cm[1,1]),
            "correct":int(np.trace(cm)),"n":len(y)},cm,pd.DataFrame({
                "class":CLASSES,"support":s,"precision":p,"recall":r,"f1":f})

def get_probs(path,key1,key2,ykey):
    with np.load(path,allow_pickle=False) as f:
        y=f[ykey].astype(np.int64)
        rare=f[key1].astype(np.float64)
        std=f[key2].astype(np.float64)
    if rare.shape!=std.shape or rare.shape!=(len(y),10):
        raise ValueError("Frozen parent probability matrices mismatch")
    if not np.isfinite(rare).all() or not np.isfinite(std).all():
        raise ValueError("Invalid frozen probabilities")
    return y,rare,std

def get_validation(base):
    records={}
    for name,key in (("rare","lgbm_rare_sqrt"),
                     ("standard","lgbm_unweighted")):
        file=base/key/"predictions_PRIVATE.npz"
        with np.load(file,allow_pickle=False) as f:
            records[name]=(f["y_true"].astype(int),f["probabilities"].astype(float))
    if not np.array_equal(records["rare"][0],records["standard"][0]):
        raise ValueError("Frozen target ordering does not agree")
    return records["rare"][0],records["rare"][1],records["standard"][1]

def candidates(rare,std):
    """Prediction pair and standard-over-rare confidence margin; features only."""
    rare_label=rare.argmax(1)
    std_label=std.argmax(1)
    switch_margin=std[np.arange(len(std)),std_label]-std[np.arange(len(std)),rare_label]
    return rare_label,std_label,switch_margin

def apply_rules(rare,std,rules):
    r,s,margin=candidates(rare,std)
    predicted=r.copy()
    applied=np.zeros(len(r),dtype=bool)
    for rule in rules:
        mask=(r==rule["original_class"])&(s==rule["alternative_class"]) & (
            margin>=rule["minimum_standard_margin"]) & (~applied)
        predicted[mask]=s[mask];applied[mask]=True
    return predicted,applied

def pass_constraints(base,current,config):
    """Improve accuracy+macro F1 simultaneously; bound DoS/backdoor/Analysis recall."""
    return (current["accuracy"]>=base["accuracy"]+config["min_accuracy_gain"]
      and current["macro_f1"]>=base["macro_f1"]+config["min_macro_f1_gain"]
      and current["dos_recall"]>=base["dos_recall"]-config["dos_recall_tolerance"]
      and current["analysis_recall"]>=base["analysis_recall"]-config["analysis_recall_tolerance"]
      and current["backdoor_recall"]>=base["backdoor_recall"]-config["backdoor_recall_tolerance"])

def select_rules(y,rare,std,cfg):
    """Greedy Pareto routing on *training-derived OOF calibration only*."""
    ref=rare.argmax(1)
    baseline=evaluate(y,ref)[0]
    raw,alt,margin=candidates(rare,std)
    rules=[]
    current=ref.copy()
    history=[]
    # Fixed candidate catalogue; only rules independently supported by 20+ samples.
    for step in range(MAX_STEPS):
        best=None
        for i in range(10):
            for j in range(10):
                if i==j:continue
                for margin_threshold in THRESHOLDS:
                    mask=(raw==i)&(alt==j)&(margin>=margin_threshold)&(current==i)
                    count=int(mask.sum())
                    if count<MIN_ROUTE:continue
                    trial=current.copy();trial[mask]=j
                    m=evaluate(y,trial)[0]
                    # Pairwise conservative improvement test: overall AND class-specific.
                    if (m["accuracy"]<=evaluate(y,current)[0]["accuracy"]
                        or m["macro_f1"]<=evaluate(y,current)[0]["macro_f1"]):
                        continue
                    if not pass_constraints(baseline,m,cfg):
                        continue
                    delta=(m["accuracy"]-baseline["accuracy"])+(
                        m["macro_f1"]-baseline["macro_f1"])
                    if best is None or delta>best[0]:
                        best=(delta,i,j,margin_threshold,count,m,trial)
        if best is None:break
        _,i,j,t,n,m,current=best
        rules.append({"original_class":i,"alternative_class":j,
                      "original_label":CLASSES[i],"alternative_label":CLASSES[j],
                      "minimum_standard_margin":float(t),"calibration_support":int(n)})
        history.append({"step":step+1,"pair":f"{CLASSES[i]}->{CLASSES[j]}","n":int(n),**m})
        print(f"CALIBRATION SELECTED RULE {step+1}: {CLASSES[i]} -> {CLASSES[j]} "
              f"(n={n}, threshold={t}) accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}",
              flush=True)
    return rules,history,baseline

def main(args):
    root=Path(args.input)
    base=Path(args.baseline)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.config).read_text())
    if cfg["official_test"]!="LOCKED":raise ValueError("Official test must stay LOCKED")
    expected=cfg["source_oof_sha256"]
    filename=root/"train_OOF_probabilities_PRIVATE.npz"
    if sha(filename)!=expected:raise ValueError("OOF archive hash mismatch")
    train_y,train_rare,train_std=get_probs(filename,"rare","standard","y_true")
    val_y,val_rare,val_std=get_validation(base)
    if len(train_y)!=140328 or len(val_y)!=35005:
        raise ValueError("Wrong split dimensions")
    calib_ids,audit_ids=train_test_split(
        np.arange(len(train_y)),test_size=cfg["audit_fraction"],
        stratify=train_y,random_state=cfg["seed"])
    rules,history,calibration_baseline=select_rules(
        train_y[calib_ids],train_rare[calib_ids],train_std[calib_ids],cfg)
    audit_ref=evaluate(train_y[audit_ids],train_rare[audit_ids].argmax(1))[0]
    audit_pred,audit_mask=apply_rules(train_rare[audit_ids],train_std[audit_ids],rules)
    audit_score=evaluate(train_y[audit_ids],audit_pred)[0]
    approved=bool(rules and pass_constraints(audit_ref,audit_score,cfg))
    # Independent OOF audit decides whether rulebook may be deployed,
    # without reading any validation labels until approval frozen.
    deployed=rules if approved else []
    write_json(out/"protocol.json",{**cfg,"rules_learned_without_validation":rules,
             "rules_deployed":deployed,
             "training_calibration_baseline":calibration_baseline,
             "train_derived_audit_baseline":audit_ref,
             "train_derived_audit_candidate":audit_score,
             "train_derived_audit_accepted":approved,
             "audit_changed_flows":int(audit_mask.sum()),
             "calibration_count":len(calib_ids),"audit_count":len(audit_ids),
             "training_calibration_history":history,
             "official_test_used":False,
             "warning":"Train-derived OOF and full-model distributions can differ. Validation is an exploratory holdout only."})
    results={}
    options={"frozen_rare":val_rare.argmax(1),
             "frozen_standard":val_std.argmax(1),
             "unrestricted_switch":val_std.argmax(1),
             "risk_pair_gated":apply_rules(val_rare,val_std,deployed)[0]}
    for name,pred in options.items():
        m,cm,per=evaluate(val_y,pred)
        directory=out/name;directory.mkdir(exist_ok=True)
        pd.DataFrame(cm,index=CLASSES,columns=CLASSES).rename_axis("true_class").to_csv(
            directory/"confusion.csv")
        per.to_csv(directory/"per_class.csv",index=False)
        np.savez_compressed(directory/"predictions_PRIVATE.npz",
            y_true=val_y.astype(np.uint8),y_pred=pred.astype(np.uint8))
        m.update({"variant":name,"official_test_used":False,
                  "changed_from_rare":int(np.sum(pred!=val_rare.argmax(1))),
                  "private_predictions_sha256":sha(directory/"predictions_PRIVATE.npz")})
        write_json(directory/"metrics.json",m)
        results[name]=m
        print(f"MEASURED {name}: accuracy={m['accuracy']:.5f} "
              f"macro-F1={m['macro_f1']:.5f} "
              f"DoS={m['dos_tp']} Backdoor={m['backdoor_tp']} Analysis={m['analysis_tp']}",
              flush=True)
    write_json(out/"summary.json",{"experiment_id":cfg["experiment_id"],
        "training_oof_calibration_count":len(calib_ids),
        "training_oof_audit_count":len(audit_ids),
        "audit_accepted":approved,"rules_deployed":deployed,
        "variants":results,"official_test_used":False})
    pd.DataFrame(results.values()).to_csv(out/"summary_scores.csv",index=False)
    print("\nCARES-PARETO-003 TRAIN-DERIVED RISK ROUTING COMPLETE",flush=True)
    print("OOF audit approved:",approved,"rules deployed:",len(deployed),flush=True)
    print(pd.read_csv(out/"summary_scores.csv")[[
        "variant","accuracy","macro_f1","dos_recall","analysis_recall",
        "backdoor_recall"]].to_string(index=False),flush=True)
    print("OUTPUTS",out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("input","baseline","config","output"):
        p.add_argument("--"+name,required=True)
    main(p.parse_args())
