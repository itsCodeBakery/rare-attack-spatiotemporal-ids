"""First genuine CARES-PAIR-001 experiment; no refitting of baseline LightGBM.

Fit three disjointly calibrated pair specialists using TRAINING fold ONLY.
Apply a prespecified probability-conserving router to frozen baseline scores.
Compare six honest ablations on the identical 35,005 development targets.
No designated official test CSV is opened, ever.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,f1_score,balanced_accuracy_score,
                             confusion_matrix,precision_recall_fscore_support)
from tqdm.auto import tqdm

from run_ctrl003_tabular_fast import prepare
from rareids.pair_probability_router import pair_adjust

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for data in iter(lambda:f.read(4194304),b""):h.update(data)
    return h.hexdigest()

def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",
                          encoding="utf-8")

def score(y,p,classes):
    pred=p.argmax(1)
    cm=confusion_matrix(y,pred,labels=np.arange(len(classes)))
    pr,rc,f1,sup=precision_recall_fscore_support(
        y,pred,labels=np.arange(len(classes)),zero_division=0)
    result={"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,labels=np.arange(len(classes)),
                                      average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(y,pred,labels=np.arange(len(classes)),
                                         average="weighted",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "n":int(len(y)),"correct":int(np.trace(cm))}
    per=pd.DataFrame({"class":classes,"support":sup.astype(int),
                     "precision":pr,"recall":rc,"f1":f1})
    return result,cm,per

def logit(p):
    clipped=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
    return np.log(clipped/(1-clipped))

def fit_expert(features,labels,valid_features,classes,pair,cfg,path,seed):
    a,b=pair
    ia,ib=classes.index(a),classes.index(b)
    selected=np.flatnonzero((labels==ia)|(labels==ib))
    positive=(labels[selected]==ia).astype(np.int64)
    if np.bincount(positive,minlength=2).min()<30:
        raise ValueError("Insufficient train-only pair examples: "+str(pair))
    fit_idx,remaining=train_test_split(
        selected,test_size=.30,random_state=seed,stratify=positive)
    stop_idx,cal_idx=train_test_split(
        remaining,test_size=.5,random_state=seed+17,
        stratify=(labels[remaining]==ia).astype(int))
    yfit=(labels[fit_idx]==ia).astype(int)
    ystop=(labels[stop_idx]==ia).astype(int)
    ycal=(labels[cal_idx]==ia).astype(int)
    rare_positive_weight=min(12.,max(1.,np.sqrt((len(yfit)-sum(yfit))/sum(yfit))))
    train=lgb.Dataset(features.iloc[fit_idx],label=yfit,
                      weight=np.where(yfit==1,rare_positive_weight,1.),
                      categorical_feature=["proto","service","state"],
                      feature_name=list(features.columns))
    stop=lgb.Dataset(features.iloc[stop_idx],label=ystop,reference=train,
                     categorical_feature=["proto","service","state"])
    params={"objective":"binary","metric":"binary_logloss",
            "learning_rate":cfg["spec"]["learning_rate"],
            "num_leaves":cfg["spec"]["num_leaves"],
            "min_data_in_leaf":cfg["spec"]["min_data_in_leaf"],
            "lambda_l2":4.,"feature_fraction":.9,"bagging_fraction":.9,
            "bagging_freq":1,"seed":seed,"num_threads":4,
            "deterministic":True,"force_col_wise":True,"verbosity":-1}
    history={}
    start=time.monotonic()
    model=lgb.train(params,train,num_boost_round=cfg["spec"]["iterations"],
        valid_sets=[stop],valid_names=["stop_only"],
        callbacks=[lgb.early_stopping(cfg["spec"]["early_stop"],verbose=False),
                   lgb.record_evaluation(history)])
    model.save_model(str(path/"specialist.txt"))
    cal_scores=logit(model.predict(features.iloc[cal_idx],
                                   num_iteration=model.best_iteration))
    calibrator=LogisticRegression(C=1.,max_iter=500,random_state=seed)
    calibrator.fit(cal_scores.reshape(-1,1),ycal)
    a1=float(calibrator.coef_[0,0])
    b1=float(calibrator.intercept_[0])
    raw_valid=logit(model.predict(valid_features,num_iteration=model.best_iteration))
    q=expit(a1*raw_valid+b1)
    np.savez_compressed(path/"validation_specialist_PRIVATE.npz",
                        conditional_positive=q.astype(np.float32))
    result={"positive_class":a,"compared_class":b,
            "training_examples":int(len(fit_idx)),
            "earlystop_examples":int(len(stop_idx)),
            "calibration_examples":int(len(cal_idx)),
            "positive_train_examples":int(sum(yfit)),
            "positive_weight":rare_positive_weight,
            "best_iteration":int(model.best_iteration),
            "platt_coefficient":a1,"platt_intercept":b1,
            "calibration_positive_rate":float(np.mean(ycal)),
            "elapsed_seconds":round(time.monotonic()-start,2),
            "checkpoint_sha256":sha(path/"specialist.txt"),
            "private_conditional_sha256":sha(path/"validation_specialist_PRIVATE.npz")}
    save_json(path/"specialist_info.json",result)
    pd.DataFrame({"iteration":list(range(1,len(history["stop_only"]["binary_logloss"])+1)),
                  "stop_binary_logloss":history["stop_only"]["binary_logloss"]}).to_csv(
                  path/"earlystop_history.csv",index=False)
    return q,result

def main(repo,config_path,ref,baseline,out):
    cfg=json.loads(Path(config_path).read_text())
    if cfg.get("official_test")!="FORBIDDEN":
        raise ValueError("Official test is reserved")
    repo,ref,baseline,out=map(Path,(repo,ref,baseline,out))
    out.mkdir(parents=True,exist_ok=True)
    previous=baseline/"summary.json"
    if not previous.is_file():raise FileNotFoundError("Frozen LightGBM baseline missing")
    baseline_meta=json.loads((baseline/"lgbm_rare_sqrt"/"metrics.json").read_text())
    baseline_predictions=baseline/"lgbm_rare_sqrt"/"predictions_PRIVATE.npz"
    if (baseline_meta["checkpoint_sha256"]!=sha(baseline/"lgbm_rare_sqrt"/"model.txt") or
        baseline_meta["private_predictions_sha256"]!=sha(baseline_predictions)):
        raise ValueError("Frozen baseline checkpoint/probabilities hash drift")
    if abs(baseline_meta["macro_f1"]-cfg["parent_macro_f1_internal"])>1e-12 if "parent_macro_f1_internal" in cfg else False:
        raise ValueError("Baseline performance drift")
    if baseline_meta["count"]!=35005:
        raise ValueError("Baseline validation target count differs")
    # Reuse the previously verified official-TRAIN-only source and identical fold.
    source_cfg=json.loads((repo/"configs"/"ctrl003_fast_tabular.json").read_text())
    Xtr,ytr,Xv,yv,classes,cats,lineage=prepare(source_cfg,ref)
    if len(Xv)!=35005 or lineage["test_dataset_accessed"]:
        raise ValueError("Target integrity failed")
    with np.load(baseline_predictions,allow_pickle=False) as saved:
        yp=saved["y_true"].astype(int)
        base=saved["probabilities"].astype(np.float64)
    if not np.array_equal(yp,yv) or base.shape!=(len(yv),10):
        raise ValueError("Private baseline targets differ from immutable reference")
    base_f1=f1_score(yv,base.argmax(1),labels=np.arange(10),
                     average="macro",zero_division=0)
    if abs(base_f1-baseline_meta["macro_f1"])>1e-10:
        raise ValueError("Frozen baseline predictions fail original F1 replay")
    np.random.seed(cfg["seed"])
    original_commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    (out/"source_commit.txt").write_text(original_commit+"\n")
    save_json(out/"config_effective.json",cfg)
    save_json(out/"data_lineage.json",lineage)
    pairs=cfg["pairs"]
    candidates={}
    for i,pair in enumerate(tqdm(pairs,desc="TRAIN CONFUSION PAIR SPECIALISTS",unit="pairs")):
        path=out/("specialist_"+pair[0].lower());path.mkdir(exist_ok=True)
        info_path=path/"specialist_info.json"
        if info_path.exists():
            info=json.loads(info_path.read_text())
            if (sha(path/"specialist.txt")!=info["checkpoint_sha256"] or
                sha(path/"validation_specialist_PRIVATE.npz")!=info["private_conditional_sha256"]):
                raise ValueError("Previously saved specialist has changed")
            with np.load(path/"validation_specialist_PRIVATE.npz") as arr:
                q=arr["conditional_positive"].astype(np.float64)
            print("REPLAYING SAVED",pair[0],"; no specialist retraining")
        else:
            q,info=fit_expert(Xtr,ytr,Xv,classes,pair,cfg,path,cfg["seed"]+i*101)
        candidates[pair[0]]=(q,info)
        print("TRAINED SPECIALIST",pair[0],"best round",info["best_iteration"],
              "calibrated on",info["calibration_examples"],"disjoint training rows",flush=True)
    args=cfg["router"]
    kwargs={"mass_threshold":args["mass_threshold"],"mass_full":args["mass_full"],
            "confidence_threshold":args["confidence_threshold"],
            "confidence_full":args["confidence_full"]}
    variants={"baseline":base.copy()}
    for cls,tag in (("DoS","dos_only"),("Backdoor","backdoor_only"),("Analysis","analysis_only")):
        pair=[x for x in pairs if x[0]==cls][0]
        values,_=pair_adjust(base,classes.index(pair[0]),classes.index(pair[1]),
                             candidates[cls][0],mode="selective",**kwargs)
        variants[tag]=values
    for mode,tag in (("unrestricted","all_unrestricted"),("selective","all_selective")):
        p=base.copy()
        for a,b in pairs:
            p,_=pair_adjust(p,classes.index(a),classes.index(b),
                            candidates[a][0],mode=mode,**kwargs)
        variants[tag]=p
    results={}
    for key,p in variants.items():
        result,cm,per=score(yv,p,classes)
        result["delta_macro_f1_vs_baseline_pp"]=100*(result["macro_f1"]-base_f1)
        result["changed_predictions_vs_baseline"]=int(np.sum(p.argmax(1)!=base.argmax(1)))
        for a,b in pairs:
            result[a+"_to_"+b]=int(cm[classes.index(a),classes.index(b)])
            result[a+"_correct"]=int(cm[classes.index(a),classes.index(a)])
        folder=out/key;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/"predictions_PRIVATE.npz",
                            y_true=yv.astype(np.uint8),
                            y_pred=p.argmax(1).astype(np.uint8),
                            probabilities=p.astype(np.float32))
        pd.DataFrame(cm,index=classes,columns=classes).rename_axis(
            "true_class").to_csv(folder/"confusion.csv")
        per.to_csv(folder/"per_class.csv",index=False)
        result["predictions_sha256"]=sha(folder/"predictions_PRIVATE.npz")
        save_json(folder/"metrics.json",result)
        results[key]=result
        print(f"{key:19} macro-F1 {100*result['macro_f1']:.3f}% "
              f"accuracy {100*result['accuracy']:.3f}% "
              f"DoS→Exploits {result['DoS_to_Exploits']} "
              f"Backdoor→Exploits {result['Backdoor_to_Exploits']}",flush=True)
    scores=pd.DataFrame([{"variant":key,**val} for key,val in results.items()])
    scores.to_csv(out/"comparison_scores.csv",index=False)
    winner=scores.loc[scores["macro_f1"].idxmax()]
    save_json(out/"summary.json",{"experiment_id":cfg["experiment_id"],
         "best_development_variant":winner["variant"],
         "best_development_macro_f1":float(winner["macro_f1"]),
         "parent_unchanged_macro_f1":float(base_f1),
         "variants":results,"specialists":{k:val[1] for k,val in candidates.items()},
         "same_original_validation_targets":True,
         "official_test_used":False,
         "warning":"Development-only; pair choice motivated by prior validation confusions. Not independent journal outperformance."})
    print("\nCARES FIRST PROPOSED-METHOD EXPERIMENT COMPLETE")
    print(scores[["variant","macro_f1","accuracy","balanced_accuracy",
                  "DoS_to_Exploits","Backdoor_to_Exploits"]].to_string(index=False))
    print("PRESERVE RUN OUTPUTS BEFORE KAGGLE SESSION EXPIRES:",out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("repo","config","reference","baseline","output"):
        ap.add_argument("--"+name,required=True)
    a=ap.parse_args()
    main(a.repo,a.config,a.reference,a.baseline,a.output)
