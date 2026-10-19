"""CARES-OOF-002: leakage-controlled multiclass residual calibration and risk gate.

Use 3-fold training-only OOF LightGBM distributions; never fit the residual
model or its correction gate on the 35,005 development-validation targets.
Freeze thresholds using a separate training-only OOF gate-selection subset.
Published SCV test remains unopened; this is an experimental methodology.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
from run_ctrl003_tabular_fast import prepare

def hash_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as fh:
        for blob in iter(lambda:fh.read(4*1024*1024),b""):h.update(blob)
    return h.hexdigest()

def save_json(p,v):
    Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")

def normalize(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-10,1)
    return p/p.sum(axis=1,keepdims=True)

def meta_features(standard,rare):
    a,b=normalize(standard),normalize(rare)
    a_sort=np.sort(a,axis=1);b_sort=np.sort(b,axis=1)
    return np.column_stack([
        np.log(a),np.log(b),
        a_sort[:,-1]-a_sort[:,-2],b_sort[:,-1]-b_sort[:,-2],
        -(a*np.log(a)).sum(axis=1),-(b*np.log(b)).sum(axis=1),
        a.max(axis=1),b.max(axis=1)]).astype(np.float64)

def metrics(y,p,classes):
    pred=p.argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=np.arange(10))
    pr,rc,f1,support=precision_recall_fscore_support(
        y,pred,labels=np.arange(10),zero_division=0)
    return ({
        "accuracy":float(accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,average="macro",labels=np.arange(10),zero_division=0)),
        "weighted_f1":float(f1_score(y,pred,average="weighted",labels=np.arange(10),zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "n":int(len(y)),"correct":int(np.trace(cm)),
        "backdoor_TP":int(cm[classes.index("Backdoor"),classes.index("Backdoor")]),
        "analysis_TP":int(cm[classes.index("Analysis"),classes.index("Analysis")]),
        "dos_TP":int(cm[classes.index("DoS"),classes.index("DoS")]),
    },cm,pd.DataFrame({"class":classes,"support":support,
                        "precision":pr,"recall":rc,"f1":f1}))

def find_gate(y,base,meta,classes,cfg):
    """Gate and interpolation hyperparameters selected without validation labels."""
    base=normalize(base);meta=normalize(meta)
    bscore,_,_=metrics(y,base,classes)
    records=[]
    for alpha in cfg["candidate_alpha"]:
        full=normalize((1-alpha)*base+alpha*meta)
        # Disagreement with original baseline is necessary before a correction.
        # Margin advantage of the new predicted class is an unlabeled routing signal.
        original_label=base.argmax(1)
        candidate_label=full.argmax(1)
        candidate_margin=full[np.arange(len(y)),candidate_label] - full[np.arange(len(y)),original_label]
        for threshold in cfg["candidate_gate_threshold"]:
            accept=(candidate_label!=original_label)&(candidate_margin>threshold)
            adjusted=base.copy()
            adjusted[accept]=full[accept]
            m,_,_=metrics(y,adjusted,classes)
            acceptable=(m["accuracy"]>=bscore["accuracy"]-cfg["max_allowed_meta_gate_accuracy_loss"]
                        and m["macro_f1"]>=bscore["macro_f1"]+cfg["min_meta_gate_macro_f1_gain"]
                        and m["backdoor_TP"]>=bscore["backdoor_TP"]
                        and m["analysis_TP"]>=bscore["analysis_TP"])
            records.append({"alpha":float(alpha),"threshold":float(threshold),
                "accepted_count":int(accept.sum()),"admissible":bool(acceptable),
                **m})
    viable=[r for r in records if r["admissible"]]
    chosen=(max(viable,key=lambda r:(r["macro_f1"],r["accuracy"]))
            if viable else {"alpha":0.0,"threshold":1.0,
                             "accepted_count":0,"admissible":False,
                             "fallback_to_parent":True,**bscore})
    return chosen,records,bscore

def gated(base,meta,alpha,threshold):
    base=normalize(base);meta=normalize(meta)
    mixture=normalize((1-alpha)*base+alpha*meta)
    a=base.argmax(1);b=mixture.argmax(1)
    margin=mixture[np.arange(len(base)),b]-mixture[np.arange(len(base)),a]
    accept=(a!=b)&(margin>threshold)
    out=base.copy()
    out[accept]=mixture[accept]
    return out,accept

def restore_parent(folder,cfg,y):
    data={}
    for model,expected in (("lgbm_unweighted",cfg["frozen_standard_predictions_sha256"]),
                            ("lgbm_rare_sqrt",cfg["frozen_rare_predictions_sha256"])):
        source=folder/model/"predictions_PRIVATE.npz"
        saved=json.loads((folder/model/"metrics.json").read_text())
        if hash_file(source)!=expected or saved["private_predictions_sha256"]!=expected:
            raise ValueError("Pinned 35,005-example parent probability archive has changed")
        with np.load(source,allow_pickle=False) as arr:
            if not np.array_equal(arr["y_true"].astype(int),y):
                raise ValueError("Parent labels are not original frozen validation labels")
            data[model]=normalize(arr["probabilities"])
    return data

def fit_fold(X,y,xt,index_train,index_hold,cat,params,rounds,weights,save):
    yy=y[index_train]
    ds=lgb.Dataset(X.iloc[index_train],label=yy,weight=(weights[yy] if weights is not None else None),
                   categorical_feature=cat,free_raw_data=False)
    model=lgb.train(params,ds,num_boost_round=rounds)
    model.save_model(str(save))
    p=normalize(model.predict(X.iloc[index_hold],num_threads=4))
    if p.shape!=(len(index_hold),10):raise ValueError("Fold class ordering mismatch")
    return p

def main(args):
    repo,ref,baseline,out=map(Path,[args.repo,args.reference,args.baseline,args.output])
    cfg=json.loads(Path(args.config).read_text())
    if cfg.get("official_test")!="FORBIDDEN":raise ValueError("Official test locked")
    source=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
    X,y,Xv,yv,classes,cats,lineage=prepare(source,ref)
    if len(Xv)!=cfg["reference_validation_examples"] or len(X)!=140328:
        raise ValueError("Different target partition")
    parent=restore_parent(baseline,cfg,yv)
    rare_val=parent["lgbm_rare_sqrt"];std_val=parent["lgbm_unweighted"]
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/"config_effective.json",cfg)
    save_json(out/"data_lineage.json",lineage)
    (out/"source_commit.txt").write_text(subprocess.check_output(
        ["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()+"\n")
    y=np.asarray(y,dtype=np.int64)
    kfold=StratifiedKFold(n_splits=cfg["folds"],shuffle=True,random_state=cfg["seed"])
    oof={key:np.full((len(y),10),np.nan,dtype=np.float64)
         for key in ("standard","rare")}
    n=np.bincount(y,minlength=10)
    weights=np.minimum(12.,np.sqrt(n.max()/n)).astype(np.float64)
    params={"objective":"multiclass","num_class":10,"metric":"multi_logloss",
            "learning_rate":cfg["fold_learning_rate"],
            "num_leaves":cfg["fold_num_leaves"],
            "min_data_in_leaf":cfg["fold_min_data_in_leaf"],
            "lambda_l2":4,"feature_fraction":0.9,
            "seed":cfg["seed"],"deterministic":True,"force_col_wise":True,
            "num_threads":cfg["fold_thread_count"],"verbosity":-1}
    for fold,(fit_ids,hold_ids) in enumerate(kfold.split(X,y),start=1):
        fold_dir=out/f"fold_{fold}";fold_dir.mkdir(exist_ok=True)
        for name,weight in (("standard",None),("rare",weights)):
            checkpoint=fold_dir/f"{name}.txt"
            npz=fold_dir/f"{name}_OOF_PRIVATE.npz"
            if checkpoint.is_file() and npz.is_file():
                with np.load(npz,allow_pickle=False) as arr:
                    if not np.array_equal(arr["heldout_indices"],hold_ids):
                        raise ValueError("Resume fold membership mismatch")
                    saved=normalize(arr["probabilities"])
                print("REUSED FROZEN OOF:",fold,name,flush=True)
            else:
                yy=y[fit_ids]
                ds=lgb.Dataset(X.iloc[fit_ids],label=yy,
                        weight=weight[yy] if weight is not None else None,
                        categorical_feature=cats)
                model=lgb.train({**params,"seed":cfg["seed"]+fold*11},
                                 ds,num_boost_round=cfg["fold_rounds"])
                model.save_model(str(checkpoint))
                saved=normalize(model.predict(X.iloc[hold_ids],num_threads=4))
                np.savez_compressed(npz,heldout_indices=hold_ids.astype(np.int32),
                                    probabilities=saved.astype(np.float32))
                print("TRAINED OOF FOLD:",fold,name,flush=True)
            oof[name][hold_ids]=saved
    if not all(np.isfinite(p).all() for p in oof.values()):
        raise ValueError("Unfilled out-of-fold predictions")
    oof_file=out/"train_OOF_probabilities_PRIVATE.npz"
    np.savez_compressed(oof_file,y_true=y.astype(np.uint8),
                        standard=oof["standard"].astype(np.float32),
                        rare=oof["rare"].astype(np.float32))
    dev_ids,gate_ids=train_test_split(np.arange(len(y)),
        test_size=1-cfg["meta_fit_fraction"],random_state=cfg["seed"]+77,stratify=y)
    trX=meta_features(oof["standard"],oof["rare"])
    scaler=StandardScaler()
    Z=scaler.fit_transform(trX[dev_ids])
    fit=LogisticRegression(C=cfg["meta_regularization_C"],
            max_iter=cfg["meta_max_iter"],solver="lbfgs",
            class_weight=None)
    fit.fit(Z,y[dev_ids])
    if not np.array_equal(fit.classes_,np.arange(10)):
        raise ValueError("Missing classes from OOF meta-training fold")
    calibrated_gate=normalize(fit.predict_proba(scaler.transform(trX[gate_ids])))
    selected,candidates,gate_baseline=find_gate(
        y[gate_ids],oof["rare"][gate_ids],calibrated_gate,classes,cfg)
    joblib.dump({"scaler":scaler,"multiclass_meta":fit,
                 "train_indices_sha256":hashlib.sha256(dev_ids.astype("<i8").tobytes()).hexdigest(),
                 "gate_indices_sha256":hashlib.sha256(gate_ids.astype("<i8").tobytes()).hexdigest()},
                out/"meta_model_PRIVATE.joblib")
    save_json(out/"gate_selection.json",{
        "selected":selected,"baseline_gate_subset":gate_baseline,
        "candidate_gate_selection":candidates,
        "selected_without_validation_label_access":True,
        "train_count":len(dev_ids),"gate_count":len(gate_ids),
        "primary_model":"lgbm_rare_sqrt"})
    pd.DataFrame(candidates).to_csv(out/"gate_search_training_only.csv",index=False)
    test_features=meta_features(std_val,rare_val)
    meta_val=normalize(fit.predict_proba(scaler.transform(test_features)))
    candidate_map={
        "frozen_base":rare_val,
        "simple_average":normalize(.5*rare_val+.5*std_val),
        "meta_full":meta_val}
    output,accepted=gated(rare_val,meta_val,selected["alpha"],selected["threshold"])
    candidate_map["risk_gated"]=output
    scores=[]
    for name,p in candidate_map.items():
        directory=out/name;directory.mkdir(exist_ok=True)
        res,cm,per=metrics(yv,p,classes)
        res.update({"variant":name,"official_test_used":False,
                    "changed_from_rare_parent":int(np.sum(p.argmax(1)!=rare_val.argmax(1))),
                    "gate_accepted_validation_records":int(accepted.sum()) if name=="risk_gated" else 0})
        np.savez_compressed(directory/"predictions_PRIVATE.npz",
            y_true=np.asarray(yv,dtype=np.uint8),
            probabilities=p.astype(np.float32),y_pred=p.argmax(1).astype(np.uint8))
        res["predictions_sha256"]=hash_file(directory/"predictions_PRIVATE.npz")
        pd.DataFrame(cm,index=classes,columns=classes).rename_axis("true_class").to_csv(directory/"confusion.csv")
        per.to_csv(directory/"per_class.csv",index=False)
        save_json(directory/"metrics.json",res)
        scores.append(res)
        print(f"CARES-OOF {name:17} accuracy={100*res['accuracy']:.3f}% "
              f"macro-F1={100*res['macro_f1']:.3f}% "
              f"Backdoor TP={res['backdoor_TP']} "
              f"Analysis TP={res['analysis_TP']}",flush=True)
    pd.DataFrame(scores).to_csv(out/"comparison_scores.csv",index=False)
    save_json(out/"summary.json",{
        "experiment_id":cfg["experiment_id"],"scope":"train-derived validation, 35,005",
        "variants":{r["variant"]:r for r in scores},
        "gate":selected,"out_of_fold_prediction_hash":hash_file(oof_file),
        "source_validation_untouched_for_meta_training":True,
        "official_test_used":False,
        "note":"Redesigned multiclass CARES prototype; true novelty not established by architecture alone"})
    print("\nCARES-OOF-002 PROPOSED METHODOLOGY COMPLETE")
    print(pd.DataFrame(scores)[["variant","accuracy","macro_f1","balanced_accuracy",
        "backdoor_TP","analysis_TP","dos_TP"]].to_string(index=False))
    print("TRAIN-ONLY GATE:",selected)
    print("ALL TRAINED OOF MODELS, PRIVATE PREDICTIONS AND SCORES SAVED IN:",out)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("repo","reference","baseline","config","output"):
        parser.add_argument("--"+key,required=True)
    main(parser.parse_args())
