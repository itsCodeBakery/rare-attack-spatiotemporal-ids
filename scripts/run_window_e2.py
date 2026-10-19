"""WINDOW-E2-001: exactly matched 5-record raw-window LightGBM ablation.

Use the original SCV split-local sliding-window *position*, NOT inferred time.
Training partition is the sole CSV opened; frozen test stays inaccessible.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,
    confusion_matrix,f1_score,precision_recall_fscore_support)
from tqdm.auto import tqdm

from rareids.tabular_eval import macro_eval,class_weights

BASELINE_IDS=("lgbm_unweighted","lgbm_rare_sqrt")

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(4194304),b""):h.update(block)
    return h.hexdigest()

def store(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def verify_frozen_reference(raw,ref,cfg,base_cfg):
    if raw.name!="UNSW_NB15_training-set.csv":
        raise ValueError("Only the designated original TRAINING file can be read")
    for k in ("train_csv_sha256","reference_split_sha256","reference_pred_sha256"):
        if cfg[k]!=base_cfg[k]:
            raise ValueError("Pre-declared hashes must equal frozen baseline")
    if digest(raw)!=cfg["train_csv_sha256"]:
        raise ValueError("TRAINING source hash mismatch")
    if digest(ref/"split_indices.npz")!=cfg["reference_split_sha256"]:
        raise ValueError("Original train/validation split hash mismatch")
    if digest(ref/"validation_predictions.npz")!=cfg["reference_pred_sha256"]:
        raise ValueError("Original saved validation predictions hash mismatch")
    with np.load(ref/"split_indices.npz",allow_pickle=False) as d:
        itr=d["dev_train_indices"].astype(np.int64)
        iva=d["validation_indices"].astype(np.int64)
        ytr=d["dev_train_labels"].astype(np.int64)
        yva=d["validation_labels"].astype(np.int64)
        names=[str(x) for x in d["classes"].tolist()]
    with np.load(ref/"validation_predictions.npz",allow_pickle=False) as d:
        original_y=d["y_true"].astype(np.int64)
        original_targets=d["validation_window_end_indices"].astype(np.int64)
    prep=json.loads((ref/"preprocessing.json").read_text(encoding="utf-8"))
    cols=prep["columns"]
    cats=prep["categorical"]
    if len(names)!=10 or len(cols)!=42 or len(cats)!=3:
        raise ValueError("Original 10 classes, 42 columns or categoricals changed")
    if set(cats)!={"proto","service","state"}:
        raise ValueError("Expected original categorical columns")
    if set(cols)&{"id","label","attack_cat","attack_category"}:
        raise ValueError("Label/id leakage in input features")
    if len(itr)!=140332 or len(iva)!=35009:
        raise ValueError("Frozen split lengths changed")
    if np.intersect1d(itr,iva).size:
        raise ValueError("Training and validation indices overlap")
    if (not np.array_equal(iva[4:],original_targets) or
        not np.array_equal(yva[4:],original_y)):
        raise ValueError("Window target sequence differs from original SCV")
    return itr,iva,ytr,yva,original_y,names,cols,cats,original_targets

def build_split_windows(partition,cols,cats,window):
    """Target offset window-1, same split-local positions as original SCV."""
    n=len(partition)-window+1
    if n<=0:raise ValueError("Not enough rows for five-record windows")
    # position 0 oldest, 4 newest; lag 4 -> lag 0.
    # Explicit shift inside the ORIGINAL split-local row sequence only.
    out={}
    for position in range(window):
        lag=window-position-1
        part=partition.iloc[position:position+n]
        for col in cols:
            name=f"lag{lag}__{col}"
            if col in cats:
                out[name]=part[col].astype("string").fillna("__MISSING__").astype(str).to_numpy()
            else:
                v=pd.to_numeric(part[col],errors="coerce").to_numpy(dtype=np.float32)
                v[~np.isfinite(v)]=np.nan
                out[name]=v
    return pd.DataFrame(out)

def build_data(cfg,base_cfg,ref):
    raw=Path(base_cfg["train_csv"])
    if not raw.is_file():raise FileNotFoundError(f"Please attach Kaggle UNSW train dataset: {raw}")
    itr,iva,ytr,yva,orig_y,names,cols,cats,orig_targets=verify_frozen_reference(
        raw,ref,cfg,base_cfg)
    # Read *only* designated training CSV. No test set is opened or referenced.
    frame=pd.read_csv(raw,usecols=list(cols)+["attack_cat"],low_memory=False)
    if len(frame)!=175341:
        raise ValueError("Training CSV unexpectedly changed")
    mapped=frame["attack_cat"].astype(str).str.strip().map({c:i for i,c in enumerate(names)})
    all_labels=mapped.to_numpy()
    if (not np.array_equal(all_labels[itr],ytr) or
        not np.array_equal(all_labels[iva],yva)):
        raise ValueError("Original SCV split labels not reproduced")
    tr=frame.iloc[itr].reset_index(drop=True)
    va=frame.iloc[iva].reset_index(drop=True)
    del frame
    print("Build exact split-local five-record windows...",flush=True)
    X=build_split_windows(tr,cols,cats,cfg["window_size"])
    V=build_split_windows(va,cols,cats,cfg["window_size"])
    del tr,va
    if len(X)!=cfg["train_target_rows"] or len(V)!=cfg["validation_target_rows"]:
        raise ValueError("Frozen target count changed after feature expansion")
    if len(X.columns)!=cfg["expected_flat_features"] or list(X.columns)!=list(V.columns):
        raise ValueError("Expected five-times-42 exact ordered columns")
    categorical=[n for n in X.columns if n.rsplit("__",1)[-1] in cats]
    if len(categorical)!=15:raise ValueError("Expected 15 native categorical lag fields")
    for name in tqdm(categorical,desc="Fit training-only category levels",unit="field"):
        # Explicit ordered TRAINING vocabulary prevents validation category leakage.
        vocab=pd.Index(X[name].unique()).sort_values()
        X[name]=pd.Categorical(X[name],categories=vocab)
        V[name]=pd.Categorical(V[name],categories=vocab)
    y=ytr[4:]
    yv=yva[4:]
    if not np.array_equal(yv,orig_y):
        raise ValueError("Validation labels not exact original SCV window targets")
    lineage={"test_dataset_accessed":False,
        "train_target_rows":int(len(y)),"validation_target_rows":int(len(yv)),
        "raw_features":42,"window_positions":5,"flat_features":int(X.shape[1]),
        "categorical_flat_features":categorical,
        "train_target_index_sha256":hashlib.sha256(
            itr[4:].astype("<i8").tobytes()).hexdigest(),
        "validation_target_index_sha256":hashlib.sha256(
            orig_targets.astype("<i8").tobytes()).hexdigest(),
        "training_csv_sha256":cfg["train_csv_sha256"],
        "original_split_sha256":cfg["reference_split_sha256"],
        "original_validation_predictions_sha256":cfg["reference_pred_sha256"],
        "class_names":names,
        "window_order":"split-local CSV row order, not verified temporal",
        "only_designated_training_csv_accessed":True}
    if (X.columns[0]!=f"lag4__{cols[0]}" or X.columns[-1]!=f"lag0__{cols[-1]}"):
        raise ValueError("Oldest-to-target flat window feature ordering changed")
    return X,y,V,yv,names,categorical,lineage

def metrics(y,p,classes):
    labels=np.arange(10)
    pred=np.asarray(p).argmax(1)
    cm=confusion_matrix(y,pred,labels=labels)
    prec,rec,f1,support=precision_recall_fscore_support(
        y,pred,labels=labels,zero_division=0)
    summary={"count":len(y),"correct":int(np.trace(cm)),
        "accuracy":float(accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,labels=labels,average="macro",zero_division=0)),
        "weighted_f1":float(f1_score(y,pred,labels=labels,average="weighted",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "DoS_to_Exploits":int(cm[classes.index("DoS"),classes.index("Exploits")]),
        "Backdoor_to_Exploits":int(cm[classes.index("Backdoor"),classes.index("Exploits")]),
        "dos_recall":float(rec[classes.index("DoS")]),
        "analysis_recall":float(rec[classes.index("Analysis")]),
        "backdoor_recall":float(rec[classes.index("Backdoor")])}
    per=pd.DataFrame({"class":classes,"support":support,"precision":prec,
                      "recall":rec,"f1":f1})
    return summary,cm,per

def parent_scores(baseline,y,classes):
    results={}
    for key in BASELINE_IDS:
        folder=baseline/key
        checkpoint=folder/"model.txt"
        saved=json.loads((folder/"metrics.json").read_text())
        p=folder/"predictions_PRIVATE.npz"
        if digest(checkpoint)!=saved["checkpoint_sha256"] or digest(p)!=saved["private_predictions_sha256"]:
            raise ValueError("Parent frozen model archive checksum drift")
        with np.load(p,allow_pickle=False) as x:
            if not np.array_equal(x["y_true"].astype(np.int64),y):
                raise ValueError("Parent labels differ from exactly matching E2 targets")
            score,_,_=metrics(y,x["probabilities"],classes)
        if abs(score["accuracy"]-saved["accuracy"])>1e-12 or abs(score["macro_f1"]-saved["macro_f1"])>1e-12:
            raise ValueError("Frozen model metrics changed")
        results[key]={"model_family":"last-record-only_frozen_LightGBM",**score}
    return results

def train_model(run,model_cfg,run_cfg,X,y,V,yv,classes,categories):
    import lightgbm as lgb
    key=model_cfg["id"]
    directory=run/key;directory.mkdir(parents=True,exist_ok=True)
    scorefile=directory/"metrics.json"
    if scorefile.exists():
        saved=json.loads(scorefile.read_text())
        if (directory/"model.txt").is_file() and
            directory/"predictions_PRIVATE.npz").is_file() and
            digest(directory/"model.txt")==saved["checkpoint_sha256"] and
            digest(directory/"predictions_PRIVATE.npz")==saved["private_predictions_sha256"]):
            print(f"REUSED HASH-VERIFIED COMPLETED E2 MODEL: {key}",flush=True)
            return saved
        raise ValueError("Existing E2 model files mismatch; refusing unsafe overwrite")
    weights=class_weights(y,model_cfg["weight_power"],model_cfg.get("max_class_weight",12))
    dtrain=lgb.Dataset(X,label=y,weight=weights[y],
        feature_name=list(X.columns),categorical_feature=categories,free_raw_data=False)
    dval=lgb.Dataset(V,label=yv,reference=dtrain,
        feature_name=list(V.columns),categorical_feature=categories,free_raw_data=False)
    params={"objective":"multiclass","num_class":10,"metric":"None",
        "learning_rate":model_cfg["learning_rate"],"num_leaves":model_cfg["num_leaves"],
        "min_data_in_leaf":model_cfg["min_data_in_leaf"],
        "feature_fraction":0.85,"bagging_fraction":0.85,"bagging_freq":1,
        "lambda_l2":5,"seed":run_cfg["seed"],"verbosity":-1,
        "deterministic":True,"force_col_wise":True,
        "num_threads":max(2,min(8,os.cpu_count() or 4))}
    evaluations={}
    current={}
    log=directory/"training_progress.txt"
    print(f"\nTRAINING {key} — {len(X.columns)} window features (CPU)",flush=True)
    start=time.monotonic()
    with tqdm(total=model_cfg["num_boost_round"],desc=f"E2 {key} boosting",
              unit="iteration",dynamic_ncols=True) as bar:
        def progress(env):
            iteration=env.iteration+1
            bar.update(max(0,iteration-bar.n))
            if iteration%100==0:
                # Recoverable partial booster written to Kaggle every 100 rounds.
                partial=directory/"partial_progress_PRIVATE.txt"
                temp=directory/"partial_progress_PRIVATE.tmp"
                env.model.save_model(str(temp),num_iteration=iteration)
                temp.replace(partial)
                current["last_saved_iteration"]=iteration
                store(directory/"partial_metadata.json",current)
                with log.open("a") as stream:
                    stream.write(f"LOCAL CHECKPOINT: {iteration} rounds\n")
                print(f"LOCAL CHECKPOINT {key} at iteration {iteration}",flush=True)
        progress.order=85
        model=lgb.train(params,dtrain,num_boost_round=model_cfg["num_boost_round"],
            valid_sets=[dval],valid_names=["internal_validation"],feval=macro_eval,
            callbacks=[lgb.early_stopping(model_cfg["early_stopping_rounds"],verbose=False),
                       lgb.record_evaluation(evaluations),
                       lgb.log_evaluation(period=50),progress])
    probs=model.predict(V,num_iteration=model.best_iteration)
    score,cm,per=metrics(yv,probs,classes)
    score.update({"candidate":key,"best_iteration":int(model.best_iteration),
        "fit_elapsed_seconds":float(round(time.monotonic()-start,2)),
        "trained_models":1,"feature_count":len(X.columns),
        "window_size":5,"class_weights":weights.tolist(),
        "validation_used_for_early_stopping":True,"official_test_used":False})
    checkpoint=directory/"model.txt"
    model.save_model(str(checkpoint),num_iteration=model.best_iteration)
    np.savez_compressed(directory/"predictions_PRIVATE.npz",
        y_true=yv.astype(np.uint8),y_pred=probs.argmax(1).astype(np.uint8),
        probabilities=probs.astype(np.float32))
    score["checkpoint_sha256"]=digest(checkpoint)
    score["private_predictions_sha256"]=digest(directory/"predictions_PRIVATE.npz")
    pd.DataFrame(cm,index=classes,columns=classes).rename_axis("true_class").to_csv(
        directory/"confusion.csv")
    per.to_csv(directory/"per_class.csv",index=False)
    pd.DataFrame({"iteration":np.arange(1,len(evaluations["internal_validation"]["macro_f1"])+1),
        "internal_validation_macro_f1":evaluations["internal_validation"]["macro_f1"]}).to_csv(
        directory/"learning.csv",index=False)
    store(scorefile,score)
    print(f"COMPLETED {key}: accuracy={score['accuracy']:.5f} macro-F1={score['macro_f1']:.5f} "
          f"DoS-recall={score['dos_recall']:.5f}",flush=True)
    return score

def run(args):
    cfg=json.loads(Path(args.config).read_text())
    if cfg.get("official_test")!="FORBIDDEN":raise ValueError("Official test forbidden")
    repo,ref,baseline,output=(Path(args.repo),Path(args.reference),
                               Path(args.baseline),Path(args.output))
    base_cfg=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
    X,y,V,yv,classes,cats,lineage=build_data(cfg,base_cfg,ref)
    output.mkdir(parents=True,exist_ok=True)
    store(output/"config_effective.json",cfg)
    store(output/"data_lineage.json",lineage)
    store(output/"feature_manifest.json",{
        "feature_count":len(X.columns),"features":list(X.columns),
        "training_only_category_vocab":True,
        "feature_origin":"exact original split-local five-row serialized traffic windows",
        "source_order_is_verified_temporal":False})
    (output/"source_code_commit.txt").write_text(
        subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()+"\n")
    if not (output/"training_environment.txt").exists():
        import lightgbm as lgb
        (output/"training_environment.txt").write_text(
            f"Python {platform.python_version()}; LightGBM {lgb.__version__}; "
            f"pandas {pd.__version__}; numpy {np.__version__}\n")
    results=parent_scores(baseline,yv,classes)
    store(output/"control_comparison.json",results)
    for candidate in cfg["candidates"]:
        key=candidate["id"]
        # Restore an already-uploaded completed candidate from an immutable
        # private GitHub draft Release if this Kaggle session was restarted.
        if args.auto_archive:
            from archive_window_e2 import restore_prior_candidate
            restore_prior_candidate(output,key,os.environ.get("GITHUB_TOKEN"))
        score=train_model(output,candidate,cfg,X,y,V,yv,classes,cats)
        results[key]={"model_family":"five-record_native-categorical_LightGBM",**score}
        store(output/"results_so_far.json",{"experiment_id":cfg["experiment_id"],
            "official_test_used":False,"models":results})
        if args.auto_archive:
            from archive_window_e2 import archive_and_sync
            archive_and_sync(output,key,os.environ.get("GITHUB_TOKEN"))
        print(f"SAFE CHECKPOINT: {key} archived before next candidate",flush=True)
    compare=pd.DataFrame([{"model":k,**s} for k,s in results.items()])
    compare.to_csv(output/"comparison_all_models.csv",index=False)
    baseline_best=max([x for x in BASELINE_IDS],key=lambda k:results[k]["macro_f1"])
    e2_best=max([x["id"] for x in cfg["candidates"]],key=lambda k:results[k]["macro_f1"])
    summary={"experiment_id":cfg["experiment_id"],"models":results,
        "best_frozen_parent":baseline_best,"best_window_candidate":e2_best,
        "delta_accuracy_window_vs_matching_parent":float(
            results[e2_best]["accuracy"]-results[baseline_best]["accuracy"]),
        "delta_macro_f1_window_vs_matching_parent":float(
            results[e2_best]["macro_f1"]-results[baseline_best]["macro_f1"]),
        "official_test_used":False,
        "purpose":"E2 hypothesis screening, 35,005 internal validation examples; no published superiority claim"}
    store(output/"summary.json",summary)
    if args.auto_archive:
        from archive_window_e2 import sync_final_summary
        sync_final_summary(output,os.environ.get("GITHUB_TOKEN"))
    print("\nWINDOW-E2-001 FINAL TRAINING AND GITHUB BACKUPS COMPLETE",flush=True)
    print(compare[["model","accuracy","macro_f1","weighted_f1",
                    "dos_recall","analysis_recall","backdoor_recall"]].to_string(index=False),
          flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("repo","reference","baseline","config","output"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--auto-archive",action="store_true")
    run(parser.parse_args())
