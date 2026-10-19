"""E6 order-robustness experiment. Original UNSW TRAIN CSV only.

New training-derived grouped split, five predeclared LightGBM controls,
nonoverlap target evaluation, no test-set access, no validation-driven tuning.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,confusion_matrix,
    f1_score,precision_recall_fscore_support)
from tqdm.auto import tqdm

from rareids.baseline_utils import (canonical_group_ids,
    group_disjoint_stratified_split,split_indices_sha256)
from rareids.tabular_eval import class_weights

if __package__:
    from scripts.run_window_e2 import build_split_windows,digest,store
    from scripts.run_window_e3 import append_stats
else:
    from run_window_e2 import build_split_windows,digest,store
    from run_window_e3 import append_stats

NAMES=("Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Normal",
       "Reconnaissance","Shellcode","Worms")
CATS=("proto","service","state")
CONDITIONS=("target_full","intact_full","shuffled_full",
            "target_nonoverlap","intact_nonoverlap")
PRIVATE_PREDICTIONS="predictions_PRIVATE.npz"

def map_labels(frame):
    y=frame["attack_cat"].astype("string").str.strip().map(
        {name:i for i,name in enumerate(NAMES)})
    if y.isna().any():
        raise ValueError("Unknown class in TRAIN partition")
    return y.to_numpy(dtype=np.int64)

def make_raw(partition,raw_cols,levels=None):
    out=build_split_windows(partition,raw_cols,CATS,5)
    names=[f"lag{lag}__{cat}" for lag in range(4,-1,-1) for cat in CATS if cat in raw_cols]
    vocabulary={}
    for name in names:
        if levels is None:
            vocab=pd.Index(out[name].unique()).sort_values().tolist()
        else:
            vocab=levels[name]
        out[name]=pd.Categorical(out[name],categories=vocab)
        vocabulary[name]=vocab
    return out,vocabulary,names

def history_shuffle(frame,raw_cols,seed):
    """Single shared permutation for ALL 4 historical rows, split-local only."""
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(frame))
    out=frame.copy()
    for lag in (4,3,2,1):
        for name in raw_cols:
            field=f"lag{lag}__{name}"
            out[field]=frame[field].array.take(perm)
    for col in raw_cols:
        if not out[f"lag0__{col}"].equals(frame[f"lag0__{col}"]):
            raise ValueError("Shuffled target features differ")
    return out

def metrics(y,p):
    predictions=np.asarray(p).argmax(1)
    cm=confusion_matrix(y,predictions,labels=np.arange(10))
    prec,rec,f1,support=precision_recall_fscore_support(
        y,predictions,labels=np.arange(10),zero_division=0)
    report={"n":int(len(y)),
        "accuracy":float(accuracy_score(y,predictions)),
        "macro_f1":float(f1_score(y,predictions,labels=np.arange(10),average="macro",zero_division=0)),
        "weighted_f1":float(f1_score(y,predictions,labels=np.arange(10),average="weighted",zero_division=0)),
        "precision_macro":float(np.mean(prec)),
        "recall_macro":float(np.mean(rec)),
        "dos_recall":float(rec[NAMES.index("DoS")]),
        "analysis_recall":float(rec[NAMES.index("Analysis")]),
        "backdoor_recall":float(rec[NAMES.index("Backdoor")]),
        "DoS_to_Exploits":int(cm[NAMES.index("DoS"),NAMES.index("Exploits")]),
        "Analysis_to_DoS":int(cm[NAMES.index("Analysis"),NAMES.index("DoS")]),
        "Backdoor_to_DoS":int(cm[NAMES.index("Backdoor"),NAMES.index("DoS")]),
        "correct":int(np.trace(cm))}
    per=pd.DataFrame({"class":NAMES,"support":support.astype(int),
        "precision":prec,"recall":rec,"f1":f1})
    return report,cm,per,predictions

def groups_for_frame(frame,features,labels,cfg):
    groups=canonical_group_ids(frame,features)
    tr,va,fold=group_disjoint_stratified_split(
        labels,groups,n_splits=cfg["n_splits"],seed=cfg["split_seed"])
    if len(np.intersect1d(groups[tr],groups[va]))!=0:
        raise ValueError("Identical-feature leakage across alternative split")
    if (not np.all(np.diff(tr)>0)) or (not np.all(np.diff(va)>0)):
        raise ValueError("Split row ordering must remain original CSV ordering")
    if len(tr)<1000 or len(va)<1000 or len(set(labels[tr]))!=10 or len(set(labels[va]))!=10:
        raise ValueError("New development split missing class support")
    return tr,va,fold

def partitions(cfg,root):
    raw=Path(cfg["training_path"])
    if raw.name!="UNSW_NB15_training-set.csv" or not raw.is_file():
        raise ValueError("Only original source TRAIN CSV may be accessed")
    if digest(raw)!=cfg["training_csv_sha256"]:
        raise ValueError("Source TRAIN sha256 mismatch")
    manifest=json.loads((root/cfg["feature_manifest_repo_path"]).read_text())
    cols=[x.removeprefix("lag0__") for x in manifest["raw_window_feature_names"]
          if x.startswith("lag0__")]
    if len(cols)!=42 or len(set(cols))!=42:
        raise ValueError("Frozen feature manifest not 42 traffic features")
    numeric=[x for x in cols if x not in CATS]
    if len(numeric)!=39:raise ValueError("Expected 39 numeric fields")
    frame=pd.read_csv(raw,usecols=list(cols)+["attack_cat"],low_memory=False)
    if len(frame)!=175341:raise ValueError("Original train rows changed")
    y=map_labels(frame)
    tr,va,fold=groups_for_frame(frame,cols,y,cfg)
    print("New train-derived group-disjoint split, fold",fold,flush=True)
    train=frame.iloc[tr].reset_index(drop=True)
    validation=frame.iloc[va].reset_index(drop=True)
    raw_tr,vocab,cats=make_raw(train,cols)
    raw_va,_,check_cats=make_raw(validation,cols,levels=vocab)
    if cats!=check_cats:raise ValueError("Categorical feature order mismatch")
    raw_features=manifest["raw_window_feature_names"]
    if list(raw_tr.columns)!=raw_features or list(raw_va.columns)!=raw_features:
        raise ValueError("Feature engineering schema drift")
    return (raw_tr,raw_va,y[tr][4:],y[va][4:],cats,cols,numeric,
            tr,va,fold,vocab,manifest,y[va])

def make_context(raw,nums,manifest):
    result,names=append_stats(raw,nums)
    if names!=manifest["statistic_feature_names"] or result.shape[1]!=444:
        raise ValueError("E6 444-feature context schema != frozen E3")
    return result

def subgroup_scores(y,p,masks):
    rows=[]
    for subset,selected in masks.items():
        if not np.asarray(selected).any():
            continue
        r,_,_,_=metrics(y[selected],p[selected])
        rows.append({"subset":subset,**r})
    return rows

def fit_and_record(name,cfg,output,train_x,y,validation_x,yval,
                   categories,nonoverlap_idx,masks):
    folder=output/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"metrics.json").exists():
        raise RuntimeError("Model metrics already exist. Restore/verify using the archival helper.")
    include_train=(np.arange(len(y))%5==0 if name.endswith("nonoverlap")
                   else np.ones(len(y),dtype=bool))
    xx=train_x.iloc[include_train] if name.endswith("nonoverlap") else train_x
    yy=y[include_train]
    if len(np.unique(yy))!=10:
        raise ValueError("Nonoverlap training set lost rare class: cannot silently change selection")
    weights=class_weights(yy,cfg["class_weight_power"],cfg["max_class_weight"])
    print(f"Training {name}: {len(xx)} target windows, {len(xx.columns)} features, "
          f"fixed {cfg['fixed_num_boost_round']} rounds",flush=True)
    ds=lgb.Dataset(xx,label=yy,weight=weights[yy],
        feature_name=list(xx.columns),categorical_feature=categories,
        free_raw_data=False)
    params={**cfg["frozen_model_hyperparameters"],
            "seed":cfg["model_seed"],"num_threads":max(2,min(8,os.cpu_count() or 4))}
    start=time.monotonic()
    with tqdm(total=cfg["fixed_num_boost_round"],
              desc="E6 "+name,unit="boosting round") as bar:
        def progress(env):
            bar.update(max(0,env.iteration+1-bar.n))
        progress.order=85
        model=lgb.train(params,ds,num_boost_round=cfg["fixed_num_boost_round"],
                        callbacks=[progress])
    model_file=folder/"model.txt"
    model.save_model(str(model_file),num_iteration=cfg["fixed_num_boost_round"])
    p=np.asarray(model.predict(validation_x),dtype=np.float32)
    if p.shape!=(len(yval),10):
        raise ValueError("Fitted model returned wrong validation shape")
    raw_score,cm,per,pred=metrics(yval,p)
    sampled,_,_,_=metrics(yval[nonoverlap_idx],p[nonoverlap_idx])
    report={**raw_score,
        "candidate":name,"experiment_id":cfg["experiment_id"],
        "nonoverlap":sampled,
        "training_windows":int(len(yy)),
        "evaluation_full_windows":int(len(yval)),
        "evaluation_disjoint_windows":int(len(nonoverlap_idx)),
        "model_feature_count":int(xx.shape[1]),
        "fixed_iterations":int(cfg["fixed_num_boost_round"]),
        "official_test_accessed":False,"test_used_for_tuning":False,
        "validation_used_for_early_stopping":False,
        "fit_seconds":round(time.monotonic()-start,3)}
    score_file=folder/"metrics.json"
    pd.DataFrame(cm,index=NAMES,columns=NAMES).rename_axis("actual_class").to_csv(
        folder/"confusion.csv")
    per.to_csv(folder/"per_class.csv",index=False)
    strata=subgroup_scores(yval,p,masks)
    pd.DataFrame(strata).to_csv(folder/"subgroups.csv",index=False)
    gains=pd.DataFrame({"feature":xx.columns,
        "gain":model.feature_importance(importance_type="gain"),
        "split_count":model.feature_importance(importance_type="split")})
    gains.sort_values("gain",ascending=False).head(40).to_csv(
        folder/"top40_features.csv",index=False)
    np.savez_compressed(folder/PRIVATE_PREDICTIONS,
        probabilities=p,y_true=yval.astype(np.uint8),
        y_pred=pred.astype(np.uint8),
        nonoverlap_validation_indices=nonoverlap_idx.astype(np.int32))
    report["model_sha256"]=digest(model_file)
    report["predictions_sha256"]=digest(folder/PRIVATE_PREDICTIONS)
    store(score_file,report)
    print(f"FINISHED {name}: FULL accuracy={report['accuracy']:.5f}, "
          f"macro-F1={report['macro_f1']:.5f} | NONOVERLAP accuracy="
          f"{sampled['accuracy']:.5f}, macro-F1={sampled['macro_f1']:.5f}",flush=True)
    return report

def run(args):
    root,output=Path(args.repo),Path(args.output)
    cfg=json.loads(Path(args.config).read_text())
    if (cfg["experiment_id"]!="E6-ORDER-ROBUSTNESS-001" or
        cfg["official_test"]!="FORBIDDEN" or
        cfg["fixed_num_boost_round"]!=58 or
        cfg["no_hyperparameter_selection_from_e6_validation"] is not True):
        raise ValueError("Prerecorded E6 train-only protocol changed")
    cfg["training_path"]=str(Path(args.train_csv))
    output.mkdir(parents=True,exist_ok=True)
    raw_x,raw_v,y,yv,categories,cols,numeric,tr,va,fold,vocab,manifest,va_source_labels=partitions(cfg,root)
    nonoverlap_idx=np.arange(0,len(yv),5,dtype=np.int64)
    if len(nonoverlap_idx)<1000 or len(np.unique(yv[nonoverlap_idx]))!=10:
        raise ValueError("Disjoint val sample excludes a class; stop instead of selecting different targets")
    # Independent, objective alternative split: no E6 model metrics inspected when selected.
    split_hash=split_indices_sha256(tr,va)
    np.savez_compressed(output/"split_indices_PRIVATE.npz",
        train_indices=tr.astype(np.int32),val_indices=va.astype(np.int32),
        y_val=yv.astype(np.uint8),nonoverlap_indices=nonoverlap_idx.astype(np.int32))
    store(output/"config_effective.json",cfg)
    val_purity=np.all(np.lib.stride_tricks.sliding_window_view(
        va_source_labels,5)[:,:-1]==yv[:,None],axis=1)
    oldest_gap=va[4:]-va[:-4]
    masks={
        "all":np.ones(len(yv),dtype=bool),
        "nonoverlap":np.isin(np.arange(len(yv)),nonoverlap_idx),
        "class_pure_history":val_purity,
        "mixed_class_history":~val_purity,
        "source_oldest_gap_ge_10":oldest_gap>=10,
        "source_oldest_gap_lt_10":oldest_gap<10}
    store(output/"split_provenance.json",{
        "source_train_csv_sha256":cfg["training_csv_sha256"],
        "alternative_train_dev_split_sha256":split_hash,
        "split_seed":cfg["split_seed"],"selected_fold":int(fold),
        "train_source_rows":int(len(tr)),"val_source_rows":int(len(va)),
        "train_windows":int(len(y)),"val_windows":int(len(yv)),
        "nonoverlap_validation_windows":int(len(nonoverlap_idx)),
        "nonoverlap_training_windows":int(len(np.arange(0,len(y),5))),
        "train_classes":{n:int((y==i).sum()) for i,n in enumerate(NAMES)},
        "val_classes":{n:int((yv==i).sum()) for i,n in enumerate(NAMES)},
        "validation_strata_counts":{k:int(mask.sum()) for k,mask in masks.items()},
        "same_source_duplicate_feature_group_across_split":False,
        "official_test_opened":False,"verified_source_is_temporal":False,
        "only_original_training_csv_opened":True})
    store(output/"model_feature_manifest.json",{
        "target_features":[f"lag0__{col}" for col in cols],
        "intact_features":manifest["raw_window_feature_names"]+manifest["statistic_feature_names"],
        "training_only_category_vocabulary":True})
    (output/"source_code_commit.txt").write_text(
        subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                                text=True).strip()+"\n")
    (output/"environment.txt").write_text(
        f"Python {platform.python_version()} LightGBM {lgb.__version__}; "
        f"numpy {np.__version__}; pandas {pd.__version__}\n")
    # Materialize exact matched five-row representations once.
    intact_train=make_context(raw_x,numeric,manifest)
    intact_val=make_context(raw_v,numeric,manifest)
    target_features=[f"lag0__{col}" for col in cols]
    target_train=raw_x[target_features].copy()
    target_val=raw_v[target_features].copy()
    if (list(intact_train.columns)!=manifest["raw_window_feature_names"]+
        manifest["statistic_feature_names"]):
        raise ValueError("Frozen learned feature ordering changed")
    representations={
        "target_full":(target_train,target_val),
        "intact_full":(intact_train,intact_val),
        "target_nonoverlap":(target_train,target_val),
        "intact_nonoverlap":(intact_train,intact_val)}
    records={}
    for name in CONDITIONS:
        if name=="shuffled_full":
            print("Creating label-independent, jointly permuted history controls...",flush=True)
            train=make_context(history_shuffle(raw_x,cols,
                cfg["label_independent_historical_block_shuffle_seed_train"]),numeric,manifest)
            val=make_context(history_shuffle(raw_v,cols,
                cfg["label_independent_historical_block_shuffle_seed_validation"]),numeric,manifest)
        else:
            train,val=representations[name]
        train_cats=[x for x in categories if x in train.columns]
        from archive_e6_order import restore_private_candidate,archive_candidate
        token=os.environ.get("GITHUB_TOKEN")
        reused=restore_private_candidate(output,name,token)
        if reused:
            with (output/name/"metrics.json").open() as f:report=json.load(f)
        else:
            report=fit_and_record(name,cfg,output,train,y,val,yv,
                train_cats,nonoverlap_idx,masks)
            if args.auto_archive:
                archive_candidate(output,name,token)
        records[name]=report
        store(output/"progress_public.json",{"experiment_id":cfg["experiment_id"],
            "completed":list(records),
            "official_test_opened":False})
        print("SECURED E6:",name,flush=True)
        del train,val
    overall=pd.DataFrame([{"model":name,"n":v["n"],"accuracy":v["accuracy"],
        "macro_f1":v["macro_f1"],"weighted_f1":v["weighted_f1"],
        "nonoverlap_accuracy":v["nonoverlap"]["accuracy"],
        "nonoverlap_macro_f1":v["nonoverlap"]["macro_f1"],
        "dos_recall":v["dos_recall"],"analysis_recall":v["analysis_recall"],
        "backdoor_recall":v["backdoor_recall"]} for name,v in records.items()])
    overall.to_csv(output/"comparison_all_models.csv",index=False)
    pair_rows=[]
    for a,b,subset in [
      ("intact_full","target_full","all"),
      ("intact_full","shuffled_full","all"),
      ("intact_full","target_full","nonoverlap"),
      ("intact_full","shuffled_full","nonoverlap"),
      ("intact_nonoverlap","target_nonoverlap","nonoverlap"),
      ("intact_nonoverlap","intact_full","nonoverlap")]:
        a_result=records[a] if subset=="all" else records[a]["nonoverlap"]
        b_result=records[b] if subset=="all" else records[b]["nonoverlap"]
        pair_rows.append({"candidate":a,"control":b,"subset":subset,
            "accuracy_delta":float(a_result["accuracy"]-b_result["accuracy"]),
            "macro_f1_delta":float(a_result["macro_f1"]-b_result["macro_f1"]),
            "n":int(a_result["n"])})
    pd.DataFrame(pair_rows).to_csv(output/"paired_comparisons.csv",index=False)
    store(output/"summary.json",{
        "experiment_id":cfg["experiment_id"],
        "official_test_opened":False,"test_was_not_reused":True,
        "source_training_sha256":cfg["training_csv_sha256"],
        "alternate_group_disjoint_split_sha256":split_hash,
        "fitted_new_models":5,
        "all_validation":{k:{"accuracy":v["accuracy"],"macro_f1":v["macro_f1"]}
                          for k,v in records.items()},
        "nonoverlap_validation":{k:v["nonoverlap"] for k,v in records.items()},
        "paired_control_differences":pair_rows,
        "conclusion_limitations":[
            "Label-dependent class clustering is present in original source CSV.",
            "Split-local five-row record position is not verified time or network flow.",
            "Block shuffle disturbs natural feature distributions; not a causal proof.",
            "Nonoverlap within evaluation partitions reduces shared-source-row overlap but not flow or domain dependence.",
            "One alternative train-derived split does not constitute external validation.",
            "Official test already used in earlier frozen phase and NOT reopened here."
        ]})
    if args.auto_archive:
        from archive_e6_order import publish_final_e6
        publish_final_e6(output,os.environ.get("GITHUB_TOKEN"))
    print("\nE6 ORDER-ROBUSTNESS COMPLETE — 5 PRIVATE MODEL BACKUPS + PUBLIC AGGREGATES",flush=True)
    print(overall.to_string(index=False),flush=True)
    print((output/"paired_comparisons.csv").read_text(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("repo","config","train-csv","output"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--auto-archive",action="store_true")
    run(p.parse_args())
