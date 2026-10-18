"""CTRL-TAB-001: fast high-quality tabular IDS on SAME SCV validation targets.

Two LightGBM candidates: ordinary and square-root class balanced. No CNN, no
fictitious temporal assumptions, no official test, no dataset recomputed split.
Every completed candidate writes its real confusion matrix, model, private
probabilities, metrics, provenance and numeric figures before the next runs.
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
from sklearn.metrics import (f1_score,accuracy_score,balanced_accuracy_score,
                             precision_recall_fscore_support,confusion_matrix)
from tqdm.auto import tqdm

from rareids.tabular_eval import class_weights,macro_eval,score_predictions

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4194304),b""):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8")

def source_commit(repo):
    return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
        text=True,stderr=subprocess.DEVNULL,timeout=10).strip()

def prepare(cfg,ref):
    raw=Path(cfg["train_csv"]);ref=Path(ref)
    assert raw.is_file(),f"Training CSV missing: {raw}"
    if len(cfg["train_csv_sha256"])!=64:
        raise ValueError("Configuration error: training CSV SHA256 must contain 64 hex characters")
    if digest(raw)!=cfg["train_csv_sha256"]:
        raise ValueError("Official training CSV SHA256 differs from frozen SCV source")
    split=ref/"split_indices.npz"
    probs=ref/"validation_predictions.npz"
    if digest(split)!=cfg["reference_split_sha256"] or digest(probs)!=cfg["reference_pred_sha256"]:
        raise ValueError("Frozen original split/prediction SHA256 failed; NO NEW SPLIT permitted")
    with np.load(split,allow_pickle=False) as s:
        idx_tr=s["dev_train_indices"].astype(np.int64)
        idx_va=s["validation_indices"].astype(np.int64)
        y_tr=s["dev_train_labels"].astype(np.int64)
        y_va=s["validation_labels"].astype(np.int64)
        classes=[str(x) for x in s["classes"].tolist()]
    with np.load(probs,allow_pickle=False) as p:
        y_ref=p["y_true"].astype(np.int64)
        ends=p["validation_window_end_indices"].astype(np.int64)
    if len(classes)!=10 or len(idx_tr)!=140332 or len(idx_va)!=35009:
        raise ValueError("Unexpected original split dimensions")
    if not np.array_equal(y_va[4:],y_ref) or not np.array_equal(idx_va[4:],ends):
        raise ValueError("Actual original SCV validation window-target sequence differs")
    if len(y_ref)!=cfg["validation_target_rows"] or len(idx_tr[4:])!=cfg["train_target_rows"]:
        raise ValueError("Target lineage counts differ from locked benchmark")
    prep=json.loads((ref/"preprocessing.json").read_text(encoding="utf-8"))
    cols=prep["columns"]
    cats=prep["categorical"]
    if len(cols)!=cfg["features"] or set(cats)!=set(cfg["categorical_columns"]):
        raise ValueError("Unexpected original model's 42 raw feature schema")
    if set(cols)&{"id","label","attack_cat"}:
        raise ValueError("Refusing label, test, or record ID leakage into features")
    print("Only the designated TRAIN CSV is opened. Original validation target hash preserved.",flush=True)
    frame=pd.read_csv(raw,low_memory=False)
    if len(frame)!=175341 or not set(cols).issubset(frame.columns):
        raise ValueError("Unexpected UNSW source CSV rows/features")
    labels=frame["attack_cat"].astype(str).str.strip().map(
        {c:i for i,c in enumerate(classes)}).to_numpy()
    if not np.array_equal(labels[idx_tr],y_tr) or not np.array_equal(labels[idx_va],y_va):
        raise ValueError("Original labels do not match saved train/validation membership")
    # Only last-record features of exactly the same supervised five-row image windows.
    tr=frame.iloc[idx_tr[4:]][cols].copy()
    va=frame.iloc[idx_va[4:]][cols].copy()
    del frame
    for col in cols:
        if col in cats:
            def string_values(x):
                return x.astype("string").fillna("__MISSING__").astype(str)
            x=string_values(tr[col]);v=string_values(va[col])
            unique=pd.Index(x.unique()).sort_values()
            tr[col]=pd.Categorical(x,categories=unique)
            va[col]=pd.Categorical(v,categories=unique)
        else:
            tr[col]=pd.to_numeric(tr[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
            va[col]=pd.to_numeric(va[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
    if list(tr.columns)!=list(va.columns):raise ValueError("Category and feature order mismatch")
    note={"exact_train_target_indices_sha256":hashlib.sha256(
        np.ascontiguousarray(idx_tr[4:].astype("<i8")).tobytes()).hexdigest(),
        "exact_validation_target_indices_sha256":hashlib.sha256(
        np.ascontiguousarray(ends.astype("<i8")).tobytes()).hexdigest(),
        "training_target_rows":int(len(tr)),"validation_target_rows":int(len(va)),
        "raw_feature_count":len(cols),"class_names":classes,"categorical_features":cats,
        "validation_target_sequence_matches_original":True,
        "train_source_sha256":cfg["train_csv_sha256"],
        "original_split_sha256":cfg["reference_split_sha256"],
        "original_saved_predictions_sha256":cfg["reference_pred_sha256"],
        "source_order_is_not_verified_chronological":True,
        "test_dataset_accessed":False}
    return tr,y_tr[4:],va,y_ref,classes,cats,note

def model_scores(y,prob,classes,ref_macro,ref_vit,model_id):
    summary,mat,pred=score_predictions(y,prob,classes)
    precision,recall,f1,support=precision_recall_fscore_support(
        y,pred,labels=np.arange(10),zero_division=0)
    summary.update({"model_id":model_id,"weighted_f1":float(f1_score(
        y,pred,labels=np.arange(10),average="weighted",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "delta_f1_vs_reconstructed_scv":float(summary["macro_f1"]-ref_macro),
        "delta_f1_vs_vit_only":float(summary["macro_f1"]-ref_vit),
        "internal_validation_only":True})
    per=pd.DataFrame({"class":classes,"support":support,
                      "precision":precision,"recall":recall,"f1":f1})
    return summary,mat,per

def plot_numbers(run,results,classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi":140,"savefig.dpi":240,
          "pdf.fonttype":42,"svg.fonttype":"none","font.size":10,
          "axes.spines.top":False,"axes.spines.right":False})
    figs=run/"figures";figs.mkdir(exist_ok=True)
    metrics=pd.DataFrame([{"model_id":key,**record["metrics"]} for key,record in results.items()])
    metrics.to_csv(run/"aggregate_model_scores.csv",index=False)
    paths=[]
    def save(fig,name,source):
        fig.tight_layout(pad=1.3)
        for ext in ("png","svg","pdf"):
            output=figs/(name+"."+ext)
            fig.savefig(output,bbox_inches="tight")
            paths.append({"file":str(output.relative_to(run)),"bytes":output.stat().st_size,
                          "sha256":digest(output),"source":source})
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    index=np.arange(len(metrics))
    ax.bar(index,metrics["macro_f1"]*100,width=.65)
    ax.axhline(57.44094617438289,linestyle="--",color="#D58447",
               label="ViT-only 57.44% (same validation labels)")
    ax.axhline(55.86639076144937,linestyle=":",color="#4675A2",
               label="CNN-ViT V0 55.87%")
    ax.set_xticks(index,metrics["model_id"],rotation=13,ha="right")
    ax.set_ylabel("Ten-class macro-F1 (%)")
    ax.set_title("Actual tabular versus image-baseline internal-validation scores")
    ax.legend(fontsize=8,frameon=False)
    save(fig,"01_macro_f1_comparison","aggregate_model_scores.csv")
    for key in results:
        runpart=run/key
        per=pd.read_csv(runpart/"per_class.csv")
        conf=pd.read_csv(runpart/"confusion.csv")
        fig,ax=plt.subplots(figsize=(8,5))
        order=per.sort_values("f1")
        ax.barh(order["class"],order["f1"]*100)
        ax.set_xlim(0,100)
        ax.set_xlabel("Per-class F1 (%)")
        ax.set_title(f"{key}: observed ten-class validation F1")
        save(fig,f"02_{key}_per_class_f1",f"{key}/per_class.csv")
        v=conf[classes].to_numpy()
        v=v/np.maximum(1,v.sum(axis=1,keepdims=True))
        fig,ax=plt.subplots(figsize=(8.5,7))
        im=ax.imshow(v,cmap="Blues",vmin=0,vmax=1)
        ax.set_xticks(range(10),classes,rotation=50,ha="right")
        ax.set_yticks(range(10),classes)
        ax.set_xlabel("Predicted attack");ax.set_ylabel("Actual attack")
        ax.set_title(f"{key}: normalized confusion")
        fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
        save(fig,f"03_{key}_confusion",f"{key}/confusion.csv")
    write_json(run/"figure_manifest.json",
        {"experiment_id":"CTRL-TAB-001-S42","exports":paths,
         "note":"Observed classification metrics only; no synthetic result plot values."})
    print(f"SAVED {len(paths)} FIGURE FILES, WITH SOURCE AND SHA256",flush=True)

def run(config_path,repo,reference,output):
    import lightgbm as lgb
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("official_test")!="FORBIDDEN":
        raise ValueError("Test must stay locked")
    np.random.seed(cfg["seed"])
    repo=Path(repo);ref=Path(reference);run=Path(output)
    run.mkdir(exist_ok=True,parents=True)
    tr,ytr,va,yva,classes,cats,lineage=prepare(cfg,ref)
    write_json(run/"config_effective.json",cfg)
    write_json(run/"data_lineage.json",lineage)
    (run/"source_commit.txt").write_text(source_commit(repo)+"\n")
    if not (run/"original_training_environment.txt").exists():
        try:
            frozen=subprocess.check_output([sys.executable,"-m","pip","freeze"],
                text=True,timeout=90)
        except (OSError,subprocess.TimeoutExpired,subprocess.CalledProcessError):
            frozen="pip freeze unavailable"
        (run/"original_training_environment.txt").write_text(
            f"Python: {platform.python_version()}\nLightGBM: {lgb.__version__}\n"+
            frozen,encoding="utf-8")
    results={}
    for candidate in cfg["candidates"]:
        key=candidate["id"];dest=run/key;dest.mkdir(exist_ok=True)
        scorefile=dest/"metrics.json"
        if scorefile.exists():
            saved=json.loads(scorefile.read_text(encoding="utf-8"))
            if (dest/"model.txt").is_file() and (dest/"predictions_PRIVATE.npz").is_file():
                results[key]={"metrics":saved}
                print(f"SKIPPING COMPLETED MODEL {key}: macro-F1={saved['macro_f1']:.5f}",flush=True)
                continue
            raise RuntimeError("Partial candidate requires investigation, refusing silent retrain: "+key)
        class_w=class_weights(ytr,candidate["weight_power"],
                              candidate.get("max_class_weight",12))
        weights=class_w[ytr].astype(np.float64)
        print("\n"+"="*65+"\nTRAINING "+key,flush=True)
        print("Class weights:",{classes[i]:round(float(w),2)
                                for i,w in enumerate(class_w)},flush=True)
        start=time.perf_counter()
        dtrain=lgb.Dataset(tr,label=ytr,weight=weights,feature_name=list(tr.columns),
                           categorical_feature=cats,free_raw_data=False)
        dval=lgb.Dataset(va,label=yva,reference=dtrain,feature_name=list(va.columns),
                         categorical_feature=cats,free_raw_data=False)
        p={"objective":"multiclass","num_class":10,"metric":"None",
           "learning_rate":candidate["learning_rate"],
           "num_leaves":candidate["num_leaves"],
           "min_data_in_leaf":candidate["min_data_in_leaf"],
           "max_depth":candidate["max_depth"],"feature_fraction":0.85,
           "bagging_fraction":0.85,"bagging_freq":1,
           "lambda_l2":5.0,"verbosity":-1,"seed":cfg["seed"],
           "num_threads":max(2,min(8,os.cpu_count() or 4)),
           "deterministic":True,"force_col_wise":True}
        evaluations={}
        bar=tqdm(total=candidate["num_boost_round"],desc=key+" boosting",
                 unit="trees",dynamic_ncols=True)
        def tick(env):
            val=env.iteration+1
            bar.update(max(0,val-bar.n))
        tick.order=80
        try:
            model=lgb.train(p,dtrain,num_boost_round=candidate["num_boost_round"],
                    valid_sets=[dval],valid_names=["internal_validation"],
                    feval=macro_eval,callbacks=[
                        lgb.early_stopping(candidate["early_stopping_rounds"],
                                           verbose=False),
                        lgb.record_evaluation(evaluations),
                        lgb.log_evaluation(period=50),
                        tick])
        finally:
            bar.close()
        pred=model.predict(va,num_iteration=model.best_iteration)
        score,mat,per=model_scores(yva,pred,classes,
                0.5586639076144937,0.5744094617438289,key)
        score["best_iteration"]=int(model.best_iteration)
        score["fit_elapsed_seconds"]=round(time.perf_counter()-start,2)
        score["class_weights"]=class_w.tolist()
        score["feature_count"]=len(tr.columns)
        model.save_model(str(dest/"model.txt"))
        np.savez_compressed(dest/"predictions_PRIVATE.npz",
            y_true=yva.astype(np.uint8),
            y_pred=pred.argmax(1).astype(np.uint8),
            probabilities=np.asarray(pred,dtype=np.float32))
        per.to_csv(dest/"per_class.csv",index=False)
        pd.DataFrame(mat,index=classes,columns=classes).rename_axis(
            "true_class").to_csv(dest/"confusion.csv")
        history=evaluations["internal_validation"]["macro_f1"]
        pd.DataFrame({"iteration":range(1,len(history)+1),
                      "val_macro_f1":history}).to_csv(dest/"learning.csv",index=False)
        score["checkpoint_sha256"]=digest(dest/"model.txt")
        score["private_predictions_sha256"]=digest(dest/"predictions_PRIVATE.npz")
        write_json(scorefile,score)
        results[key]={"metrics":score}
        print("\nMEASURED "+key+":",json.dumps(
            {k:score[k] for k in ["accuracy","macro_f1","balanced_accuracy",
                  "DoS_to_Exploits","Backdoor_to_Exploits",
                  "best_iteration","fit_elapsed_seconds"]},indent=2),flush=True)
        write_json(run/"current_results.json",
                   {"completed_candidates":results,
                    "official_test_accessed":False,
                    "unbiased_test_results_exist":False})
    if not results:raise RuntimeError("No candidates finished")
    best=max(results,key=lambda x:results[x]["metrics"]["macro_f1"])
    plot_numbers(run,results,classes)
    write_json(run/"summary.json",{
        "experiment_id":cfg["experiment_id"],
        "best_internal_validation_model":best,
        "best_internal_validation_macro_f1":results[best]["metrics"]["macro_f1"],
        "model_results":results,"reference_validation_target_count":35005,
        "same_original_validation_targets":True,"official_test_used":False,
        "comparison_warning":"Single seed; models selected using internal validation; no official test or published-paper superiority claims."})
    print("\n"+"="*75)
    print("FAST TABULAR REAL RESULTS COMPLETE")
    for key in results:
        z=results[key]["metrics"]
        print(f"{key:24} macro-F1={z['macro_f1']*100:7.3f}% "
              f"accuracy={z['accuracy']*100:7.3f}% "
              f"balanced-acc={z['balanced_accuracy']*100:7.3f}%")
    print("BEST VALIDATION MODEL:",best,flush=True)
    print("FULL CHECKPOINTS AND PER-ROW PREDICTIONS ARE SAVED LOCALLY — KEEP KAGGLE SESSION",flush=True)
    print("="*75,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--reference",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    run(args.config,args.repo,args.reference,args.output)
