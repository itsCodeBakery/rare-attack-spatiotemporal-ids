"""CTRL-CAT-001: measured CatBoost and prespecified probability ensembles.

TRAINING-ONLY development using frozen reference fold and 35,005 targets.
No use of designated official test and no re-training of prior models.
Objective: screen accuracy AND macro-F1 Pareto versus SCV-IDS Table V,
not to assert direct superiority before a locked official-test comparison.
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
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import (
    accuracy_score,f1_score,precision_recall_fscore_support,
    balanced_accuracy_score,confusion_matrix
)
from run_ctrl003_tabular_fast import prepare

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(4194304),b""):h.update(block)
    return h.hexdigest()

def save_json(path,content):
    Path(path).write_text(json.dumps(content,sort_keys=True,indent=2)+"\n",
                          encoding="utf-8")

def evaluate(y,probs,classes):
    p=np.asarray(probs,dtype=np.float64)
    if p.shape!=(len(y),10) or not np.isfinite(p).all():
        raise ValueError("Unexpected multiclass prediction matrix")
    if not np.allclose(p.sum(axis=1),1.,atol=1e-5,rtol=0):
        raise ValueError("Predicted class probabilities must sum to 1")
    pred=p.argmax(axis=1)
    cm=confusion_matrix(y,pred,labels=np.arange(10))
    prec,rec,f1,sup=precision_recall_fscore_support(
        y,pred,labels=np.arange(10),zero_division=0)
    vals={"count":int(len(y)),"correct":int(np.trace(cm)),
          "accuracy":float(accuracy_score(y,pred)),
          "macro_f1":float(f1_score(y,pred,labels=np.arange(10),average="macro",zero_division=0)),
          "weighted_f1":float(f1_score(y,pred,labels=np.arange(10),average="weighted",zero_division=0)),
          "weighted_precision":float(np.average(prec,weights=sup)),
          "weighted_recall":float(np.average(rec,weights=sup)),
          "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
          "doS_TP":int(cm[classes.index("DoS"),classes.index("DoS")]),
          "backdoor_TP":int(cm[classes.index("Backdoor"),classes.index("Backdoor")]),
          "analysis_TP":int(cm[classes.index("Analysis"),classes.index("Analysis")])}
    per_class=pd.DataFrame({"class":classes,"support":sup,
                            "precision":prec,"recall":rec,"f1":f1})
    return vals,cm,per_class

def store_variant(folder,y,probs,classes,detail):
    folder.mkdir(parents=True,exist_ok=True)
    score,cm,per=evaluate(y,probs,classes)
    score.update(detail)
    np.savez_compressed(folder/"predictions_PRIVATE.npz",
        y_true=np.asarray(y,dtype=np.uint8),
        probabilities=np.asarray(probs,dtype=np.float32),
        y_pred=np.asarray(probs).argmax(1).astype(np.uint8))
    score["private_probabilities_sha256"]=digest(folder/"predictions_PRIVATE.npz")
    pd.DataFrame(cm,index=classes,columns=classes).rename_axis("true_class").to_csv(
        folder/"confusion.csv")
    per.to_csv(folder/"per_class.csv",index=False)
    save_json(folder/"metrics.json",score)
    print("MEASURED:",folder.name,
          f"Accuracy={score['accuracy']*100:.3f}%",
          f"Macro-F1={score['macro_f1']*100:.3f}%",
          f"Weighted-F1={score['weighted_f1']*100:.3f}%",
          f"Precision(w)={score['weighted_precision']*100:.3f}%",flush=True)
    return score

def verify_baselines(baseline,cfg,y):
    probs={}
    for key,stem in (("lgbm_unweighted","unweighted"),("lgbm_rare_sqrt","weighted")):
        folder=baseline/key
        info=json.loads((folder/"metrics.json").read_text())
        model=folder/"model.txt"
        predictions=folder/"predictions_PRIVATE.npz"
        if (digest(model)!=cfg[f"{stem}_lightgbm_checkpoint_sha256"] or
            digest(predictions)!=cfg["reference_probabilities"][stem]):
            raise ValueError("Frozen LightGBM checksum drift for "+key)
        if (info["checkpoint_sha256"]!=digest(model) or
            info["private_predictions_sha256"]!=digest(predictions)):
            raise ValueError("Original LightGBM results do not match frozen evidence")
        with np.load(predictions,allow_pickle=False) as f:
            if not np.array_equal(f["y_true"].astype(int),y):
                raise ValueError("Original prediction targets differ from fixed split")
            prob=f["probabilities"].astype(np.float64)
            if prob.shape!=(len(y),10):
                raise ValueError("Wrong original LightGBM probability shape")
            probs[key]=prob
    return probs

def render_figures(run,models,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi":140,"savefig.dpi":260,
        "font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    figdir=run/"figures";figdir.mkdir(exist_ok=True)
    data=pd.DataFrame([{"candidate":k,**v} for k,v in models.items()])
    data.to_csv(run/"summary_scores.csv",index=False)
    evidence=[]
    def save(fig,name,source):
        fig.tight_layout(pad=1.3)
        for ext in ("png","svg","pdf"):
            p=figdir/f"{name}.{ext}";fig.savefig(p,bbox_inches="tight")
            evidence.append({"path":str(p.relative_to(run)),
                             "sha256":digest(p),"bytes":p.stat().st_size,"source":source})
        plt.close(fig)
    for fld,name in (("accuracy","01_accuracy"),
                     ("macro_f1","02_macro_f1"),
                     ("weighted_f1","03_weighted_f1"),
                     ("weighted_precision","04_weighted_precision")):
        fig,ax=plt.subplots(figsize=(10.5,4.8))
        ax.bar(np.arange(len(data)),100*data[fld],color="#377e9f")
        ax.set_xticks(np.arange(len(data)),data["candidate"],rotation=28,ha="right")
        ax.set_ylabel(fld.replace("_"," ").title()+" (%)")
        ax.set_title("Real train-derived validation: "+fld.replace("_"," "))
        if fld in ("accuracy","weighted_f1","weighted_precision"):
            threshold=cfg["paper_thresholds"][{"accuracy":"accuracy",
             "weighted_f1":"f1_aggregate","weighted_precision":"precision_aggregate"}[fld]]
            ax.axhline(threshold*100,linestyle="--",color="#c87638",
                       label="Paper-reported aggregate (not same evaluation split)")
            ax.legend(fontsize=8,frameon=False)
        save(fig,name,"summary_scores.csv")
    fig,ax=plt.subplots(figsize=(7.9,6))
    ax.scatter(100*data["accuracy"],100*data["macro_f1"],s=75)
    for _,row in data.iterrows():
        ax.annotate(row["candidate"],(100*row["accuracy"],100*row["macro_f1"]),
                    xytext=(5,5),textcoords="offset points",fontsize=7)
    ax.axvline(cfg["paper_thresholds"]["accuracy"]*100,linestyle="--",
               color="#c87638",label="Paper accuracy (separate test)")
    ax.axhline(cfg["paper_thresholds"]["macro_f1"]*100,linestyle=":",
               color="#c87638",label="Paper macro-F1 (separate test)")
    ax.set_xlabel("Accuracy (%)");ax.set_ylabel("Ten-class macro-F1 (%)")
    ax.set_title("Observed exploratory accuracy–macro-F1 trade-off")
    ax.legend(fontsize=8,frameon=False)
    save(fig,"05_pareto_tradeoff","summary_scores.csv")
    save_json(run/"figure_manifest.json",{
        "experiment_id":cfg["experiment_id"],"exports":evidence,
        "count":15,
        "scope":"Development-only, thresholds from paper do not confirm matched test superiority."})
    print("FIGURES: 5 verified numeric plot families × 3 formats = 15",flush=True)

def run(args):
    repo=Path(args.repo);run=Path(args.output);reference=Path(args.reference)
    baseline=Path(args.baseline);cfg=json.loads(Path(args.config).read_text())
    if cfg.get("official_test")!="LOCKED":raise ValueError("Official test forbidden")
    src_cfg=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
    if src_cfg["train_csv_sha256"]!=cfg["train_csv_sha256"] or src_cfg["reference_split_sha256"]!=cfg["split_sha256"]:
        raise ValueError("Immutable original training source or fold mismatch")
    X,y,V,yval,classes,categorical,lineage=prepare(src_cfg,reference)
    if len(yval)!=35005 or lineage["test_dataset_accessed"]:
        raise ValueError("Original internal target counts changed")
    for name in categorical:
        X[name]=X[name].astype("string").fillna("__MISSING__").astype(str)
        V[name]=V[name].astype("string").fillna("__MISSING__").astype(str)
    y=y.astype(np.int32)
    yval=yval.astype(np.int32)
    frozen=verify_baselines(baseline,cfg,yval)
    run.mkdir(exist_ok=True,parents=True)
    save_json(run/"config_effective.json",cfg)
    save_json(run/"data_lineage.json",lineage)
    (run/"source_code_commit.txt").write_text(subprocess.check_output(
        ["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()+"\n")
    train_pool=Pool(X,label=y,cat_features=categorical)
    valid_pool=Pool(V,label=yval,cat_features=categorical)
    allprobs=dict(frozen)
    models={}
    n=np.bincount(y,minlength=10)
    for setting in cfg["models"]:
        key=setting["id"];path=run/key;path.mkdir(exist_ok=True)
        modelpath=path/"trained_model.cbm"
        scorepath=path/"metrics.json"
        predictionpath=path/"predictions_PRIVATE.npz"
        if scorepath.exists() and modelpath.exists() and predictionpath.exists():
            info=json.loads(scorepath.read_text())
            if digest(modelpath)!=info.get("model_sha256") or digest(predictionpath)!=info.get("private_probabilities_sha256"):
                raise ValueError("Completed saved CatBoost artifact hash mismatch; refuse retrain")
            with np.load(predictionpath,allow_pickle=False) as f:
                if not np.array_equal(f["y_true"].astype(int),yval):
                    raise ValueError("Previously saved CatBoost labels mismatch")
                allprobs[key]=f["probabilities"].astype(np.float64)
            models[key]=info
            print("RESTORED COMPLETED CATBOOST:",key,flush=True)
            continue
        weights=np.clip((n.max()/n.astype(float))**setting["weight_power"],
                        1,setting.get("max_weight",100))
        print("\nTRAINING",key,"class weights",dict(zip(classes,np.round(weights,2))),
              flush=True)
        start=time.monotonic()
        model=CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=setting["iterations"],
            depth=setting["depth"],
            learning_rate=setting["learning_rate"],
            l2_leaf_reg=cfg["catboost"]["l2_leaf_reg"],
            random_seed=cfg["catboost"]["seed"],
            thread_count=cfg["catboost"]["thread_count"],
            task_type=cfg["catboost"]["task_type"],
            class_weights=weights.tolist(),
            early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"],
            one_hot_max_size=12,
            verbose=50,
            allow_writing_files=False
        )
        model.fit(train_pool,eval_set=valid_pool,use_best_model=True)
        model.save_model(str(modelpath))
        p=model.predict_proba(valid_pool)
        if list(np.asarray(model.classes_,dtype=int))!=list(range(10)):
            raise ValueError("CatBoost reported non-frozen class ordering")
        elapsed=time.monotonic()-start
        info=store_variant(path,yval,p,classes,{
            "best_iteration":int(model.get_best_iteration()),
            "training_seconds":float(elapsed),
            "class_weights":weights.tolist(),
            "model_sha256":digest(modelpath),
            "experiment_id":cfg["experiment_id"],
            "is_predeclared_ensemble":False
        })
        models[key]=info
        allprobs[key]=np.asarray(p,dtype=np.float64)
    for ensemble in cfg["fixed_ensembles"]:
        key=ensemble["id"]
        if key in models:continue
        p=sum(w*allprobs[parent] for parent,w in
              zip(ensemble["parents"],ensemble["weights"]))
        if abs(sum(ensemble["weights"])-1)>1e-12:
            raise ValueError("Ensemble weights do not sum to 1")
        models[key]=store_variant(run/key,yval,p,classes,{
            "parents":ensemble["parents"],"weights":ensemble["weights"],
            "is_predeclared_ensemble":True,
            "experiment_id":cfg["experiment_id"]
        })
    # Record both paper-metric crossing and strong validation-baseline crossing.
    thresholds=cfg["paper_thresholds"]
    summary={}
    for key,score in models.items():
        summary[key]={"above_paper_numeric_thresholds_on_DIFFERENT_split":bool(
            score["accuracy"]>thresholds["accuracy"] and
            score["weighted_precision"]>thresholds["precision_aggregate"] and
            score["weighted_recall"]>thresholds["recall_aggregate"] and
            score["weighted_f1"]>thresholds["f1_aggregate"] and
            score["macro_f1"]>thresholds["macro_f1"]),
            "improves_our_current_internal_macro_f1":bool(
                score["macro_f1"]>cfg["current_development_baseline"]["macro_f1"]),
            "improves_our_current_internal_accuracy":bool(
                score["accuracy"]>cfg["current_development_baseline"]["accuracy"])}
    winner=max(models,key=lambda k:(summary[k]["above_paper_numeric_thresholds_on_DIFFERENT_split"],
                                   models[k]["macro_f1"],models[k]["accuracy"]))
    render_figures(run,models,cfg)
    save_json(run/"summary.json",{
        "experiment_id":cfg["experiment_id"],"models":models,"screening":summary,
        "development_best_candidate":winner,
        "official_test_used":False,"reference_validation_count":35005,
        "NO_OFFICIAL_SUPERIORITY_CLAIM":True,
        "warning":"Paper and current models are on different splits and aggregate metric averaging may differ. No final test evaluated."})
    print("\nCATBOOST MULTIMETRIC EXPERIMENT COMPLETE")
    tab=pd.read_csv(run/"summary_scores.csv")
    print(tab[["candidate","accuracy","weighted_precision","weighted_recall",
               "weighted_f1","macro_f1","backdoor_TP","analysis_TP",
               "doS_TP"]].to_string(index=False))
    print("DEVELOPMENT CANDIDATE:",winner)
    print("Official test remains LOCKED; save Kaggle experiment outputs.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("repo","config","reference","baseline","output"):
        parser.add_argument("--"+arg,required=True)
    run(parser.parse_args())
