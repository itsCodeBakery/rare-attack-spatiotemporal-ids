"""E5-CONTEXT-DIAGNOSTIC-001: TRAIN-only context/artifact audit.

Read-only frozen E3 seed-42 booster; NO training, official-test CSV,
official-test predictions, or model selection in this diagnostic.
No row-level data, model files or probabilities leave private Kaggle storage.
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

if __package__:
    from scripts.run_window_e2 import build_data,digest,store
    from scripts.run_window_e3 import append_stats
    from scripts.archive_window_e2 import validate as validate_e2
    from scripts.archive_window_e3 import validate as validate_e3
    from rareids.baseline_utils import canonical_group_ids
else:
    from run_window_e2 import build_data,digest,store
    from run_window_e3 import append_stats
    from archive_window_e2 import validate as validate_e2
    from archive_window_e3 import validate as validate_e3
    from rareids.baseline_utils import canonical_group_ids

def source_sequence_summary(labels,classes,partition):
    """Descriptive class ordering, NOT chronological dynamics."""
    y=np.asarray(labels,dtype=np.int64)
    if len(y)<5:raise ValueError("Need at least five serialized records")
    if y.min()<0 or y.max()>=len(classes):raise ValueError("Invalid category encoding")
    counts=np.bincount(y,minlength=len(classes))
    p=counts/len(y)
    changes=np.flatnonzero(y[1:]!=y[:-1])+1
    runs=np.diff(np.r_[0,changes,len(y)])
    adjacent=(y[1:]==y[:-1])
    summary={"partition":partition,"source_records":int(len(y)),
        "num_label_runs":int(len(runs)),
        "median_label_run_length":float(np.median(runs)),
        "p90_label_run_length":float(np.quantile(runs,.90)),
        "p99_label_run_length":float(np.quantile(runs,.99)),
        "max_label_run_length":int(max(runs)),
        "observed_adjacent_same_label_rate":float(np.mean(adjacent)),
        "independent_class_marginal_same_label_expectation":float(np.sum(p*p)),
        "label_run_order_is_verified_time":False}
    transitions=np.zeros((len(classes),len(classes)),dtype=np.int64)
    np.add.at(transitions,(y[:-1],y[1:]),1)
    transition_rows=[]
    for i,actual in enumerate(classes):
        for j,nxt in enumerate(classes):
            transition_rows.append({"partition":partition,
                "previous_class":actual,"next_class":nxt,
                "pair_count":int(transitions[i,j]),
                "conditional_probability_given_previous":
                 float(transitions[i,j]/max(1,transitions[i,:].sum()))})
    return summary,transition_rows

def window_purity(labels,classes,partition,window=5):
    y=np.asarray(labels,dtype=np.int64)
    if len(y)<window:raise ValueError("Insufficient source records")
    spans=np.lib.stride_tricks.sliding_window_view(y,window_shape=window)
    current=spans[:,-1]
    same=(spans[:,:-1]==current[:,None])
    pure=np.all(same,axis=1)
    records=[]
    for idx,name in enumerate(classes):
        m=current==idx
        records.append({"partition":partition,"class":name,
            "target_windows":int(m.sum()),
            "all_four_history_labels_match_target_fraction":
                float(pure[m].mean()) if m.any() else None,
            "mean_number_of_matching_previous_labels":
                float(same[m].sum(axis=1).mean()) if m.any() else None})
    records.append({"partition":partition,"class":"ALL",
        "target_windows":int(len(current)),
        "all_four_history_labels_match_target_fraction":float(pure.mean()),
        "mean_number_of_matching_previous_labels":float(same.sum(axis=1).mean())})
    return records

def position_gaps(indices,partition,window=5):
    """Gap in ORIGINAL CSV row positions between split-local neighbors."""
    source=np.asarray(indices,dtype=np.int64)
    if not np.all(np.diff(source)>0):
        raise ValueError("Frozen split-local source record ordering not increasing")
    result=[]
    for lag in range(1,window):
        gap=source[window-1:]-source[window-1-lag:-lag]
        result.append({"partition":partition,"history_position":lag,
            "target_windows":int(len(gap)),
            "min_original_source_index_gap":int(gap.min()),
            "median_original_source_index_gap":float(np.median(gap)),
            "p90_original_source_index_gap":float(np.quantile(gap,.9)),
            "maximum_original_source_index_gap":int(gap.max()),
            "fully_contiguous_original_source_positions_fraction":
                float(np.mean(gap==lag))})
    return result

def class_diagnostics(y,probs,classes):
    pred=np.argmax(probs,axis=1)
    cm=confusion_matrix(y,pred,labels=np.arange(len(classes)))
    prec,rec,f1,sup=precision_recall_fscore_support(
        y,pred,labels=np.arange(len(classes)),zero_division=0)
    error_rows=[];confidence_bins=[];error_pairs=[]
    confidence=probs.max(axis=1)
    correct=pred==y
    for c,name in enumerate(classes):
        m=y==c
        good=m&correct
        bad=m&~correct
        true_conf=probs[m,c]
        class_rows={"class":name,"support":int(m.sum()),
           "precision":float(prec[c]),"recall":float(rec[c]),
           "f1":float(f1[c]),"wrong_count":int(bad.sum()),
           "mean_probability_on_true_class":float(true_conf.mean()) if m.any() else None,
           "mean_max_confidence_on_wrong_predictions":
                float(confidence[bad].mean()) if bad.any() else None,
           "high_confidence_wrong_ge_080":int((bad&(confidence>=.80)).sum()),
           "high_confidence_wrong_ge_095":int((bad&(confidence>=.95)).sum())}
        error_rows.append(class_rows)
        for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            sel=m&(confidence>=low)&((confidence<high) if high<1 else (confidence<=high))
            count=int(sel.sum())
            confidence_bins.append({"class":name,"left":round(float(low),2),
                 "right":round(float(high),2),"count":count,
                 "mean_confidence":float(confidence[sel].mean()) if count else None,
                 "accuracy_in_bin":float(correct[sel].mean()) if count else None})
        for j,wrong_name in enumerate(classes):
            if c!=j and cm[c,j]>0:
                error_pairs.append({"true_class":name,"predicted_class":wrong_name,
                    "count":int(cm[c,j]),"fraction_of_true_class":
                        float(cm[c,j]/max(1,sup[c]))})
    ece=sum(
        np.sum((confidence>=lo)&(confidence<(hi if hi<1 else 1.00000001))) /len(y) *
        abs(float(correct[(confidence>=lo)&(confidence<(hi if hi<1 else 1.00000001))].mean())-
            float(confidence[(confidence>=lo)&(confidence<(hi if hi<1 else 1.00000001))].mean()))
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:])
        if ((confidence>=lo)&(confidence<(hi if hi<1 else 1.00000001))).any()
    )
    brier=float(np.mean(np.sum(probs*probs,axis=1)-2*probs[np.arange(len(y)),y]+1))
    scored={"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,labels=np.arange(len(classes)),average="macro",zero_division=0)),
            "ece_10_bins":float(ece),"multiclass_brier":brier,
            "n":int(len(y))}
    return scored,cm,error_rows,confidence_bins,sorted(
        error_pairs,key=lambda item:-item["count"])

def importance_group(feature):
    if feature.startswith("stats__"):return "engineered_window_statistics"
    lag=feature.split("__",1)[0]
    if lag in {"lag0","lag1","lag2","lag3","lag4"}:
        return "current_record_lag0" if lag=="lag0" else f"historical_{lag}"
    raise ValueError("Unknown learned feature name: "+feature)

def feature_importance(model,features):
    fitted=model.feature_name()
    if fitted!=list(features):
        raise ValueError("E3 checkpoint feature names != exact E3 frozen input order")
    gains=np.asarray(model.feature_importance(importance_type="gain"),dtype=np.float64)
    splits=np.asarray(model.feature_importance(importance_type="split"),dtype=np.int64)
    if len(gains)!=len(features) or len(features)!=444:
        raise ValueError("Unexpected 444-feature LightGBM model")
    table=pd.DataFrame({"feature":fitted,"group":[importance_group(n) for n in fitted],
           "gain":gains,"split_count":splits})
    bygroup=table.groupby("group").agg(
        n_features=("feature","size"),gain=("gain","sum"),splits=("split_count","sum")
    ).reset_index().sort_values("gain",ascending=False)
    bygroup["gain_fraction"]=bygroup["gain"]/max(1.0,bygroup["gain"].sum())
    top=table.sort_values(["gain","feature"],ascending=[False,True]).head(40)
    return bygroup,top

def perturb_one_lag(V,rawcols,lag,seed):
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(V))
    out=V.copy()
    for name in rawcols:
        key=f"lag{lag}__{name}"
        out[key]=V[key].array.take(perm)
    for name in rawcols:
        if not out[f"lag0__{name}"].equals(V[f"lag0__{name}"]):
            raise ValueError("Diagnostic intervention modified current target record")
    return out

def grouped_delta_bootstrap(y,p2,p3,seed=42,groupsize=64,repeats=200):
    """Block bootstrap of *dependent validation windows*, NOT independent test CI."""
    n=len(y)
    rng=np.random.default_rng(seed)
    blocks=[np.arange(i,min(n,i+groupsize)) for i in range(0,n,groupsize)]
    paired=[]
    for _ in tqdm(range(repeats),desc="Exploratory contiguous-window bootstrap",
                  unit="resamples",leave=False):
        sample=rng.integers(0,len(blocks),size=len(blocks))
        idx=np.concatenate([blocks[j] for j in sample])
        m2=f1_score(y[idx],p2[idx],labels=np.arange(10),average="macro",zero_division=0)
        m3=f1_score(y[idx],p3[idx],labels=np.arange(10),average="macro",zero_division=0)
        paired.append((float(np.mean(p3[idx]==y[idx])-np.mean(p2[idx]==y[idx])),
                       float(m3-m2)))
    data=np.asarray(paired)
    return {
      "window_block_length":groupsize,"bootstrap_resamples":repeats,
      "scope":"one original internal validation partition; used for earlier model stopping, descriptive only",
      "paired_accuracy_difference_quantiles_025_50_975":np.quantile(data[:,0],[.025,.5,.975]).tolist(),
      "paired_macro_f1_difference_quantiles_025_50_975":np.quantile(data[:,1],[.025,.5,.975]).tolist()
    }

def run(args):
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["official_test"]!="FORBIDDEN" or cfg["experiment_id"]!="E5-CONTEXT-DIAGNOSTIC-001":
        raise ValueError("E5 may not use official test")
    repo=Path(args.repo);output=Path(args.output)
    raw_path=Path(args.train_csv)
    if raw_path.name!="UNSW_NB15_training-set.csv" or digest(raw_path)!=cfg["train_csv_sha256"]:
        raise ValueError("Only hash-verified original TRAIN source accepted")
    original_cfg=json.loads((repo/"configs/window_e2_001_seed42.json").read_text())
    base_cfg=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
    if not (cfg["frozen_split_sha256"]==original_cfg["reference_split_sha256"]==
            base_cfg["reference_split_sha256"]):
        raise ValueError("Frozen split mismatch")
    X,y,V,yv,classes,categorical,lineage=build_data(
        original_cfg,base_cfg,Path(args.reference))
    if classes!=cfg["classes"] or len(yv)!=35005 or len(V.columns)!=210:
        raise ValueError("E5 targets or source feature schema changed")
    e2=Path(args.e2)
    e3=Path(args.e3)
    validate_e2(e2,"window_rare_sqrt")
    validate_e3(e3,"stats_rare_sqrt")
    model_path=e3/"stats_rare_sqrt"/"model.txt"
    npz3=e3/"stats_rare_sqrt"/"predictions_PRIVATE.npz"
    npz2=e2/"window_rare_sqrt"/"predictions_PRIVATE.npz"
    for actual,expected in ((model_path,cfg["frozen_e3_model_sha256"]),
        (npz3,cfg["frozen_e3_predictions_sha256"]),
        (e2/"window_rare_sqrt"/"model.txt",cfg["frozen_e2_model_sha256"]),
        (npz2,cfg["frozen_e2_predictions_sha256"])):
        if digest(actual)!=expected:
            raise ValueError("Original immutable internal-val private artifact hash mismatch")
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    store(output/"config_effective.json",cfg)
    with np.load(Path(args.reference)/"split_indices.npz",allow_pickle=False) as data:
        itr=data["dev_train_indices"].astype(np.int64)
        iva=data["validation_indices"].astype(np.int64)
        tr_labels=data["dev_train_labels"].astype(np.int64)
        va_labels=data["validation_labels"].astype(np.int64)
    raw_original=pd.read_csv(raw_path,usecols=list(
        [name.removeprefix("lag0__") for name in V.columns if name.startswith("lag0__")])+["attack_cat"],
        low_memory=False)
    raw_cols=[k.removeprefix("lag0__") for k in V.columns if k.startswith("lag0__")]
    all_labels=raw_original["attack_cat"].astype(str).str.strip().map(
        {name:i for i,name in enumerate(classes)}).to_numpy()
    if (len(all_labels)!=175341 or not np.array_equal(all_labels[itr],tr_labels) or
        not np.array_equal(all_labels[iva],va_labels) or
        not np.array_equal(va_labels[4:],yv)):
        raise ValueError("Training-only CSV raw class labels do not match original frozen split")
    groups=canonical_group_ids(raw_original,raw_cols)
    group_train=np.unique(groups[itr]);group_val=np.unique(groups[iva])
    if np.intersect1d(group_train,group_val).size:
        raise ValueError("Source feature duplicate groups leak over development train/val")
    duplicate_group_count=int(np.sum(np.bincount(groups)>1))
    label_group_nunique=pd.DataFrame({"g":groups,"y":all_labels}).groupby("g")["y"].nunique()
    crosslabel_duplicate_groups=int(np.sum(label_group_nunique>1))
    print("Original full TRAIN-only row ordering and frozen split verified",flush=True)
    summary_by_partition={};transitions=[];purity=[];gaps=[]
    for key,partlabels in (
        ("all_original_training_rows",all_labels),
        ("internal_train_split_order",tr_labels),
        ("internal_val_split_order",va_labels)
    ):
        info,rows=source_sequence_summary(partlabels,classes,key)
        summary_by_partition[key]=info;transitions.extend(rows)
        purity.extend(window_purity(partlabels,classes,key))
    gaps.extend(position_gaps(itr,"internal_train",5))
    gaps.extend(position_gaps(iva,"internal_val",5))
    store(output/"source_order_summary.json",{
        "partition_diagnostics":summary_by_partition,
        "distinct_source_feature_groups":int(len(label_group_nunique)),
        "duplicate_feature_groups":duplicate_group_count,
        "same_source_feature_group_crossing_frozen_split":False,
        "source_feature_groups_with_conflicting_labels":crosslabel_duplicate_groups,
        "windows_are_verified_real_time":False,
        "index_semantics":"Original UNSW training CSV row indices, not timestamps"})
    pd.DataFrame(transitions).to_csv(output/"source_transitions.csv",index=False)
    pd.DataFrame(purity).to_csv(output/"window_purity.csv",index=False)
    pd.DataFrame(gaps).to_csv(output/"position_gaps.csv",index=False)
    del raw_original,groups,all_labels,group_train,group_val
    # No fitting: reload EXACT frozen E3 booster and original validation predictions.
    model=lgb.Booster(model_file=str(model_path))
    numeric=[name for name in raw_cols if name not in ("proto","service","state")]
    if len(numeric)!=39:raise ValueError("Unexpected number of numerical fields")
    with np.load(npz3,allow_pickle=False) as data:
        if not np.array_equal(data["y_true"],yv):
            raise ValueError("Original E3 validation labels have changed")
        p3=data["probabilities"].astype(np.float64)
    with np.load(npz2,allow_pickle=False) as data:
        if not np.array_equal(data["y_true"],yv):
            raise ValueError("Original E2 validation labels have changed")
        p2=data["probabilities"].astype(np.float64)
    y2pred=np.argmax(p2,axis=1)
    intact,cols=append_stats(V,numeric)
    if len(cols)!=234 or len(intact.columns)!=444:
        raise ValueError("E3 frozen 444-field representation altered")
    fitted=model.feature_name()
    if fitted!=list(intact.columns):
        raise ValueError("Loaded fitted E3 model expects different features")
    ref_pred=np.asarray(model.predict(intact,num_threads=4))
    if not np.array_equal(ref_pred.argmax(1),p3.argmax(1)):
        raise ValueError("Replayed frozen LightGBM produces different validation decisions")
    baseline,cm,perclass,bins,errors=class_diagnostics(yv,p3,classes)
    base_e2,_,_,_,_=class_diagnostics(yv,p2,classes)
    pd.DataFrame(cm,index=classes,columns=classes).rename_axis("true_class").to_csv(
        output/"class_confusion.csv")
    pd.DataFrame(perclass).to_csv(output/"classwise_errors.csv",index=False)
    pd.DataFrame(bins).to_csv(output/"confidence_bins.csv",index=False)
    pd.DataFrame(errors[:50]).to_csv(output/"top_misclassifications.csv",index=False)
    bygroup,top=feature_importance(model,intact.columns)
    bygroup.to_csv(output/"importance_by_group.csv",index=False)
    top.to_csv(output/"top_features.csv",index=False)
    print("Frozen E3 internal-validation scores and importance audited.",flush=True)
    # Read-only E3 model context counterfactuals: features remain 444;
    # the same original 35,005 true labels and newest-record features are fixed.
    interventions=[]
    for lag in cfg["counterfactual"]["historical_positions"]:
        for seed in cfg["counterfactual"]["seeds"]:
            print(f"Perturb ONLY historical lag{lag}, seed={seed}...",flush=True)
            variant=perturb_one_lag(V,raw_cols,lag,seed)
            transformed,_=append_stats(variant,numeric)
            if list(transformed.columns)!=fitted:
                raise ValueError("Same pre-trained E3 Booster must see identical ordered columns")
            probabilities=np.asarray(model.predict(transformed,num_threads=4))
            p=probabilities.argmax(axis=1)
            pp,rr,ff,ss=precision_recall_fscore_support(
                yv,p,labels=np.arange(10),zero_division=0)
            row={"historical_position":lag,"shuffle_seed":seed,
                "accuracy":float(accuracy_score(yv,p)),
                "macro_f1":float(f1_score(yv,p,labels=np.arange(10),average="macro",zero_division=0)),
                "accuracy_change_from_unmodified":float(accuracy_score(yv,p)-baseline["accuracy"]),
                "macro_f1_change_from_unmodified":float(f1_score(yv,p,labels=np.arange(10),average="macro",zero_division=0)-baseline["macro_f1"]),
                "dos_recall":float(rr[classes.index("DoS")]),
                "analysis_recall":float(rr[classes.index("Analysis")]),
                "backdoor_recall":float(rr[classes.index("Backdoor")]),
                "model_refitted":False}
            interventions.append(row)
            del variant,transformed,probabilities
    pd.DataFrame(interventions).to_csv(output/"position_ablation.csv",index=False)
    bootstrap=grouped_delta_bootstrap(yv,y2pred,p3.argmax(axis=1),seed=cfg["seed"])
    summary={
      "experiment_id":cfg["experiment_id"],"official_test_accessed":False,
      "trained_new_models":0,"frozen_private_e3_model_sha256":digest(model_path),
      "frozen_private_e3_val_predictions_sha256":digest(npz3),
      "frozen_private_e2_val_predictions_sha256":digest(npz2),
      "verified_validation_targets":int(len(yv)),
      "original_source_csv_sha256":cfg["train_csv_sha256"],
      "e3_internal_validation":baseline,"e2_internal_validation":base_e2,
      "delta_e3_minus_e2_accuracy":baseline["accuracy"]-base_e2["accuracy"],
      "delta_e3_minus_e2_macro_f1":baseline["macro_f1"]-base_e2["macro_f1"],
      "exploratory_block_bootstrap":bootstrap,
      "row_order_diagnostics":summary_by_partition,
      "source_crosslabel_feature_group_count":crosslabel_duplicate_groups,
      "source_feature_group_overlap_across_frozen_split":0,
      "gain_fraction_by_group":{
          str(row["group"]):float(row["gain_fraction"])
          for _,row in bygroup.iterrows()},
      "position_ablation_mean_effects":pd.DataFrame(interventions).groupby(
          "historical_position")[["accuracy_change_from_unmodified",
              "macro_f1_change_from_unmodified"]].mean().to_dict(orient="index"),
      "important_limitations":[
        "Same training-derived validation used to choose early stopping; results are descriptive.",
        "Permutation is a distribution-shift stress-test, not a causal feature attribution.",
        "CSV row order does not establish temporal or per-flow continuity.",
        "Source adjacent-label purity may reveal dataset sorting/label clustering artifacts.",
        "Official test evidence not loaded and was already evaluated once in separate prior phase."
      ]}
    store(output/"summary.json",summary)
    store(output/"provenance.json",{
        "source_commit":subprocess.check_output(
            ["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        "original_internal_validation_count":int(len(yv)),
        "official_test_opened":False,
        "trained_boosters":0,
        "model_sha256":digest(model_path),
        "training_csv_sha256":digest(raw_path),
        "frozen_split_sha256":digest(Path(args.reference)/"split_indices.npz"),
        "frozen_val_predictions_sha256":digest(Path(args.reference)/"validation_predictions.npz"),
        "python":platform.python_version(),"lightgbm":lgb.__version__,
        "pandas":pd.__version__,"numpy":np.__version__,
        "elapsed_seconds":round(time.monotonic()-start,2)})
    if args.sync:
        from sync_e5_aggregate import push_public_e5
        push_public_e5(output,cfg,os.environ.get("GITHUB_TOKEN"))
    print("\nE5 TRAIN-ONLY AUDIT COMPLETE — NO NEW MODELS, NO OFFICIAL TEST",flush=True)
    print(json.dumps({
        "validation_accuracy":baseline["accuracy"],
        "validation_macro_f1":baseline["macro_f1"],
        "ordering":summary_by_partition,
        "gain_fraction_by_feature_group":summary["gain_fraction_by_group"],
        "position_ablation_mean_effects":summary["position_ablation_mean_effects"],
    },indent=2),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for flag in ("repo","config","train-csv","reference","e2","e3","output"):
        parser.add_argument("--"+flag,required=True)
    parser.add_argument("--sync",action="store_true")
    run(parser.parse_args())
