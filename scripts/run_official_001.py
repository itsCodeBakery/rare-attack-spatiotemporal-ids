"""WINDOW-OFFICIAL-001: frozen full-training GBDT, *one* official test pass.

The trained booster and immutable metadata are checkpointed BEFORE the official
testing CSV is ever opened. Never select epochs/models/thresholds using tests.
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
from sklearn.metrics import (
    accuracy_score,balanced_accuracy_score,confusion_matrix,
    f1_score,precision_recall_fscore_support,
)
from tqdm.auto import tqdm

if __package__:
    from scripts.run_window_e2 import build_split_windows,class_weights,store,digest
    from scripts.run_window_e3 import append_stats
else:
    from run_window_e2 import build_split_windows,class_weights,store,digest
    from run_window_e3 import append_stats

CLASS_ORDER=("Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
             "Normal","Reconnaissance","Shellcode","Worms")
CATEGORIES=("proto","service","state")
NEEDED_REPORTS=("accuracy","macro_f1","weighted_f1","precision_macro",
                "recall_macro","precision_weighted","recall_weighted")

def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()

def paths_and_schema(repo,cfg):
    repo=Path(repo)
    manifest=json.loads((repo/"results/WINDOW-E3-001-S42/feature_manifest.json").read_text())
    lineage=json.loads((repo/"results/WINDOW-E3-001-S42/data_lineage.json").read_text())
    features=manifest["raw_window_feature_names"]
    statistics=manifest["statistic_feature_names"]
    categories=lineage["categorical_flat_features"]
    if (len(features)!=210 or len(statistics)!=234 or len(categories)!=15 or
        tuple(lineage["class_names"])!=CLASS_ORDER or
        cfg["feature_count"]!=444 or cfg["final_train_iterations"]!=58):
        raise ValueError("Frozen E3 feature schema / class ordering has changed")
    names=[name.removeprefix("lag0__") for name in features if name.startswith("lag0__")]
    numeric=[x for x in names if x not in CATEGORIES]
    if (len(names)!=42 or len(numeric)!=39 or
        not all(f"lag{lag}__{x}" in features for lag in range(5) for x in names)):
        raise ValueError("Unrecognized frozen original 42-feature input schema")
    if any(x in names for x in ("attack_cat","label","id")):
        raise ValueError("Target labels/ids are not allowed as features")
    if set(categories)!={f"lag{lag}__{x}" for lag in range(5) for x in CATEGORIES}:
        raise ValueError("Categorical fields differ from E3")
    return names,numeric,features,statistics,categories

def csv_path(source_root,filename):
    root=Path(source_root)
    matches=sorted(root.rglob(filename))
    if len(matches)!=1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} attached in /kaggle/input, found {len(matches)}. "
            "Attach the original UNSW-NB15 training/testing dataset before running.")
    return matches[0]

def attack_targets(frame):
    y=frame["attack_cat"].astype("string").str.strip().map(
        {name:i for i,name in enumerate(CLASS_ORDER)})
    if y.isna().any():
        unexpected=frame.loc[y.isna(),"attack_cat"].value_counts(dropna=False).to_dict()
        raise ValueError("Unexpected official class names: "+str(unexpected))
    return y.to_numpy(dtype=np.int64)

def prepare_partition(frame,feature_names,numeric,flat_features,statistic_names,
                      training_vocabs=None):
    raw=build_split_windows(frame,feature_names,CATEGORIES,5)
    categories=[f"lag{lag}__{cat}" for lag in range(4,-1,-1) for cat in CATEGORIES]
    vocabs={}
    for key in categories:
        if training_vocabs is None:
            vocab=pd.Index(raw[key].unique()).sort_values().tolist()
        else:
            vocab=training_vocabs[key]
        raw[key]=pd.Categorical(raw[key],categories=vocab)
        vocabs[key]=vocab
    augmented,extra=append_stats(raw,numeric)
    if (list(raw.columns)!=flat_features or extra!=statistic_names or
        len(augmented.columns)!=444):
        raise ValueError("Final 444-feature matrix no longer matches original E3 order")
    if np.isinf(augmented.select_dtypes(include=[np.number]).to_numpy()).any():
        raise ValueError("Unusable infinite numerical context feature")
    return augmented,categories,vocabs

def all_metrics(y,p):
    yp=p.argmax(axis=1)
    labels=np.arange(10)
    cm=confusion_matrix(y,yp,labels=labels)
    prec,rec,f,s=precision_recall_fscore_support(y,yp,labels=labels,zero_division=0)
    record={
       "n":int(len(y)),
       "correct":int(np.trace(cm)),
       "accuracy":float(accuracy_score(y,yp)),
       "macro_f1":float(f1_score(y,yp,labels=labels,average="macro",zero_division=0)),
       "weighted_f1":float(f1_score(y,yp,labels=labels,average="weighted",zero_division=0)),
       "precision_macro":float(np.mean(prec)),
       "recall_macro":float(np.mean(rec)),
       "precision_weighted":float(np.average(prec,weights=s)),
       "recall_weighted":float(np.average(rec,weights=s)),
       "precision_micro":float(accuracy_score(y,yp)),
       "recall_micro":float(accuracy_score(y,yp)),
       "f1_micro":float(accuracy_score(y,yp)),
       "balanced_accuracy":float(balanced_accuracy_score(y,yp)),
       "DoS_to_Exploits":int(cm[CLASS_ORDER.index("DoS"),CLASS_ORDER.index("Exploits")]),
       "Backdoor_to_Exploits":int(cm[CLASS_ORDER.index("Backdoor"),CLASS_ORDER.index("Exploits")]),
       "dos_recall":float(rec[CLASS_ORDER.index("DoS")]),
       "analysis_recall":float(rec[CLASS_ORDER.index("Analysis")]),
       "backdoor_recall":float(rec[CLASS_ORDER.index("Backdoor")]),
    }
    per=pd.DataFrame({"class":CLASS_ORDER,"support":s,
                       "precision":prec,"recall":rec,"f1":f})
    y_binary=(y!=CLASS_ORDER.index("Normal")).astype(int)
    yp_binary=(yp!=CLASS_ORDER.index("Normal")).astype(int)
    bp,br,bf,bs=precision_recall_fscore_support(
        y_binary,yp_binary,labels=[0,1],zero_division=0)
    record["binary_attack_precision"]=float(bp[1])
    record["binary_attack_recall"]=float(br[1])
    record["binary_attack_f1"]=float(bf[1])
    record["binary_attack_accuracy"]=float(accuracy_score(y_binary,yp_binary))
    return record,cm,per,yp

def run(args):
    repo=Path(args.repo)
    out=Path(args.output)
    cfg=json.loads(Path(args.config).read_text())
    if cfg["status"]!="PRE_REGISTERED_BEFORE_OPENING_TEST" or not cfg["frozen_before_official_test"]:
        raise ValueError("Frozen methodological configuration not preregistered")
    if out.joinpath("official_evaluation_complete.json").exists():
        raise RuntimeError("Official test was already evaluated; do not repeat it")
    if cfg["no_additional_training_variants"] is not True:
        raise ValueError("This phase permits one preselected model only")
    train_file=csv_path(args.input,cfg["original_training_csv"])
    test_file=csv_path(args.input,cfg["original_testing_csv"])
    if train_file.resolve()==test_file.resolve():
        raise ValueError("Both CSV files must be distinct")
    if digest(train_file)!=cfg["training_csv_sha256"]:
        raise ValueError("Official training CSV SHA256 differs from frozen source")
    # Only TRAINING input content is read before checkpointed pre-test freeze.
    names,numeric,expected_raw,expected_stats,cats=paths_and_schema(repo,cfg)
    print("OFFICIAL 001: TRAINING file SHA256 verified; TEST labels NOT opened.",flush=True)
    training_frame=pd.read_csv(train_file,usecols=names+["attack_cat"],low_memory=False)
    if len(training_frame)!=cfg["original_training_rows"]:
        raise ValueError("Full official training source row count differs")
    train_labels=attack_targets(training_frame)
    training_x,categorical,levels=prepare_partition(
        training_frame,names,numeric,expected_raw,expected_stats)
    y=train_labels[4:]
    if len(training_x)!=cfg["full_training_window_count"] or len(y)!=len(training_x):
        raise ValueError("Full train five-record window count differs")
    del training_frame,train_labels
    out.mkdir(parents=True,exist_ok=True)
    store(out/"config_effective.json",cfg)
    store(out/"training_category_vocab_PRIVATE.json",levels)
    store(out/"data_lineage.json",{
        "official_test_used_for_training":False,
        "official_test_used_for_model_selection":False,
        "official_test_opened_before_checkpoint":False,
        "train_csv_sha256":cfg["training_csv_sha256"],
        "train_rows":cfg["original_training_rows"],
        "train_windows":len(y),"test_rows_expected":cfg["original_testing_rows"],
        "test_windows_expected":cfg["official_test_window_count"],
        "raw_feature_count":42,"feature_count":444,
        "training_only_category_vocab":True,
        "serialized_order_has_verified_time":False,
        "class_names":list(CLASS_ORDER),
        "paper_comparison_requires_averaging_definition_check":True})
    store(out/"feature_manifest.json",{
        "ordered_features":list(training_x.columns),
        "native_categorical_features":categorical,
        "train_windows":len(y),
        "training_only_categorical_vocabulary":True})
    (out/"source_code_commit.txt").write_text(subprocess.check_output(
         ["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()+"\n")
    (out/"training_environment.txt").write_text(
         f"Python={platform.python_version()},lightgbm={lgb.__version__},pandas={pd.__version__},numpy={np.__version__}\n")
    w=class_weights(y,cfg["class_weight_power"],cfg["max_class_weight"])
    ds=lgb.Dataset(training_x,label=y,weight=w[y],
                   feature_name=list(training_x.columns),
                   categorical_feature=categorical,free_raw_data=False)
    params={
        "objective":"multiclass","num_class":10,"metric":"None",
        "learning_rate":cfg["learning_rate"],
        "num_leaves":cfg["num_leaves"],
        "min_data_in_leaf":cfg["min_data_in_leaf"],
        "feature_fraction":cfg["feature_fraction"],
        "bagging_fraction":cfg["bagging_fraction"],
        "bagging_freq":cfg["bagging_freq"],
        "lambda_l2":cfg["lambda_l2"],
        "seed":cfg["seed"],"deterministic":cfg["deterministic"],
        "force_col_wise":cfg["force_col_wise"],
        "num_threads":max(2,min(8,os.cpu_count() or 4)),
        "verbosity":-1,
    }
    start=time.monotonic()
    with tqdm(total=cfg["final_train_iterations"],desc="OFFICIAL full-train fixed boosting",
              unit="round",dynamic_ncols=True) as bar:
        def progress(env):
            bar.update(env.iteration+1-bar.n)
        progress.order=85
        model=lgb.train(params,ds,num_boost_round=cfg["final_train_iterations"],
                        callbacks=[progress,lgb.log_evaluation(period=10)])
    model_file=out/"final_model_PRIVATE.txt"
    model.save_model(str(model_file),num_iteration=cfg["final_train_iterations"])
    fitted_hash=digest(model_file)
    frozen={
        "experiment_id":cfg["experiment_id"],"source_training_sha256":digest(train_file),
        "model_checkpoint_sha256":fitted_hash,
        "fitted_full_training_windows":len(y),
        "fixed_rounds":cfg["final_train_iterations"],
        "selected_before_test":True,
        "official_test_opened":False,"runtime_seconds":round(time.monotonic()-start,3),
        "train_label_counts":{CLASSES:int((y==i).sum()) for i,CLASSES in enumerate(CLASS_ORDER)}
    }
    store(out/"PRETEST_FROZEN_CHECKPOINT.json",frozen)
    print("PRETEST FROZEN MODEL SHA256:",fitted_hash,flush=True)
    print("TEST FILE HAS NOT BEEN OPENED. Model is fixed.",flush=True)
    if args.auto_archive:
        from archive_official_001 import archive_pretest
        archive_pretest(out,os.environ.get("GITHUB_TOKEN"))
    del ds,training_x
    # This is the one and only official evaluation. Test contents first opened below.
    print("NOW OPENING ORIGINAL OFFICIAL TEST FOR ONE FROZEN EVALUATION.",flush=True)
    test_frame=pd.read_csv(test_file,usecols=names+["attack_cat"],low_memory=False)
    if len(test_frame)!=cfg["original_testing_rows"]:
        raise ValueError("Official test row count differs; no scores computed")
    y_test=attack_targets(test_frame)[4:]
    test_x,check_categories,_=prepare_partition(
        test_frame,names,numeric,expected_raw,expected_stats,levels)
    if (len(test_x)!=cfg["official_test_window_count"] or
        len(y_test)!=len(test_x) or
        list(test_x.columns)!=list(json.loads(
            (out/"feature_manifest.json").read_text())["ordered_features"]) or
        categorical!=check_categories):
        raise ValueError("Official test input structure does not match frozen model")
    test_hash=digest(test_file)
    probabilities=model.predict(test_x,num_iteration=cfg["final_train_iterations"])
    probabilities=np.asarray(probabilities,dtype=np.float32)
    if probabilities.shape!=(cfg["official_test_window_count"],10):
        raise ValueError("Official test prediction matrix has wrong shape")
    report,cm,per,yp=all_metrics(y_test,probabilities)
    report.update({"official_test_used":True,"used_for_training":False,
       "used_for_tuning":False,"model_locked_before_test":True,
       "test_csv_sha256":test_hash,"model_checkpoint_sha256":fitted_hash,
       "fixed_training_rounds":cfg["final_train_iterations"],
       "evaluation_scope":"original official UNSW testing-set, 82,328 fifth-row window labels",
       "evaluation_window_count":len(y_test),
       "training_full_source_rows":cfg["original_training_rows"],
       "training_full_window_count":len(y)})
    predfile=out/"official_test_predictions_PRIVATE.npz"
    np.savez_compressed(predfile,
        y_true=y_test.astype(np.uint8),
        y_pred=yp.astype(np.uint8),
        probabilities=probabilities,
        official_test_row_indices=np.arange(4,len(test_frame),dtype=np.int32))
    report["official_predictions_sha256"]=digest(predfile)
    pd.DataFrame(cm,index=CLASS_ORDER,columns=CLASS_ORDER).rename_axis(
        "actual_class").to_csv(out/"official_confusion.csv")
    per.to_csv(out/"official_per_class.csv",index=False)
    store(out/"official_metrics.json",report)
    comparison={
        "our_model_metrics":report,
        "paper_aggregate_metrics":cfg["paper_comparator"],
        "numeric_accuracy_difference_pp":100*(report["accuracy"]-cfg["paper_comparator"]["accuracy"]),
        "numeric_macro_f1_difference_pp":100*(report["macro_f1"]-cfg["paper_comparator"]["macro_f1"]),
        "comparison_status":"official split and windowing comparable; differing pipeline/training details and paper metric aggregation must be disclosed",
        "paper_aggregate_f1_not_assumed_weighted":True,
        "method_preselected":True,"official_test_evaluated_once":True,
    }
    store(out/"paper_comparison.json",comparison)
    store(out/"official_evaluation_complete.json",{
        "official_test_evaluated_once":True,
        "experiment_id":cfg["experiment_id"],
        "test_csv_sha256":test_hash,
        "model_checkpoint_sha256":fitted_hash,
        "official_prediction_sha256":digest(predfile),
        "accuracy":report["accuracy"],"macro_f1":report["macro_f1"],
        "num_test_windows":len(y_test)})
    if args.auto_archive:
        from archive_official_001 import archive_final
        archive_final(out,os.environ.get("GITHUB_TOKEN"))
    print("\nOFFICIAL-001 FINAL TEST COMPLETED, PRIVATE ARCHIVED AND VERIFIED",flush=True)
    print(pd.DataFrame([{"source":"Frozen full-train final model",
                         "accuracy":report["accuracy"],"macro_f1":report["macro_f1"],
                         "weighted_f1":report["weighted_f1"],
                         "macro_precision":report["precision_macro"],
                         "macro_recall":report["recall_macro"],
                         "dos_recall":report["dos_recall"],
                         "analysis_recall":report["analysis_recall"],
                         "backdoor_recall":report["backdoor_recall"]}]).to_string(index=False),flush=True)
    print("OFFICIAL TEST NUMERIC ACCURACY DIFFERENCE VS SCV-IDS:",
          round(comparison["numeric_accuracy_difference_pp"],3),"percentage points",flush=True)
    print("OFFICIAL TEST MACRO-F1 NUMERIC DIFFERENCE VS SCV-IDS:",
          round(comparison["numeric_macro_f1_difference_pp"],3),"percentage points",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for option in ("repo","config","input","output"):
        p.add_argument("--"+option,required=True)
    p.add_argument("--auto-archive",action="store_true")
    run(p.parse_args())
