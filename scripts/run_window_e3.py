"""WINDOW-E3-001: preregistered statistics vs shuffled-context ablations.

Three new LightGBM fits on the fixed SCV split. The E2 original target labels
and private checkpoint predictions are independently verified. 'Lag' is a
serialized row position; no verified chronological claim is made.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from run_window_e2 import (build_data, train_model, metrics, parent_scores,
                           store, digest)
from archive_window_e2 import validate as validate_e2

CONTROLS=("window_unweighted","window_rare_sqrt")
FEATURE_STATS=("five_mean","five_std","five_min","five_max",
               "oldest_to_target_diff","previous_four_mean_to_target_diff")

def append_stats(frame,raw_cols):
    """Six row-position aggregates for each of 39 numeric raw features."""
    added={}
    for col in tqdm(raw_cols,desc="Within-window feature statistics",unit="feature"):
        values=np.stack([frame[f"lag{lag}__{col}"].to_numpy(dtype=np.float32)
                         for lag in (4,3,2,1,0)],axis=1)
        with np.errstate(invalid="ignore",divide="ignore"):
            added[f"stats__{col}__five_mean"]=np.nanmean(values,axis=1).astype(np.float32)
            added[f"stats__{col}__five_std"]=np.nanstd(values,axis=1).astype(np.float32)
            added[f"stats__{col}__five_min"]=np.nanmin(values,axis=1).astype(np.float32)
            added[f"stats__{col}__five_max"]=np.nanmax(values,axis=1).astype(np.float32)
            added[f"stats__{col}__oldest_to_target_diff"]=(values[:,4]-values[:,0])
            added[f"stats__{col}__previous_four_mean_to_target_diff"]=(
                values[:,4]-np.nanmean(values[:,:4],axis=1)).astype(np.float32)
    stats=pd.DataFrame(added,index=frame.index)
    if stats.shape[1]!=len(raw_cols)*6:
        raise ValueError("Wrong context statistic count")
    enhanced=pd.concat((frame,stats),axis=1)
    if not np.isfinite(stats.to_numpy(dtype=np.float64)[
            np.isfinite(stats.to_numpy(dtype=np.float64))]).all():
        raise ValueError("Nonnumeric E3 statistics")
    return enhanced, list(stats.columns)

def shuffled_history(frame,columns,seed):
    """Joint permutation within each original lag, separately per split.

    Every lag's entire 42-feature record is moved as one unit.
    The target record (lag0) and every label remain fixed.
    """
    original=frame.copy(deep=True)
    out=frame.copy(deep=True)
    rng=np.random.default_rng(seed)
    for lag in range(1,5):
        perm=rng.permutation(len(frame))
        for col in columns:
            name=f"lag{lag}__{col}"
            values=original[name]
            # .array.take preserves CategoricalDtype train-only vocabularies.
            out[name]=values.array.take(perm)
    for col in columns:
        k=f"lag0__{col}"
        if not out[k].equals(original[k]):
            raise ValueError("Negative control changed newest target-record features")
    return out

def frozen_e2(e2root,y,classes):
    results={}
    for key in CONTROLS:
        validate_e2(e2root,key)
        folder=e2root/key
        recorded=json.loads((folder/"metrics.json").read_text())
        with np.load(folder/"predictions_PRIVATE.npz",allow_pickle=False) as d:
            if not np.array_equal(d["y_true"].astype(np.int64),y):
                raise ValueError("E2 frozen validation targets mismatch")
            score,_,_=metrics(y,d["probabilities"],classes)
        if (abs(score["accuracy"]-recorded["accuracy"])>1e-12 or
            abs(score["macro_f1"]-recorded["macro_f1"])>1e-12):
            raise ValueError("Frozen E2 probability comparison changed")
        results[key]={"model_family":"frozen_exact_5_record_LightGBM",**score}
    return results

def run(args):
    cfg=json.loads(Path(args.config).read_text())
    if cfg["official_test"]!="FORBIDDEN":raise ValueError("Official test access forbidden")
    repo,ref,baseline,e2root,out=map(Path,(
        args.repo,args.reference,args.baseline,args.e2,args.output))
    rootcfg=json.loads((repo/"configs/window_e2_001_seed42.json").read_text())
    source=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
    for name in ("train_csv_sha256","reference_split_sha256","reference_pred_sha256"):
        if cfg[name]!=source[name] or cfg[name]!=rootcfg[name]:
            raise ValueError("E3 source/split hash drift")
    if (cfg["window_size"]!=5 or cfg["total_statistical_features"]!=444 or
        cfg["numerical_feature_count"]!=39):
        raise ValueError("E3 preregistered feature structure changed")
    X,y,V,yv,classes,cats,lineage=build_data(rootcfg,source,ref)
    if X.shape!=(140328,210) or V.shape!=(35005,210):
        raise ValueError("Frozen E2 raw-window baseline dimensions changed")
    numerical=[name.removeprefix("lag0__") for name in X.columns
               if name.startswith("lag0__") and name not in cats]
    assert len(numerical)==cfg["numerical_feature_count"] and len(cats)==15
    e2_results=frozen_e2(e2root,yv,classes)
    old_results=parent_scores(baseline,yv,classes)
    results={**old_results,**e2_results}
    out.mkdir(parents=True,exist_ok=True)
    store(out/"config_effective.json",cfg)
    lineage.update({"derived_statistic_count":len(numerical)*6,
        "model_feature_counts":{"stats_unweighted":444,
                                "stats_rare_sqrt":444,"shuffled_rare_sqrt":210},
        "categorical_derived_features":False,
        "historical_shuffle_train_seed":cfg["shuffling_seed_training"],
        "historical_shuffle_val_seed":cfg["shuffling_seed_validation"],
        "target_record_preserved_in_shuffling":True})
    store(out/"data_lineage.json",lineage)
    (out/"source_code_commit.txt").write_text(
        subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
                                text=True).strip()+"\n",encoding="utf-8")
    import lightgbm as lgb
    (out/"training_environment.txt").write_text(
        f"Python {platform.python_version()}, LightGBM {lgb.__version__}, "
        f"numpy {np.__version__}, pandas {pd.__version__}\n",encoding="utf-8")
    enhanced_x,stat_cols=append_stats(X,numerical)
    enhanced_v,stat_cols_v=append_stats(V,numerical)
    if stat_cols!=stat_cols_v or len(stat_cols)!=234:
        raise ValueError("Different engineered train/validation features")
    if list(enhanced_x.columns)!=list(enhanced_v.columns):
        raise ValueError("Engineered column schema differs")
    if enhanced_x.shape[1]!=444 or enhanced_v.shape[1]!=444:
        raise ValueError("Wrong 444-column E3 feature matrix")
    store(out/"feature_manifest.json",{"experiment_id":cfg["experiment_id"],
         "raw_window_feature_names":list(X.columns),
         "statistic_feature_names":stat_cols,
         "training_only_categorical_levels":True,
         "target_row_is_position_lag0":True,
         "source_order_is_verified_chronological":False,
         "train_rows":len(X),"validation_rows":len(V),
         "model_feature_counts":lineage["model_feature_counts"]})
    store(out/"control_comparison.json",results)
    print("E3 210-feature E2 baseline and 444-feature window statistics ready.",flush=True)
    for cand in cfg["candidates"]:
        name=cand["id"]
        if args.auto_archive:
            from archive_window_e3 import restore_prior_candidate
            restore_prior_candidate(out,name,os.environ.get("GITHUB_TOKEN"))
        if cand["representation"]=="window_plus_stats":
            train,val=enhanced_x,enhanced_v
        elif cand["representation"]=="permuted_historical_context":
            print("Shuffling full historical records within train/validation independently...",flush=True)
            train=shuffled_history(X,
                [name.removeprefix("lag0__") for name in X.columns
                 if name.startswith("lag0__")],cfg["shuffling_seed_training"])
            val=shuffled_history(V,
                [name.removeprefix("lag0__") for name in V.columns
                 if name.startswith("lag0__")],cfg["shuffling_seed_validation"])
        else:raise ValueError("Unknown representation: "+name)
        score=train_model(out,cand,cfg,train,y,val,yv,classes,cats)
        results[name]={"model_family":cand["representation"],**score}
        store(out/"results_so_far.json",{
            "experiment_id":cfg["experiment_id"],"official_test_used":False,
            "models":results})
        if args.auto_archive:
            from archive_window_e3 import archive_and_sync
            archive_and_sync(out,name,os.environ.get("GITHUB_TOKEN"))
        print("E3 CANDIDATE PRIVATE GITHUB BACKUP COMPLETE:",name,flush=True)
    comparison=pd.DataFrame([{"model":name,**result} for name,result in results.items()])
    comparison.to_csv(out/"comparison_all_models.csv",index=False)
    summary={"experiment_id":cfg["experiment_id"],"models":results,
             "official_test_used":False,
             "frozen_validation_target_count":len(yv),
             "predeclared_candidate_names":[x["id"] for x in cfg["candidates"]],
             "window_statistics_vs_E2":{
              key:{"delta_accuracy":results[key]["accuracy"]-e2_results["window_"+key.split("_",1)[1]]["accuracy"],
                   "delta_macro_f1":results[key]["macro_f1"]-e2_results["window_"+key.split("_",1)[1]]["macro_f1"]}
              for key in ("stats_unweighted","stats_rare_sqrt")},
             "negative_control":"shuffled history has unchanged newest-row features and target labels; separately permuted context",
             "scientific_limit":"No verified temporal ordering, no official test, internal validation only."}
    store(out/"summary.json",summary)
    if args.auto_archive:
        from archive_window_e3 import sync_final_summary
        sync_final_summary(out,os.environ.get("GITHUB_TOKEN"))
    print("\nWINDOW-E3-001 RESULTS AND PRIVATE BACKUPS COMPLETE",flush=True)
    print(comparison[["model","accuracy","macro_f1","weighted_f1",
         "dos_recall","analysis_recall","backdoor_recall"]].to_string(index=False),flush=True)
    print("PUBLIC E3:",f"https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/tree/main/results/{cfg['experiment_id']}",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("repo","reference","baseline","e2","config","output"):
        p.add_argument("--"+x,required=True)
    p.add_argument("--auto-archive",action="store_true")
    run(p.parse_args())
