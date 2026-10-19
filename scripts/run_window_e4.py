"""WINDOW-E4-REPL-001: five-seed E3 replication and two fair 444-feature controls.

Fixed source split, exact frozen targets; only original UNSW TRAIN CSV opened.
Seed 42 intact outcome is linked to immutable E3 source artifacts, never fit again.
"""
import argparse,hashlib,json,os,platform,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if __package__:
    from scripts.run_window_e2 import build_data,train_model,parent_scores,metrics,store,digest
    from scripts.run_window_e3 import append_stats,frozen_e2
    from scripts.archive_window_e3 import validate as validate_e3
else:
    from run_window_e2 import build_data,train_model,parent_scores,metrics,store,digest
    from run_window_e3 import append_stats,frozen_e2
    from archive_window_e3 import validate as validate_e3

FAMILIES=("stats_rare_sqrt","block_shuffled_rare_sqrt","independent_shuffled_rare_sqrt")

def perturb(frame,raw_features,seed,kind):
    """Corrupt historical 4 records while preserving lag0, marginals and rows."""
    if kind not in ("block_shuffled","independent_shuffled"):
        raise ValueError("Unknown fixed control type")
    rng=np.random.default_rng(seed)
    out=frame.copy()
    block_perm=rng.permutation(len(frame)) if kind=="block_shuffled" else None
    for lag in (4,3,2,1):
        perm=block_perm if kind=="block_shuffled" else rng.permutation(len(frame))
        for col in raw_features:
            field=f"lag{lag}__{col}"
            out[field]=frame[field].array.take(perm)
    for col in raw_features:
        if not out[f"lag0__{col}"].equals(frame[f"lag0__{col}"]):
            raise ValueError("Shuffled control changed target record")
    return out

def seed_perm(seed,validation,kind):
    if kind not in ("block_shuffled","independent_shuffled"):
        raise ValueError("Invalid permutation type")
    return (20000 if validation else 10000)+97*seed+(
        100000 if kind=="independent_shuffled" else 0)

def score_reused_e3(e3root,y,classes,cfg):
    """Verify *original* seed42 E3 model and exact per-record labels."""
    validate_e3(e3root,"stats_rare_sqrt")
    saved=json.loads((e3root/"stats_rare_sqrt"/"metrics.json").read_text())
    if (saved["checkpoint_sha256"]!=cfg["original_seed42_intact_model_sha256"]
            or saved["private_predictions_sha256"]!=cfg[
                "original_seed42_intact_probabilities_sha256"]):
        raise ValueError("Seed42 E3 experiment not equal preregistered checkpoint")
    with np.load(e3root/"stats_rare_sqrt"/"predictions_PRIVATE.npz",allow_pickle=False) as p:
        if not np.array_equal(p["y_true"].astype(np.int64),y):
            raise ValueError("Seed42 reference targets differ")
        score,_,_=metrics(y,p["probabilities"],classes)
    if (abs(score["accuracy"]-saved["accuracy"])>1e-12 or
        abs(score["macro_f1"]-saved["macro_f1"])>1e-12):
        raise ValueError("Seed42 reference results differ")
    return {"seed":42,"family":"stats_rare_sqrt","provenance":"verified, reused exact E3 fitted model",
            "training_performed_in_E4":False,**score,
            "source_model_sha256":saved["checkpoint_sha256"],
            "source_predictions_sha256":saved["private_predictions_sha256"]}

def collect(seedroot,family,seed):
    path=seedroot/family/"metrics.json"
    metrics_json=json.loads(path.read_text())
    return {"seed":seed,"family":family,"provenance":"actual E4 fit",
        "training_performed_in_E4":True,
        "source_model_sha256":metrics_json["checkpoint_sha256"],
        "source_predictions_sha256":metrics_json["private_predictions_sha256"],
        **{key:metrics_json[key] for key in (
            "accuracy","macro_f1","weighted_f1","balanced_accuracy","count",
            "correct","dos_recall","analysis_recall","backdoor_recall",
            "DoS_to_Exploits","Backdoor_to_Exploits")}}

def summary_tables(root,cfg,records):
    df=pd.DataFrame(records)
    if len(df)!=len(cfg["seeds"])*len(FAMILIES):
        raise ValueError("Missing seed-family rows")
    if len(df[["seed","family"]].drop_duplicates())!=len(df):
        raise ValueError("Duplicated model comparison")
    df=df.sort_values(["family","seed"])
    df.to_csv(root/"all_seed_scores.csv",index=False)
    stats=df.groupby("family").agg(
        runs=("seed","count"),mean_accuracy=("accuracy","mean"),
        sd_accuracy=("accuracy","std"),mean_macro_f1=("macro_f1","mean"),
        sd_macro_f1=("macro_f1","std"),mean_weighted_f1=("weighted_f1","mean"),
        sd_weighted_f1=("weighted_f1","std"),
        mean_dos_recall=("dos_recall","mean"),
        mean_analysis_recall=("analysis_recall","mean"),
        mean_backdoor_recall=("backdoor_recall","mean")).reset_index()
    stats.to_csv(root/"five_seed_summary.csv",index=False)
    paired=df.pivot(index="seed",columns="family",values=["accuracy","macro_f1","dos_recall"])
    entries=[]
    for ablate in FAMILIES[1:]:
        for metric in ("accuracy","macro_f1","dos_recall"):
            delta=paired[metric]["stats_rare_sqrt"]-paired[metric][ablate]
            entries.append({"control":ablate,"metric":metric,
                "paired_intact_minus_control_mean":float(delta.mean()),
                "paired_intact_minus_control_sd":float(delta.std(ddof=1)),
                "seed_differences":{str(x):float(v) for x,v in delta.items()}})
    pd.DataFrame([{k:v for k,v in x.items() if k!="seed_differences"}
                  for x in entries]).to_csv(root/"paired_differences.csv",index=False)
    store(root/"paired_difference_details.json",entries)
    return stats,entries

def run(args):
    repo,reference,baseline,e2root,e3root,root=map(
        Path,(args.repo,args.reference,args.baseline,args.e2,args.e3,args.output))
    cfg=json.loads(Path(args.config).read_text())
    if cfg["official_test"]!="FORBIDDEN" or cfg["seeds"]!=[42,43,44,45,46]:
        raise ValueError("Preregistered E4 official-test lock/seeds have changed")
    original=json.loads((repo/"configs/window_e2_001_seed42.json").read_text())
    base=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
    for k in ("train_csv_sha256","reference_split_sha256","reference_pred_sha256"):
        if not cfg[k]==original[k]==base[k]:
            raise ValueError("Frozen original file hashes changed")
    X,y,V,yv,classes,categories,lineage=build_data(original,base,reference)
    raw_features=[col.removeprefix("lag0__") for col in X.columns
                  if col.startswith("lag0__")]
    numerics=[col for col in raw_features if f"lag0__{col}" not in categories]
    if (len(raw_features)!=42 or len(numerics)!=39 or
        len(X)!=140328 or len(V)!=35005):
        raise ValueError("Frozen window/feature counts changed")
    frozen_e2(e2root,yv,classes)
    parent_scores(baseline,yv,classes)
    intact_seed42=score_reused_e3(e3root,yv,classes,cfg)
    root.mkdir(parents=True,exist_ok=True)
    store(root/"config_effective.json",cfg)
    store(root/"data_lineage.json",{
        **lineage,"test_dataset_accessed":False,
        "experiment_id":cfg["experiment_id"],
        "statistics_feature_count":234,
        "total_expanded_features":444,
        "input_original_train_only":True,
        "seed_count":len(cfg["seeds"]),"independent_validation_splits":1,
        "historical_record_order_is_verified_temporal":False})
    (root/"source_code_commit.txt").write_text(subprocess.check_output(
        ["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()+"\n")
    import lightgbm as lgb
    (root/"training_environment.txt").write_text(
        f"Python {platform.python_version()} LightGBM {lgb.__version__}, "
        f"numpy {np.__version__}, pandas {pd.__version__}\n")
    print("Preparing intact window statistics (444 features)...",flush=True)
    intact_x,stats=append_stats(X,numerics)
    intact_v,stats_val=append_stats(V,numerics)
    if (stats!=stats_val or len(stats)!=234 or
        intact_x.shape!=(140328,444) or intact_v.shape!=(35005,444)):
        raise ValueError("E4 statistic representation inconsistent")
    store(root/"feature_manifest.json",{
        "experiment_id":cfg["experiment_id"],"feature_count":444,
        "feature_names":list(intact_x.columns),
        "numeric_statistic_names":stats,
        "categorical_feature_names":categories,
        "validation_labels_as_model_features":False,
        "window_temporality_claim":False})
    records=[]
    for seed in cfg["seeds"]:
        folder=root/f"S{seed}"
        folder.mkdir(exist_ok=True)
        store(folder/"protocol.json",{
            "experiment_id":cfg["experiment_id"],"seed":seed,
            "official_test_used":False,"frozen_target_count":len(yv),
            "conditions":list(FAMILIES)})
        if seed==42:
            records.append(intact_seed42)
            print("REUSED ORIGINAL E3 SEED42 INTACT MODEL (no retraining)",
                  flush=True)
        for family in FAMILIES:
            if family=="stats_rare_sqrt" and seed==42:continue
            restored=False
            if args.auto_archive:
                from archive_window_e4 import restore_prior_candidate
                restored=restore_prior_candidate(root,seed,family,os.environ.get("GITHUB_TOKEN"))
            if family=="stats_rare_sqrt":
                tr,va=intact_x,intact_v
            else:
                kind=("block_shuffled" if family.startswith("block_")
                      else "independent_shuffled")
                print(f"Build E4 {family} seed {seed} negative-control features...",flush=True)
                tr,_=append_stats(perturb(X,raw_features,seed_perm(
                    seed,False,kind),kind),numerics)
                va,_=append_stats(perturb(V,raw_features,seed_perm(
                    seed,True,kind),kind),numerics)
            if tr.shape!=(140328,444) or va.shape!=(35005,444):
                raise ValueError("E4 mismatched control feature dimensions")
            modelcfg={"id":family,**cfg["lightgbm"]}
            score=train_model(folder,modelcfg,{"seed":seed},
                tr,y,va,yv,classes,categories)
            records.append(collect(folder,family,seed))
            if args.auto_archive and not restored:
                from archive_window_e4 import archive_and_sync
                archive_and_sync(root,seed,family,os.environ.get("GITHUB_TOKEN"))
            print(f"E4 SAVED + ARCHIVED: {family} seed={seed}",flush=True)
        existing=[r for r in records if r["seed"]==seed]
        if len(existing)!=3:raise ValueError("E4 missing one seed-family result")
        pd.DataFrame(existing).to_csv(folder/"seed_scores.csv",index=False)
        if args.auto_archive:
            from archive_window_e4 import sync_progress
            sync_progress(root,seed,records,os.environ.get("GITHUB_TOKEN"))
        print(f"E4 SEED {seed} COMPLETE (3 families)",flush=True)
    stats,deltas=summary_tables(root,cfg,records)
    store(root/"summary.json",{
        "experiment_id":cfg["experiment_id"],"official_test_used":False,
        "n_seeds":5,"n_independent_validation_splits":1,
        "n_fitted_models_in_E4":14,
        "seed42_intact_reused_exact_E3":True,
        "seed42_original_model_sha256":cfg["original_seed42_intact_model_sha256"],
        "paired_controls":deltas,
        "limitations":"Single original group-disjoint internal validation split; window order not verified chronology; official test untouched."})
    if args.auto_archive:
        from archive_window_e4 import sync_final
        sync_final(root,os.environ.get("GITHUB_TOKEN"))
    print("\nWINDOW-E4-REPL-001 FIVE-SEED RESULTS VERIFIED AND ARCHIVED",flush=True)
    print(stats.to_string(index=False),flush=True)
    print("Paired control differences:",(root/"paired_differences.csv").read_text(),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("repo","reference","baseline","e2","e3","config","output"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--auto-archive",action="store_true")
    run(p.parse_args())
