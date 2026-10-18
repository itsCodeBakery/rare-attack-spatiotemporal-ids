"""Single-command, bounded overnight CatBoost repetitions with immediate per-seed backup.

Requires clean-session restored pinned originals; never opens designated test.
All measurements are REAL development-validation data, not claims of published
SCV-IDS official-test superiority.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd

SEEDS=(42,43,44,45,46)

def save(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf-8")

def run_subprocess(command,repo,env):
    print("\nRUN:",command[1],Path(command[2]).name,flush=True)
    subprocess.run(command,cwd=str(repo),env=env,check=True)

def progress_summary(root,seeds):
    records=[]
    for seed in seeds:
        p=root/f"S{seed}"/"summary_scores.csv"
        if not p.is_file():
            continue
        d=pd.read_csv(p)
        required={"candidate","accuracy","macro_f1","weighted_precision",
                  "weighted_recall","weighted_f1","balanced_accuracy",
                  "backdoor_TP","analysis_TP","doS_TP"}
        if not required.issubset(d.columns) or len(d)!=10:
            raise ValueError("Seed summary incomplete: "+str(p))
        for _,row in d.iterrows():
            records.append({"seed":seed,**{key:(float(row[key]) if key in
                 (required-{"candidate"}) else str(row[key]))
                 for key in required}})
    if not records:return
    raw=pd.DataFrame(records)
    raw.to_csv(root/"ALL_SEED_SCORES.csv",index=False)
    groups=raw.groupby("candidate",sort=True)
    agg=groups[["accuracy","macro_f1","weighted_f1","weighted_precision",
                 "weighted_recall","balanced_accuracy"]].agg(
                 ["mean","std","min","max"]).reset_index()
    agg.columns=["_".join(map(str,x)).strip("_") for x in agg.columns]
    agg["completed_seeds"]=agg["candidate"].map(
        groups["seed"].nunique())
    agg.to_csv(root/"ACROSS_SEED_SUMMARY.csv",index=False)
    print("\nOVER-NIGHT REAL COMPLETED-SEED SUMMARY:",flush=True)
    print(agg[["candidate","completed_seeds",
               "accuracy_mean","macro_f1_mean",
               "weighted_f1_mean","macro_f1_std"]].to_string(
               index=False,float_format=lambda x:f"{x:.5f}"),flush=True)
    return len(records)

def main(repo,reference,baseline,root,seed_config,max_hours):
    repo,reference,baseline,root,seed_config=map(Path,(
        repo,reference,baseline,root,seed_config))
    root.mkdir(parents=True,exist_ok=True)
    for path in [reference/"split_indices.npz",
          reference/"validation_predictions.npz",
          baseline/"lgbm_unweighted"/"model.txt",
          baseline/"lgbm_rare_sqrt"/"model.txt"]:
        if not path.is_file():
            raise FileNotFoundError("Pinned original not restored: "+str(path))
    if not os.getenv("GITHUB_TOKEN"):
        raise RuntimeError("Enable authorized pushRare secret BEFORE overnight running")
    first=time.monotonic()
    status={"started_at_utc":datetime.now(timezone.utc).isoformat(),
        "experiment":"CTRL-CAT-OVERNIGHT",
        "seed_order":list(SEEDS),"seed_records":{},
        "max_wall_clock_hours":max_hours,"official_test_used":False}
    save(root/"OVERNIGHT_STATUS.json",status)
    fixed=json.loads(seed_config.read_text(encoding="utf-8"))
    for seed in SEEDS:
        elapsed=(time.monotonic()-first)/3600
        if elapsed>=max_hours:
            print("TIME BUDGET REACHED. Completed seeds are safely backed up.",flush=True)
            break
        data=json.loads(json.dumps(fixed))
        data["experiment_id"]=f"CTRL-CAT-001-S{seed}"
        data["catboost"]["seed"]=seed
        data["seed_role"]="fixed-seed training stochasticity check with identical single original holdout; not a new train-validation fold"
        config_path=root/f"cfg_S{seed}.json"
        save(config_path,data)
        run=root/f"S{seed}"
        print("\n"+"="*75,flush=True)
        print(f"OVERNIGHT SEED {seed} / {SEEDS[-1]} | "
              f"elapsed {elapsed:.2f}h | official test LOCKED",flush=True)
        t=time.monotonic()
        try:
            run_subprocess([sys.executable,
                str(repo/"scripts"/"run_ctrl004_catboost_pareto.py"),
                "--repo",str(repo),
                "--config",str(config_path),
                "--reference",str(reference),
                "--baseline",str(baseline),
                "--output",str(run)],repo,os.environ.copy())
            status["seed_records"][str(seed)]={"training_completed":True}
            save(root/"OVERNIGHT_STATUS.json",status)
            # Back up EACH fully finished seed now, not after five seeds.
            run_subprocess([sys.executable,
                str(repo/"scripts"/"archive_sync_catboost_overnight.py"),
                "--run",str(run),
                "--seed",str(seed),
                "--destination",str(root)],repo,os.environ.copy())
            status["seed_records"][str(seed)].update({
                "github_archive_and_sync_completed":True,
                "elapsed_seconds":round(time.monotonic()-t,2)})
            print(f"SEED {seed}: all metrics and private models backed up",flush=True)
        except Exception as exc:
            status["seed_records"][str(seed)].update({
                "error_type":type(exc).__name__,"error_message":str(exc),
                "github_archive_and_sync_completed":False})
            print("OVERNIGHT PROCESS STOPPED AFTER ERROR:",repr(exc),flush=True)
            save(root/"OVERNIGHT_STATUS.json",status)
            progress_summary(root,SEEDS)
            raise
        save(root/"OVERNIGHT_STATUS.json",status)
        progress_summary(root,SEEDS)
    status["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    status["finished"]=True
    status["seeds_preserved"]=[
        int(k) for k,v in status["seed_records"].items()
        if v.get("github_archive_and_sync_completed")]
    save(root/"OVERNIGHT_STATUS.json",status)
    progress_summary(root,SEEDS)
    print("\n"+"="*75,flush=True)
    print("OVERNIGHT CATBOOST EXPERIMENTS COMPLETED AND ARCHIVED",flush=True)
    print("SEEDS:",status["seeds_preserved"],flush=True)
    print("CANDIDATES PER SEED: 2 genuine CatBoost fits + 8 fixed mixes",flush=True)
    print("NO OFFICIAL TEST RESULTS; NO PUBLICATION SUPERIORITY CLAIM",flush=True)
    print("LOCAL OUTPUTS:",root,flush=True)
    print("="*75,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("repo","reference","baseline","output","config"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--max-hours",type=float,default=9.0)
    args=p.parse_args()
    main(args.repo,args.reference,args.baseline,args.output,
         args.config,args.max_hours)
