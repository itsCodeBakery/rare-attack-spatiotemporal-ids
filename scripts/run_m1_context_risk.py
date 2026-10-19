"""M1 train-derived, out-of-fold context-utility and risk controls.

Run ONLY original UNSW training CSV. Never read or tune on the official test.
See docs/M1_CONTEXT_RISK_METHOD_PROTOCOL.md for scientific caveats.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rareids.baseline_utils import canonical_group_ids, split_indices_sha256
from rareids.tabular_eval import class_weights
from rareids.m1_risk_gate import (CLASSES, calibrate, fit_gate, history_gate_features,
                                   mix, numeric_scale, reports)
from scripts.run_e6_order_robustness import (make_raw, make_context, map_labels,
                                               CATS, digest)


def store(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    temp.replace(path)


def secure_file(path):
    """Fails closed if path refers to any test or unapproved input."""
    p=Path(path)
    if p.is_symlink() or p.name!="UNSW_NB15_training-set.csv":
        raise ValueError("M1 permits only the original UNSW_NB15_training-set.csv")
    if not p.is_file():raise FileNotFoundError(p)
    return p


def frozen_config(repo, config_path, train_path, seed):
    cfg=json.loads(Path(config_path).read_text())
    old=json.loads((repo/cfg["base_config"]).read_text())
    if (cfg["experiment_id"]!="M1-CONTEXT-RISK-001"
        or cfg["official_unsw_test"]["access"]!="FORBIDDEN"
        or cfg["training_csv_sha256"]!=old["training_csv_sha256"]
        or cfg["learners"]["num_boost_round"]!=old["fixed_num_boost_round"]
        or seed not in cfg["seeds"]):
        raise ValueError("M1 predeclared protocol/seed mismatch")
    path=secure_file(train_path)
    if digest(path)!=cfg["training_csv_sha256"]:
        raise ValueError("Training source SHA-256 mismatch; no other file allowed")
    feature_manifest=json.loads((repo/cfg["feature_manifest"]).read_text())
    numeric=list(x.replace("lag0__","") for x in feature_manifest["raw_window_feature_names"]
                if x.startswith("lag0__") and x.replace("lag0__","") not in CATS)
    cols=list(x.replace("lag0__","") for x in feature_manifest["raw_window_feature_names"]
              if x.startswith("lag0__"))
    if len(cols)!=42 or len(numeric)!=39 or tuple(cfg["class_order"])!=CLASSES:
        raise ValueError("M1 raw schema does not match preregistered E3")
    return cfg,old,feature_manifest,cols,numeric,path


def fold_zero(y, groups, n_splits, seed):
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    tr, va=next(cv.split(np.zeros(len(y)),y,groups))
    assert_separation(groups,tr,va)
    if set(y[tr])!=set(range(10)) or set(y[va])!=set(range(10)):
        raise ValueError("Frozen group-disjoint fold zero lacks rare-class support")
    return tr,va


def assert_separation(groups,train_idx,valid_idx):
    if len(set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[valid_idx])):
        raise AssertionError("Duplicate-feature source groups cross partition boundary")
    if set(train_idx.tolist()) & set(valid_idx.tolist()):
        raise AssertionError("Same source row appears across partition boundary")


def checked_windows(train,valid, cols,numeric,manifest):
    if len(train)<6 or len(valid)<6:raise ValueError("Partition insufficient for windows")
    a,vocab,cats=make_raw(train,cols)
    b,_,_=make_raw(valid,cols,levels=vocab)
    if a.columns.tolist()!=manifest["raw_window_feature_names"] or b.columns.tolist()!=a.columns.tolist():
        raise ValueError("E3 raw feature order changed")
    contextual_train=make_context(a,numeric,manifest)
    contextual_valid=make_context(b,numeric,manifest)
    tcols=[f"lag0__{x}" for x in cols]
    if len(contextual_train.columns)!=444 or len(tcols)!=42:
        raise ValueError("Frozen E3 444/42 feature count mismatch")
    return a,b,(a[tcols],contextual_train),(b[tcols],contextual_valid),cats


def fit_experts(name,root,frame_train,frame_val,cols,numeric,manifest,base_cfg,seed):
    root.mkdir(parents=True,exist_ok=True)
    raw_train,raw_val,train_features,val_features,categories=checked_windows(
        frame_train,frame_val,cols,numeric,manifest)
    y_train=map_labels(frame_train)[4:]
    y_val=map_labels(frame_val)[4:]
    if set(y_train.tolist())!=set(range(10)) or set(y_val.tolist())!=set(range(10)):
        raise ValueError("Ten-class support absent from a fitted/held-out inner partition")
    pred_file=root/"expert_predictions_PRIVATE.npz"
    model_files=[root/"expert_TARGET_PRIVATE.txt",root/"expert_CONTEXT_PRIVATE.txt"]
    json_file=root/"expert_provenance.json"
    if pred_file.exists() or json_file.exists() or any(p.exists() for p in model_files):
        if not all(p.exists() for p in [pred_file,json_file,*model_files]):
            raise ValueError("Partial expert checkpoint: refuse silent overwrite")
        record=json.loads(json_file.read_text())
        if record["seed"]!=seed or record["name"]!=name or record["n_targets"]!=len(y_val):
            raise ValueError("Restored expert checkpoint metadata mismatch")
        for p,key in [(pred_file,"npz_sha256"),
                      (model_files[0],"target_model_sha256"),
                      (model_files[1],"context_model_sha256")]:
            if digest(p)!=record[key]:
                raise ValueError("Restored expert SHA256 mismatch: "+str(p))
        with np.load(pred_file,allow_pickle=False) as z:
            if not np.array_equal(z["y"].astype(int),y_val):
                raise ValueError("Restored expert target labels mismatch")
            return y_val,z["p_target"].astype(float),z["p_context"].astype(float),\
                    z["gate_features"].astype(float)
    baseline=dict(base_cfg["frozen_model_hyperparameters"])
    params=dict(baseline,seed=int(seed),num_threads=max(2,min(8,os.cpu_count() or 4)))
    weight=class_weights(y_train,base_cfg["class_weight_power"],base_cfg["max_class_weight"])
    probs=[]
    start=time.monotonic()
    # Store original fitted models only after both complete; orphan files fail closed.
    for i,tag in enumerate(("TARGET","CONTEXT")):
        train_x=train_features[i]
        val_x=val_features[i]
        cat=[x for x in categories if x in train_x.columns]
        dtrain=lgb.Dataset(train_x,label=y_train,weight=weight[y_train],
                           categorical_feature=cat,feature_name=list(train_x.columns),
                           free_raw_data=False)
        m=lgb.train(params,dtrain,num_boost_round=base_cfg["fixed_num_boost_round"])
        p=np.asarray(m.predict(val_x),dtype=np.float32)
        if p.shape!=(len(y_val),10) or not np.isfinite(p).all():
            raise ValueError("Real LightGBM expert returned invalid predictions")
        m.save_model(str(model_files[i]))
        probs.append(p)
        print(f"M1 {name} {tag} fit complete in {time.monotonic()-start:.1f}s",flush=True)
    # This scaler uses ONLY the expert-training partition, never target-only heldout rows.
    scaler=numeric_scale(frame_train,numeric)
    gf=history_gate_features(raw_val,probs[0],probs[1],numeric,CATS,scaler)
    tmp=pred_file.with_suffix(".partial.npz")
    np.savez_compressed(tmp,y=y_val.astype(np.uint8),
                        p_target=probs[0],p_context=probs[1],
                        gate_features=gf.astype(np.float32))
    tmp.replace(pred_file)
    store(json_file,{"seed":seed,"name":name,"n_targets":len(y_val),
          "n_training_targets":len(y_train),"rounds":base_cfg["fixed_num_boost_round"],
          "npz_sha256":digest(pred_file),"target_model_sha256":digest(model_files[0]),
          "context_model_sha256":digest(model_files[1]),
          "only_train_derived_source":True,"source_order_is_verified_temporal":False})
    return y_val,probs[0].astype(float),probs[1].astype(float),gf


def check_single_source_label_support(y):
    if set(np.asarray(y,dtype=int).tolist()) != set(range(10)):
        raise ValueError("Stop: subset missing one or more of the ten classes")


def safe_class_summary(y,p,classes):
    r=reports(y,p)
    return {"n":r["n"],"accuracy":r["accuracy"],"macro_f1":r["macro_f1"],
            "weighted_f1":r["weighted_f1"],"benign_fpr":r["benign_fpr"],
            "brier":r["brier"],"support":r["supports"],
            "rare_recalls":{c:r["recall"][c] for c in ("Analysis","Backdoor","DoS","Worms")}}


def run(args):
    repo=Path(args.repo).resolve()
    cfg,base,manifest,cols,numeric,path=frozen_config(repo,args.config,args.train_csv,args.seed)
    out=Path(args.output).resolve()/f"seed_{args.seed}"
    if (out/"final_public_summary.json").exists():
        raise ValueError("Already completed seed: refuse to reopen sealed outer evaluation")
    out.mkdir(parents=True,exist_ok=True)
    store(out/"config_locked.json",cfg)
    source_sha=digest(path)
    frame=pd.read_csv(path,usecols=cols+["attack_cat"],low_memory=False)
    if len(frame)!=cfg["required_original_rows"]:
        raise ValueError("Training source size mismatch")
    y=map_labels(frame)
    groups=canonical_group_ids(frame,cols)
    tr_idx,outer_idx=fold_zero(y,groups,cfg["outer_evaluation"]["n_splits"],
                               cfg["outer_evaluation"]["seed"])
    dev=frame.iloc[tr_idx].reset_index(drop=True)
    held=frame.iloc[outer_idx].reset_index(drop=True)
    dev_y=y[tr_idx]
    dev_groups=groups[tr_idx]
    split_manifest={"outer_source_hash":source_sha,
        "outer_split_sha256":split_indices_sha256(tr_idx,outer_idx),
        "train_raw_rows":len(tr_idx),"heldout_raw_rows":len(outer_idx),
        "source_group_intersection":0,"outer_fold":0,
        "seed":args.seed,"test_dataset_accessed":False}
    store(out/"outer_split_manifest.json",split_manifest)
    # OOF: every held-out expert's fitted data has disjoint raw feature tuple groups.
    cv=StratifiedGroupKFold(n_splits=cfg["inner_base_oof"]["n_splits"],
          shuffle=True,random_state=cfg["inner_base_oof"]["seed"])
    oof=[]; seen=set()
    for j,(inner_tr,inner_va) in enumerate(cv.split(np.zeros(len(dev_y)),dev_y,dev_groups)):
        assert_separation(dev_groups,inner_tr,inner_va)
        inner_tr=np.sort(inner_tr);inner_va=np.sort(inner_va)
        training=dev.iloc[inner_tr].reset_index(drop=True)
        validation=dev.iloc[inner_va].reset_index(drop=True)
        actual,p_t,p_c,gfeat=fit_experts(f"inner_{j}",out/f"inner_{j}",training,
                     validation,cols,numeric,manifest,base,args.seed)
        ids=tr_idx[inner_va[4:]]
        if len(ids)!=len(actual) or len(seen.intersection(ids.tolist())):
            raise ValueError("OOF target source collision")
        seen.update(ids.tolist())
        oof.append((ids,actual,p_t,p_c,gfeat))
        print(f"OOF {j+1}/4 complete; {len(ids)} new independent-target predictions",flush=True)
    ids=np.concatenate([r[0] for r in oof])
    yy=np.concatenate([r[1] for r in oof])
    pt=np.concatenate([r[2] for r in oof])
    pc=np.concatenate([r[3] for r in oof])
    feat=np.concatenate([r[4] for r in oof])
    oo_groups=groups[ids]
    if not set(ids.tolist()).issubset(set(tr_idx.tolist())) or len(np.unique(ids))!=len(ids):
        raise ValueError("Gate-fit OOF includes outer held-out source or duplicate target")
    check_single_source_label_support(yy)
    gate_i,cal_i=fold_zero(yy,oo_groups,4,cfg["gate_vs_cap_partition"]["seed"])
    if not np.array_equal(np.sort(np.r_[gate_i,cal_i]),np.arange(len(yy))):
        raise ValueError("OOF gate/calibration partition does not cover all OOF targets")
    check_single_source_label_support(yy[gate_i]);check_single_source_label_support(yy[cal_i])
    np.savez_compressed(out/"oof_predictions_PRIVATE.npz",source_ids=ids.astype(np.int32),
                        y=yy.astype(np.uint8),p_target=pt.astype(np.float32),
                        p_context=pc.astype(np.float32),features=feat.astype(np.float32),
                        gate_indices=gate_i.astype(np.int32),cal_indices=cal_i.astype(np.int32))
    gate=fit_gate(feat[gate_i],yy[gate_i],pt[gate_i],pc[gate_i],
                  used=np.arange(11),lam=cfg["gating"]["regularizer_lambda"])
    restricted=fit_gate(feat[gate_i],yy[gate_i],pt[gate_i],pc[gate_i],
                        used=np.arange(8),lam=cfg["gating"]["regularizer_lambda"])
    cap=calibrate(yy[cal_i],pt[cal_i],pc[cal_i],gate.predict(feat[cal_i]),cfg["calibration"])
    cap_noctx=calibrate(yy[cal_i],pt[cal_i],pc[cal_i],
                        restricted.predict(feat[cal_i]),cfg["calibration"])
    store(out/"calibration_public.json",{"M1":cap,"NO_CONTEXT":cap_noctx,
             "oof_target_count":len(ids),"gate_fit_n":len(gate_i),"cal_n":len(cal_i),
             "calibration_uses_outer_heldout":False,
             "group_intersection":int(len(set(oo_groups[gate_i])&set(oo_groups[cal_i])))})
    np.savez_compressed(out/"gates_PRIVATE.npz",full=gate.coefficient,mean=gate.mean,std=gate.std,
                        no_context=restricted.coefficient,nc_mean=restricted.mean,nc_std=restricted.std)
    # Explicit stage: the outer-heldout labels have never been used to train/calibrate.
    print("M1 LOCKED OOF FIT + CALIBRATION; now performing single sealed outer evaluation.",flush=True)
    outer_y,outer_t,outer_c,outer_features=fit_experts("outer_frozen",out/"outer_frozen",
                      dev,held,cols,numeric,manifest,base,args.seed)
    if not np.array_equal(outer_y,y[outer_idx][4:]):
        raise ValueError("Outer target provenance not exact")
    g_outer=gate.predict(outer_features)
    ng_outer=restricted.predict(outer_features)
    variants={"C0_TARGET":outer_t,"C1_CONTEXT":outer_c,
        "C2_BLEND":.5*outer_t+.5*outer_c,
        "C3_GATE":mix(outer_t,outer_c,g_outer,1.0),
        "C4_RISK":mix(outer_t,outer_c,g_outer,cap["alpha"]),
        "C5_NOCONTEXT":mix(outer_t,outer_c,ng_outer,cap_noctx["alpha"])}
    import csv
    summary={}
    for name,prob in variants.items():
        r=reports(outer_y,prob)
        summary[name]=safe_class_summary(outer_y,prob,CLASSES)
        with (out/f"{name}_confusion.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["actual_class",*CLASSES])
            for cat,row in zip(CLASSES,r["confusion"]):writer.writerow([cat,*row])
        with (out/f"{name}_per_class.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["class","support","precision","recall","f1"])
            for cat,s in zip(CLASSES,r["supports"]):writer.writerow(
                [cat,s,r["precision"][cat],r["recall"][cat],r["f1"][cat]])
    private=out/"sealed_outer_predictions_PRIVATE.npz"
    np.savez_compressed(private,y=outer_y.astype(np.uint8),
        source_ids=outer_idx[4:].astype(np.int32),
        **{name:p.astype(np.float32) for name,p in variants.items()})
    winner=summary["C4_RISK"]
    report={"experiment_id":cfg["experiment_id"],"seed":args.seed,
      "status":"COMPLETED_PENDING_INDEPENDENT_RESULT_AUDIT",
      "evaluation_scope":"new TRAIN-derived outer group-disjoint fold0; NOT official test",
      "dataset_sha256":source_sha,"split_hash":split_manifest["outer_split_sha256"],
      "outer_target_count":len(outer_y),"oof_target_count":len(yy),
      "gate_calibration_status":cap["status"],"selected_alpha":cap["alpha"],
      "no_context_calibration_status":cap_noctx["status"],
      "official_test_accessed":False,"verified_temporal_order":False,
      "original_official_test_result_unchanged":True,
      "private_predictions_sha256":digest(private),
      "complete_controls":summary,
      "primary_delta_macro_f1_vs_target":winner["macro_f1"]-summary["C0_TARGET"]["macro_f1"],
      "primary_delta_macro_f1_vs_context":winner["macro_f1"]-summary["C1_CONTEXT"]["macro_f1"],
      "scientific_novelty":"NOT_PROVEN"}
    store(out/"final_public_summary.json",report)
    print("M1 COMPLETED TRAIN-ONLY OUTER HOLDOUT. All six model outcomes preserved.",flush=True)
    print("DO NOT CLAIM PAPER OUTPERFORMANCE FROM THIS NEW DEVELOPMENT SPLIT.",flush=True)
    print(pd.DataFrame({k:{"accuracy":v["accuracy"],"macro_f1":v["macro_f1"],
                                "benign_fpr":v["benign_fpr"]} for k,v in summary.items()}).T)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("repo","config","train-csv","output"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--seed",type=int,default=42)
    run(p.parse_args())
