"""Finalize CTRL-VIT-001 from frozen best-checkpoint probabilities. NO TRAINING.

Build metrics, 8 independent numeric plot source tables, 14 PNG/SVG/PDF figures,
and a complete checkpoint/prediction SHA-256 artifact manifest. No official test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,
    precision_recall_fscore_support,roc_auc_score,average_precision_score,
    roc_curve,precision_recall_curve
)
from tqdm.auto import tqdm
from plot_ctrl001_vit import plot_run


def checksum(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for data in iter(lambda:f.read(4194304),b""):h.update(data)
    return h.hexdigest()


def save_json(path,body):
    Path(path).write_text(json.dumps(body,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def finalize(root,weights,ref):
    root,weights,ref=map(Path,(root,weights,ref))
    conf=json.loads((root/"config_effective.json").read_text(encoding="utf-8"))
    line=json.loads((root/"data_lineage.json").read_text(encoding="utf-8"))
    status=json.loads((root/"status.json").read_text(encoding="utf-8"))
    if not status.get("completed") or status["epochs"]!=50:
        raise ValueError("All 50 epochs and best predictions must exist before finalizing.")
    if not line.get("window_target_equal_to_original") or line["validation_window_count"]!=35005:
        raise ValueError("Historical SCV validation targets were not matched")
    if conf["official_test"]!="FORBIDDEN":
        raise ValueError("No official test evaluation permitted")
    if checksum(ref/"split_indices.npz")!=conf["reference_split_sha256"]:
        raise ValueError("Original reference split changed")
    if checksum(ref/"validation_predictions.npz")!=conf["reference_predictions_sha256"]:
        raise ValueError("Original reference predictions changed")
    with np.load(ref/"validation_predictions.npz",allow_pickle=False) as reference:
        original_y=reference["y_true"].astype(np.int64)
        original_p=reference["y_pred"].astype(np.int64)
    with np.load(root/"best_validation_predictions.npz",allow_pickle=False) as output:
        actual_y=output["y_true"].astype(np.int64)
        actual_pred=output["y_pred"].astype(np.int64)
        probabilities=output["probabilities"].astype(np.float64)
    if (not np.array_equal(original_y,actual_y) or
        len(actual_y)!=35005 or
        probabilities.shape!=(35005,10) or
        not np.isfinite(probabilities).all() or
        np.max(np.abs(probabilities.sum(axis=1)-1))>1e-4 or
        not np.array_equal(actual_pred,probabilities.argmax(1))):
        raise ValueError("Control must match original labels, order, and valid probabilities")
    hist=pd.read_csv(root/"training_history.csv")
    if list(hist["epoch"])!=list(range(1,51)):
        raise ValueError("History must contain all 50 sequential epochs")
    if not (weights/"best_val_macro_f1.pt").is_file():
        raise FileNotFoundError("Best control checkpoint missing")
    if checksum(weights/"best_val_macro_f1.pt")!=status["checkpoint_sha256"]:
        raise ValueError("Best checkpoint changed after training")
    if checksum(root/"best_validation_predictions.npz")!=status["predictions_sha256"]:
        raise ValueError("Saved prediction file changed after training")
    classes=status["class_names"]
    if len(classes)!=10:raise ValueError("Unexpected class list")
    oldmacro=float(f1_score(original_y,original_p,average="macro",zero_division=0))
    if abs(oldmacro-conf["reference_original_macro_f1"])>1e-10:
        raise ValueError("Baseline internal-validation macro-F1 changed")

    folder=root/"plot_data";folder.mkdir(parents=True,exist_ok=True)
    hist.to_csv(folder/"training_history.csv",index=False)
    precision,recall,f1,support=precision_recall_fscore_support(
          actual_y,actual_pred,labels=np.arange(10),zero_division=0)
    per=pd.DataFrame({"class":classes,"support":support.astype(int),
        "precision":precision,"recall":recall,"f1":f1})
    per.to_csv(folder/"per_class.csv",index=False)
    mat=confusion_matrix(actual_y,actual_pred,labels=np.arange(10))
    pd.DataFrame(mat,index=classes,columns=classes).rename_axis(
        "true_class").to_csv(folder/"confusion_matrix.csv")
    with np.load(ref/"split_indices.npz",allow_pickle=False) as split:
        train_windows=split["dev_train_labels"][4:].astype(np.int64)
    pd.DataFrame({"class":classes,
       "train_windows":[int(np.sum(train_windows==i)) for i in range(10)],
       "validation_windows":[int(np.sum(actual_y==i)) for i in range(10)]
    }).to_csv(folder/"split_support.csv",index=False)
    pd.DataFrame({"class":classes,
       "actual_proportion":[float(np.mean(actual_y==i)) for i in range(10)],
       "predicted_proportion":[float(np.mean(actual_pred==i)) for i in range(10)]
    }).to_csv(folder/"prediction_proportions.csv",index=False)
    confidence=probabilities.max(axis=1);correct=actual_y==actual_pred
    bounds=np.linspace(0,1,16)
    reliability=[];ece=0.
    for i in range(15):
        match=(confidence>=bounds[i])&(
          (confidence<bounds[i+1]) if i!=14 else (confidence<=bounds[i+1]))
        if match.any():
            acc=float(correct[match].mean())
            avg=float(confidence[match].mean())
            ece+=float(match.mean())*abs(acc-avg)
            reliability.append({"bin":i,"n":int(match.sum()),
              "mean_confidence":avg,"observed_accuracy":acc})
    pd.DataFrame(reliability).to_csv(folder/"reliability.csv",index=False)
    rocpr=[];auc={}
    for i,name in enumerate(tqdm(classes,desc="Generate ROC/PR source tables",unit="classes")):
        positive=(actual_y==i).astype(np.uint8)
        if positive.sum()==0 or positive.sum()==len(positive):
            raise ValueError("Missing class in identical internal validation fold")
        scores=probabilities[:,i]
        roc_auc=float(roc_auc_score(positive,scores))
        ap=float(average_precision_score(positive,scores))
        auc[name]={"roc_auc":roc_auc,"average_precision":ap}
        for kind,xx,yy in (
          ("roc",*roc_curve(positive,scores)[:2]),
          ("pr",*precision_recall_curve(positive,scores)[1::-1])
        ):
            keep=np.unique(np.linspace(0,len(xx)-1,min(len(xx),301),dtype=int))
            for j in keep:
                rocpr.append({"class":name,"kind":kind,
                              "x":float(xx[j]),"y":float(yy[j])})
    pd.DataFrame(rocpr).to_csv(folder/"curves.csv",index=False)
    cm_pred_correct=int(np.trace(mat));acc=cm_pred_correct/len(actual_y)
    macro=float(f1_score(actual_y,actual_pred,average="macro",zero_division=0))
    if (abs(acc-float(accuracy_score(actual_y,actual_pred)))>1e-12 or
        abs(macro-status["best_macro_f1"])>5e-4):
        raise ValueError("Confusion-derived metrics or best history score inconsistent")
    old_acc=float(accuracy_score(original_y,original_p))
    current={"experiment_id":conf["experiment_id"],
        "result_scope":"TRAINING-DERIVED ORIGINAL INTERNAL VALIDATION ONLY",
        "validation_window_count":len(actual_y),
        "correct":cm_pred_correct,"accuracy":acc,
        "macro_f1":macro,
        "weighted_f1":float(f1_score(actual_y,actual_pred,average="weighted",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(actual_y,actual_pred)),
        "normal_false_positive_rate":float(np.mean(
             actual_pred[actual_y==classes.index("Normal")]!=classes.index("Normal"))),
        "ece_15_bins":ece,"per_class_auc_ap":auc,
        "best_epoch":status["best_checkpoint_epoch"],
        "epochs_completed":50,"parameter_count":status["parameter_count"],
        "control_vs_original_reference_same_targets":True,
        "original_reference_accuracy":old_acc,"original_reference_macro_f1":oldmacro,
        "original_reference_comparison_note":"Single-seed control comparison; NO significance or journal superiority claim.",
        "no_official_test":True}
    save_json(root/"metrics_validation.json",current)
    figs=plot_run(root)
    if figs["figure_count"]!=conf["figures_expected"]:
        raise ValueError("Unexpected figure count")
    def record(p):return {"path":str(p),"sha256":checksum(p),"bytes":p.stat().st_size}
    checkpoint_files=sorted(weights.glob("*.pt"))
    frozen=[root/"best_validation_predictions.npz"]
    for p in frozen+checkpoint_files:
        if not p.is_file():raise FileNotFoundError(str(p))
    manifest={"experiment_id":conf["experiment_id"],
       "reference_original_train_source_sha256":conf["source_sha256"],
       "source_training_commit":(root/"source_code_commit.txt").read_text().strip(),
       "frozen_original_split_sha256":conf["reference_split_sha256"],
       "original_reference_predictions_sha256":conf["reference_predictions_sha256"],
       "control_best_predictions_sha256":checksum(frozen[0]),
       "all_checkpoint_and_private_prediction_files":[record(p) for p in frozen+checkpoint_files],
       "figure_count":figs["figure_count"],
       "figure_manifest_sha256":checksum(root/"figure_manifest.json"),
       "figure_exports":figs["figure_count"]*3,
       "numeric_sources":[record(p) for p in sorted(folder.glob("*.csv"))],
       "software_finalization":{"python":platform.python_version(),
         "numpy":np.__version__,"pandas":pd.__version__},
       "official_test_used":False,
       "warning":"Binary checkpoints and individual prediction rows NOT synced publicly. Save a Kaggle version and archive separately."}
    save_json(root/"artifact_manifest.json",manifest)
    (root/"run_report.md").write_text(
        "# CTRL-VIT-001-S42 — validated matched-split standalone ViT control\n\n"
        f"- Original paper-aligned reconstructed SCV, macro-F1: {oldmacro:.6f} (internal validation)\n"
        f"- Standalone ViT macro-F1: **{macro:.6f}** (internal validation)\n"
        f"- Original SCV accuracy: {old_acc:.6f}; standalone ViT accuracy: **{acc:.6f}**\n"
        f"- Identical validation window labels: **YES**, {len(actual_y):,} examples.\n"
        f"- Original fold and fitted preprocessing are reused unchanged.\n"
        f"- Best checkpoint epoch: **{status['best_checkpoint_epoch']} of 50**.\n"
        f"- Figures: **{figs['figure_count']}** × PNG/SVG/PDF.\n"
        "- Training seeds: **one**; no significance/generalization claim.\n"
        "- Official test used: **NO**.\n"
        "- This control is not a proposed novel method; it benchmarks a genuine standalone ViT.\n"
        "- Preserve checkpoints/predictions in private saved Kaggle output; GitHub stores only hashes.\n",
        encoding="utf-8")
    print("CTRL VIT FINALIZATION COMPLETE")
    print(json.dumps({"macro_f1":macro,"accuracy":acc,
          "baseline_scvids_macro_f1":oldmacro,
          "identical_validation_targets":True,"figures":figs["figure_count"]},indent=2))
    return current

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("run","checkpoints","reference"):
        p.add_argument("--"+arg,required=True)
    x=p.parse_args()
    finalize(x.run,x.checkpoints,x.reference)
