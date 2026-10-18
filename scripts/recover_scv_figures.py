"""SCV-IDS-REPRO-001 post-crash figure/metadata recovery — NO TRAINING.

Run in the ORIGINAL Kaggle session with existing /kaggle/working outputs.
Requires best checkpoint, full training history, frozen validation probabilities,
raw sampled embedding vectors and plot source files saved before t-SNE crash.
Emits 28 PNG/SVG/PDF figures, full integrity/provenance manifest and metrics.
Does NOT read any source CSV (including official test), train or infer a model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score)

from rareids.safe_projection import project_saved_embeddings
try:
    from scripts.plot_scv_figures import Gallery
except ModuleNotFoundError:
    from plot_scv_figures import Gallery

PLOT_SOURCES=(
  "training_history.csv","per_class_validation.csv",
  "confusion_validation.csv","split_class_counts.csv",
  "rgb_sample_pixels.csv","reliability_bins.csv",
  "confidence_histogram.csv","roc_pr_curves.csv"
)

def hash_file(path):
    sha=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):sha.update(block)
    return sha.hexdigest()

def save_json(data,path):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def try_source_sha(repository):
    try:
        return subprocess.check_output(["git","-C",str(repository),"rev-parse","HEAD"],
                            stderr=subprocess.DEVNULL,text=True,timeout=15).strip()
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        return None

def verify_saved_data(out,checkpoints):
    paths=[out/"training_history.csv",out/"config_effective.json",
           out/"preprocessing.json",
           out/"split_indices.npz",out/"validation_predictions.npz",
           out/"embedding_vectors.npz",checkpoints/"scv_best.pt",
           checkpoints/"scv_last_resume.pt"]
    paths += [out/"plot_data"/name for name in PLOT_SOURCES]
    print("Checking saved pre-crash evidence:",flush=True)
    missing=[]
    for f in tqdm(paths,desc="Validate existing files",unit="files"):
        if not f.is_file() or f.stat().st_size==0:
            missing.append(str(f))
    if missing:
        raise FileNotFoundError(
            "Saved pre-crash artifacts missing. DO NOT rerun 50 epochs automatically. "
            "Stay in the original Kaggle session; locate any saved output version. "
            "Missing:\n"+"\n".join(missing))
    config=json.loads((out/"config_effective.json").read_text(encoding="utf-8"))
    if config.get("experiment_id")!="SCV-IDS-REPRO-001":
        raise ValueError("Unexpected experiment identity")
    if config.get("official_test_csv")!="DO_NOT_READ_DURING_THIS_EXPERIMENT":
        raise ValueError("Recovery only supports the no-official-test protocol")
    history=pd.read_csv(out/"training_history.csv")
    if len(history)!=int(config["training"]["epochs"]):
        raise ValueError("History does not include all 50 completed epochs")
    if list(history["epoch"])!=list(range(1,len(history)+1)):
        raise ValueError("Incomplete or inconsistent training epoch sequence")
    return config,history,paths

def recovery(out,checkpoints,repository):
    start=time.monotonic()
    out,checkpoints,repository=map(Path,(out,checkpoints,repository))
    cfg,history,required=verify_saved_data(out,checkpoints)
    with np.load(out/"validation_predictions.npz",allow_pickle=False) as archive:
        truth=np.asarray(archive["y_true"],dtype=np.int64)
        pred=np.asarray(archive["y_pred"],dtype=np.int64)
        prob=np.asarray(archive["probabilities"],dtype=np.float64)
        origins=np.asarray(archive["validation_window_end_indices"],dtype=np.int64)
    with np.load(out/"split_indices.npz",allow_pickle=False) as arrays:
        dev=np.asarray(arrays["dev_train_indices"],dtype=np.int64)
        val=np.asarray(arrays["validation_indices"],dtype=np.int64)
        classes=[str(x) for x in arrays["classes"].tolist()]
        tr_y=np.asarray(arrays["dev_train_labels"],dtype=np.int64)
        va_y=np.asarray(arrays["validation_labels"],dtype=np.int64)
    if len(classes)!=10 or len(set(classes))!=10:
        raise ValueError("Missing ten-class official category names")
    if not np.isfinite(prob).all():
        raise ValueError("Saved probabilities contain non-finite values")
    if prob.shape!=(len(truth),10) or len(pred)!=len(truth) or len(origins)!=len(truth):
        raise ValueError("Saved validation array shape inconsistent")
    if (np.abs(prob.sum(axis=1)-1)>1e-3).any():
        raise ValueError("Saved probabilities fail probability sum check")
    if not np.array_equal(np.argmax(prob,axis=1),pred):
        raise ValueError("Saved argmax predictions differ from probability tensor")
    if len(val)!=len(va_y) or len(dev)!=len(tr_y):
        raise ValueError("Saved training/validation row indices inconsistent")
    win=int(cfg["preprocessing"]["window"])
    if len(truth)!=len(val)-win+1 or not np.array_equal(origins,val[win-1:]):
        raise ValueError("Validation lineage does not match saved window labels")
    if not np.array_equal(truth,va_y[win-1:]):
        raise ValueError("Validation target labels do not match original split")
    if len(set(dev)&set(val)):
        raise ValueError("Development and validation ROW indices intersect")
    matrix=confusion_matrix(truth,pred,labels=np.arange(10))
    saved=pd.read_csv(out/"plot_data"/"confusion_validation.csv")
    if saved["true_class"].tolist()!=classes or not np.array_equal(matrix,saved[classes].to_numpy()):
        raise ValueError("Pre-crash confusion source mismatches saved predictions")
    original_history_source=out/"plot_data"/"training_history.csv"
    from_plot=pd.read_csv(original_history_source)
    if not from_plot.equals(history):
        raise ValueError("Pre-crash plot history differs from original epoch history")
    # Do not pretend the recovery environment represents the training environment.
    training_environment_path=out/"environment.txt"
    original_environment_available=training_environment_path.is_file()
    if original_environment_available:
        environment=training_environment_path.read_text(encoding="utf-8")
        recorded=[line.split(":",1)[1].strip() for line in environment.splitlines()
                  if line.startswith("Git commit:")]
        original_sha=recorded[0] if recorded else None
    else:
        original_sha=None
        print("WARNING: environment.txt was not captured by the original run. "
              "Training software freeze and exact training Git SHA remain unverified. "
              "Preserving observed recovery environment SEPARATELY.",flush=True)
    recovery_environment={
        "capture_scope":"RECOVERY_PROCESS_ONLY_NOT_ORIGINAL_TRAINING",
        "original_training_environment_txt_present":original_environment_available,
        "original_training_git_sha_verified":original_sha,
        "python":platform.python_version(),"pandas":pd.__version__,
        "numpy":np.__version__,
        "scikit_learn":__import__("sklearn").__version__,
        "matplotlib":__import__("matplotlib").__version__,
        "source_recovery_git_sha":try_source_sha(repository),
        "recorded_utc":datetime.now(timezone.utc).isoformat()
    }
    save_json(recovery_environment,out/"recovery_environment.json")
    n=int(len(truth))
    cls_idx=classes.index("Normal")
    normal=truth==cls_idx
    if not normal.any():raise ValueError("Missing Normal validation class")
    maxprob=prob.max(axis=1)
    correct=truth==pred
    ece=0.0
    edges=np.linspace(0,1,16)
    for i in range(15):
        mask=(maxprob>=edges[i])&((maxprob<edges[i+1]) if i<14 else
                                      (maxprob<=edges[i+1]))
        if mask.any():
            ece+=mask.mean()*abs(correct[mask].mean()-maxprob[mask].mean())
    auc={}
    for cls_id,label in tqdm(list(enumerate(classes)),desc="Recompute AUC and AP",unit="classes"):
        yy=(truth==cls_id).astype(int)
        if yy.sum()==0 or yy.sum()==n:continue
        auc[label]={"roc_auc":float(roc_auc_score(yy,prob[:,cls_id])),
                    "average_precision":float(average_precision_score(yy,prob[:,cls_id]))}
    best_row=history.sort_values(["val_macro_f1","val_loss","epoch"],
                                 ascending=[False,True,True]).iloc[0]
    best_epoch=int(best_row["epoch"])
    # Independent check that training finished, as opposed to rerunning anything.
    f1=float(f1_score(truth,pred,average="macro",zero_division=0))
    best_f1=float(best_row["val_macro_f1"])
    if abs(f1-best_f1)>5e-4:
        raise ValueError(
            f"Saved best-checkpoint predictions macro-F1 ({f1:.5f}) do not match "
            f"the recorded best epoch ({best_f1:.5f}). Check for mixed outputs.")
    ptrue=np.clip(prob[np.arange(n),truth],1e-12,1.0)
    focal_loss=float(np.mean(-((1-ptrue)**2.0)*np.log(ptrue)))
    training_s=float(history["epoch_seconds"].sum())
    metrics={
      "accuracy":float(accuracy_score(truth,pred)),
      "macro_f1":f1,
      "weighted_f1":float(f1_score(truth,pred,average="weighted",zero_division=0)),
      "balanced_accuracy":float(balanced_accuracy_score(truth,pred)),
      "focal_validation_loss_recomputed_from_saved_probabilities":focal_loss,
      "ece_15_bins":float(ece),
      "normal_to_attack_false_positive_rate":float(np.mean(pred[normal]!=cls_idx)),
      "best_epoch":best_epoch,"best_epoch_history_macro_f1":best_f1,
      "epochs_executed":int(len(history)),
      "training_seconds_sum_logged_epochs":training_s,
      "wall_clock_training_duration_log_only":True,
      "validation_window_count":n,"train_window_count":int(len(dev)-win+1),
      "label_classes":classes,
      "one_vs_rest_auc_ap":auc,
      "metric_scope":"internal validation ONLY",
      "status":"recovered_from_frozen_probabilities"
    }
    # Keep plotting sources, and save diagnostics about uninformative representations.
    print("Rebuilding only frozen-feature PCA projections (no t-SNE, no GPU)...",flush=True)
    details=project_saved_embeddings(out/"embedding_vectors.npz",
        out/"plot_data"/"embedding_coordinates.csv",classes)
    save_json(metrics,out/"metrics_validation.json")
    print("Regenerating publication figures from saved numeric tables...",flush=True)
    figures=Gallery(out).run()
    if figures["figure_count"]!=28:
        raise ValueError("Expected 28 figures, found "+str(figures["figure_count"]))
    hashes=[]
    paths=sorted(checkpoints.glob("scv_*.pt"))+[
      out/"validation_predictions.npz",out/"split_indices.npz",
      out/"embedding_vectors.npz"]
    for path in tqdm(paths,desc="Hash preserved heavyweight artifacts",unit="files"):
        hashes.append({"file":str(path),"sha256":hash_file(path),
                       "size_bytes":path.stat().st_size,
                       "storage":"original Kaggle run; preserve by saving a version"})
    evidence={
      "experiment_id":cfg["experiment_id"],
      "status":"postprocessing_recovered_from_original_training",
      "original_training_source_git_sha":original_sha,
      "original_training_environment_available":original_environment_available,
      "training_provenance_limitation":(
          "Original environment.txt absent; original Python dependency freeze "
          "and exact training code SHA cannot be established from this file set."
          if not original_environment_available else None),
      "recovery_environment_manifest":"recovery_environment.json",
      "recovery_source_git_sha":try_source_sha(repository),
      "original_training_finished_epochs":int(len(history)),
      "source_training_data_sha256":cfg["official_training_sha256"],
      "official_test_used_for_training_validation_or_recovery":False,
      "user_reported_failure":"Native SIGSEGV after t-SNE/PCA warnings in scikit-learn; post-epoch processing.",
      "recovery_algorithm":"Variance-safe deterministic two-dimensional PCA from saved embedding vectors",
      "pca_diagnostics":details,
      "unchanged_saved_predictions_sha256":hash_file(out/"validation_predictions.npz"),
      "unchanged_saved_split_sha256":hash_file(out/"split_indices.npz"),
      "validation_row_count":int(len(val)),
      "development_training_row_count":int(len(dev)),
      "validation_window_count":n,
      "saved_dev_validation_index_overlap":int(len(set(dev)&set(val))),
      "validation_row_indices_sha256":hash_file(out/"split_indices.npz"),
      "validation_performance_from_frozen_probabilities":metrics,
      "heavyweight_files":hashes,
      "software_recovery":{"python":platform.python_version(),
          "pandas":pd.__version__,"numpy":np.__version__,
          "sklearn":__import__("sklearn").__version__},
      "plot_sources":[{
          "file":str(p.relative_to(out)),"sha256":hash_file(p),"size_bytes":p.stat().st_size
      } for p in sorted((out/"plot_data").glob("*.csv"))],
      "figure_manifest_sha256":hash_file(out/"figure_manifest.json"),
      "figure_count":figures["figure_count"],
      "figure_formats":["png","svg","pdf"],
      "note":"No new training, inference, or dataset downloads occurred. "
          "Numerical metrics recomputed from the originally saved best-checkpoint predictions."
    }
    save_json(evidence,out/"artifact_manifest.json")
    report=[
      "# SCV-IDS reproduction — recovered postprocessing", "",
      "**Internal group-disjoint validation only; NO official test evaluation.**",
      "",
      "- Training: 50/50 epochs completed in original Kaggle process.",
      "- This recovery: reuses frozen predictions and embeddings; **no retraining**.",
      f"- Best-checkpoint validation macro-F1: **{f1:.6f}**",
      f"- Accuracy: **{metrics['accuracy']:.6f}**",
      f"- Weighted-F1: **{metrics['weighted_f1']:.6f}**",
      f"- Best checkpoint selected at epoch: **{best_epoch}**",
      f"- Validation window labels: **{n:,}**",
      f"- Preserved checkpoints: **{len(hashes)-3}**",
      f"- Figures: **{figures['figure_count']}** × 3 formats = "
           f"**{figures['figure_count']*3}** exports",
      f"- PCA diagnostics: **{ {k:v['status'] for k,v in details.items()} }**",
      f"- Original training Git SHA: `{original_sha or 'unavailable; original environment file missing'}`",
      f"- Original training environment.txt present: **{original_environment_available}**",
      f"- Recovery code Git SHA: `{evidence['recovery_source_git_sha']}`",
      f"- Original best model checkpoint SHA-256: `{hash_file(checkpoints/'scv_best.pt')}`",
      "",
      "The figure scripts, source CSV tables, metrics and SHA-256 manifest are synchronized",
      "separately. Original checkpoint files, raw per-row probabilities, split indices and",
      "embeddings must be **preserved in a Kaggle saved output version**.",
      "",
      "PCA is intentionally used instead of the native t-SNE routine that crashed.",
      "If a representation has no measurable variation, its plot is explicitly marked",
      "as uninformative; no artificial jitter or fake separation is introduced.",
      "CSV row order remains unvalidated as true chronological flow order.",
      ""
    ]
    (out/"run_report.md").write_text("\n".join(report),encoding="utf-8")
    print("SCV POSTPROCESSING RECOVERY COMPLETE — NO TRAINING",flush=True)
    print("Best validation macro-F1:",round(f1,6),
          "best epoch:",best_epoch,flush=True)
    print("Recovered figure files:",figures["figure_count"]*3,flush=True)
    print("Postprocessing seconds:",round(time.monotonic()-start,1),flush=True)
    print("Artifacts:",out,flush=True)
    return evidence

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoints",required=True)
    parser.add_argument("--repo",required=True)
    args=parser.parse_args()
    recovery(args.output,args.checkpoints,args.repo)
