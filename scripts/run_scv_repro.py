"""SCV-IDS-REPRO-001 — paper-aligned reconstruction on train-only validation.

No official-test CSV is ever read. All experiment output is generated from
executed observations. Model weights and row-level probabilities remain in
private/persistent Kaggle output; plot sources and figures are GitHub-synced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import torch
from torch.utils.data import TensorDataset, DataLoader

from rareids.baseline_utils import (
    canonical_group_ids, group_disjoint_stratified_split, split_indices_sha256
)
from rareids.scv_preprocessing import fit_preprocessor, transform_preprocessor, make_windows
from rareids.scv_model import SCVIDS, MulticlassFocalLoss

def hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for data in iter(lambda: stream.read(4*1024*1024),b""): h.update(data)
    return h.hexdigest()

def save_json(data,path):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def source_git():
    return subprocess.check_output(
        ["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def calculate_validation(model,loader,device,loss_fn):
    model.eval()
    truths,preds,probs=[],[],[]
    total_loss=0.0;total=0
    with torch.no_grad():
        for xi,yi in tqdm(loader,desc="Validation",unit="batches",leave=False):
            x=xi.to(device,non_blocking=True).float().div_(255)
            y=yi.to(device,non_blocking=True)
            z=model(x)
            loss=loss_fn(z,y)
            prob=torch.softmax(z,dim=1).cpu().numpy()
            p=np.argmax(prob,axis=1)
            truths.append(yi.numpy());preds.append(p);probs.append(prob)
            total_loss+=float(loss.item())*len(y);total+=len(y)
    return (np.concatenate(truths),np.concatenate(preds),
            np.concatenate(probs),total_loss/total)

def checkpoint(path,model,optimizer,epoch,best_f1,best_loss,history,generator):
    record={
        "model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "epoch":epoch,"best_f1":best_f1,"best_loss":best_loss,
        "history":history,"torch_rng_state":torch.get_rng_state(),
        "numpy_rng_state":np.random.get_state(),
        "python_rng_state":random.getstate(),
        "data_loader_generator_state":generator.get_state()
    }
    if torch.cuda.is_available():
        record["cuda_rng_state"]=torch.cuda.get_rng_state_all()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    torch.save(record,temporary)
    os.replace(temporary,path)

def resume(path,model,optimizer,generator,device):
    payload=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    generator.set_state(payload["data_loader_generator_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return payload

def sampled_indices(y,classes,seed,max_per_class=120):
    rng=np.random.default_rng(seed)
    sample=[]
    for i in range(len(classes)):
        valid=np.flatnonzero(y==i)
        selected=rng.choice(valid,size=min(max_per_class,len(valid)),replace=False)
        sample.extend(selected.tolist())
    return np.array(sorted(sample),dtype=np.int64)

def metric_plot_sources(out,history,classes,train_targets,val_targets,
                        truth,preds,prob,train_imgs,val_imgs,
                        model,device,val_feats,config):
    plot=out/"plot_data"; plot.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(history).to_csv(plot/"training_history.csv",index=False)
    p,r,f,s=precision_recall_fscore_support(
        truth,preds,labels=np.arange(len(classes)),zero_division=0)
    class_df=pd.DataFrame({"class":classes,"support":s.astype(int),
                           "precision":p,"recall":r,"f1":f})
    class_df.to_csv(plot/"per_class_validation.csv",index=False)
    cm=confusion_matrix(truth,preds,labels=np.arange(len(classes)))
    pd.DataFrame(cm,index=classes,columns=classes).rename_axis(
        "true_class").to_csv(plot/"confusion_validation.csv")
    pd.DataFrame({"class":classes,
                  "train_count":[int((train_targets==i).sum()) for i in range(len(classes))],
                  "validation_count":[int((val_targets==i).sum()) for i in range(len(classes))]
                  }).to_csv(plot/"split_class_counts.csv",index=False)
    # Raw RGB sources for image re-rendering, not actual traffic rows.
    examples=[]
    for i,name in enumerate(classes):
        positions=np.flatnonzero(train_targets==i)
        if len(positions)==0: continue
        image=train_imgs[int(positions[0])].transpose(1,2,0).reshape(256,3)
        for j,(red,green,blue) in enumerate(image):
            examples.append({"class":name,"pixel":j,"r":int(red),
                             "g":int(green),"b":int(blue)})
    pd.DataFrame(examples).to_csv(plot/"rgb_sample_pixels.csv",index=False)
    confidence=prob.max(axis=1)
    correct=truth==preds
    edges=np.linspace(0,1,16);data=[]
    ece=0.0
    for i in range(15):
        mask=(confidence>=edges[i])&((confidence<edges[i+1]) if i<14 else
                                        (confidence<=edges[i+1]))
        count=int(mask.sum())
        mean=float(confidence[mask].mean()) if count else np.nan
        acc=float(correct[mask].mean()) if count else np.nan
        if count:ece+=(count/len(confidence))*abs(acc-mean)
        if count:data.append({"bin":i,"n":count,"left":edges[i],
                              "right":edges[i+1],"mean_confidence":mean,
                              "fraction_correct":acc})
    pd.DataFrame(data).to_csv(plot/"reliability_bins.csv",index=False)
    bins=np.linspace(0,1,21)
    good,_=np.histogram(confidence[correct],bins=bins)
    bad,_=np.histogram(confidence[~correct],bins=bins)
    pd.DataFrame({"bin_center":(bins[1:]+bins[:-1])/2,
                  "correct":good,"incorrect":bad}).to_csv(
                      plot/"confidence_histogram.csv",index=False)
    ovr=[];auc_info={}
    for i,name in enumerate(classes):
        yy=(truth==i).astype(np.uint8)
        if yy.sum()==0 or yy.sum()==len(yy):continue
        fpr,tpr,_=roc_curve(yy,prob[:,i])
        precision,recall,_=precision_recall_curve(yy,prob[:,i])
        auc_info[name]={"roc_auc":float(roc_auc_score(yy,prob[:,i])),
                        "average_precision":float(average_precision_score(yy,prob[:,i]))}
        # Dense saved curve points are monotonically downsampled only for vector clarity.
        for kind,xx,yy_curve in (("roc",fpr,tpr),("pr",recall,precision)):
            ix=np.unique(np.linspace(0,len(xx)-1,min(301,len(xx)),dtype=int))
            for j in ix:
                ovr.append({"class":name,"kind":kind,
                            "x":float(xx[j]),"y":float(yy_curve[j])})
    pd.DataFrame(ovr).to_csv(plot/"roc_pr_curves.csv",index=False)

    sample=sampled_indices(truth,classes,config["seed"],
                           config["figure_generation"]["tsne_sample_per_class_max"])
    sample_t=torch.from_numpy(val_imgs[sample])
    reprs={"cnn":[],"vit":[],"fused":[]}
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0,len(sample),64),
                          desc="Extract saved t-SNE features",unit="batches"):
            chunk=sample_t[start:start+64].to(device).float().div_(255)
            _,feat=model(chunk,return_features=True)
            for key in reprs:reprs[key].append(feat[key].cpu().numpy())
    embeddings={k:np.concatenate(v) for k,v in reprs.items()}
    embeddings["raw"]=val_feats[sample+config["preprocessing"]["window"]-1]
    # Keep raw embeddings in persistent Kaggle-only NPZ as independent evidence.
    np.savez_compressed(out/"embedding_vectors.npz",sample_index=sample,
                        y=truth[sample],**embeddings)
    # The original run used scikit-learn t-SNE and crashed after saving
    # embedding_vectors.npz on a degenerate feature map. Do not silently
    # retry that native implementation; use deterministic variance-safe PCA.
    from rareids.safe_projection import project_saved_embeddings
    project_saved_embeddings(out/"embedding_vectors.npz",
                             plot/"embedding_coordinates.csv",classes)
    return class_df,cm,float(ece),auc_info

def main(args):
    root=Path(args.repo);out=Path(args.output);weights=Path(args.checkpoints)
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["stage"]!="training_only_validation" or config["official_test_csv"]!="DO_NOT_READ_DURING_THIS_EXPERIMENT":
        raise ValueError("Official test access forbidden during Step 7")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in Kaggle settings")
    device=torch.device("cuda:0")
    seed_everything(config["seed"])
    dataset_path=Path(config["official_training_csv"])
    if not dataset_path.is_file():raise FileNotFoundError(str(dataset_path))
    if hash_file(dataset_path)!=config["official_training_sha256"]:
        raise ValueError("Source dataset SHA-256 mismatches verified manifest")
    out.mkdir(parents=True,exist_ok=True);weights.mkdir(parents=True,exist_ok=True)
    save_json(config,out/"config_effective.json")
    # Full resolved environment, preserving dependency details for later figure edits.
    try:
        freeze=subprocess.check_output(
            [os.sys.executable,"-m","pip","freeze"],
            text=True,stderr=subprocess.DEVNULL,timeout=90)
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        freeze="DEPENDENCY FREEZE UNAVAILABLE; see artifact_manifest.json"
    (out/"environment.txt").write_text(
        "Python: "+platform.python_version()+"\n"
        "PyTorch: "+str(torch.__version__)+"\n"
        "CUDA build: "+str(torch.version.cuda)+"\n"
        "Git commit: "+source_git()+"\n\n"
        "=== pip freeze ===\n"+freeze,encoding="utf-8")
    frame=pd.read_csv(dataset_path,low_memory=False)
    if len(frame)!=175341 or frame.shape[1]!=45:
        raise ValueError("Designated training partition has unexpected size/schema")
    excluded={"id","attack_cat","label"}
    features=[c for c in frame.columns if c not in excluded]
    if len(features)!=42:raise ValueError("Expected 42 benchmark traffic features")
    categories=sorted(frame["attack_cat"].astype(str).str.strip().unique().tolist())
    if len(categories)!=10:raise ValueError("Expected 10 labels")
    labels=frame["attack_cat"].astype(str).str.strip().map(
        {name:i for i,name in enumerate(categories)}).to_numpy(dtype=np.int64)
    # Same feature grouping rule as the independently verified tabular baseline.
    groups=canonical_group_ids(frame,features)
    tr_idx,va_idx,fold=group_disjoint_stratified_split(
        labels,groups,n_splits=config["validation"]["n_splits"],seed=config["seed"])
    if len(set(groups[tr_idx])&set(groups[va_idx]))!=0:raise AssertionError("Group leakage")
    prep=fit_preprocessor(frame,features,tr_idx)
    train_values,train_out=transform_preprocessor(prep,frame.iloc[tr_idx])
    val_values,val_out=transform_preprocessor(prep,frame.iloc[va_idx])
    save_json(prep.as_dict(),out/"preprocessing.json")
    window=config["preprocessing"]["window"]
    train_images,train_labels=make_windows(train_values,labels[tr_idx],window,
                                           progress=tqdm)
    val_images,val_labels=make_windows(val_values,labels[va_idx],window,
                                       progress=tqdm)
    np.savez_compressed(out/"split_indices.npz",dev_train_indices=tr_idx,
                        validation_indices=va_idx,dev_train_labels=labels[tr_idx],
                        validation_labels=labels[va_idx],classes=np.array(categories))
    del train_values # val_values retained for stratified t-SNE

    dataset=TensorDataset(torch.from_numpy(train_images),torch.from_numpy(train_labels))
    valset=TensorDataset(torch.from_numpy(val_images),torch.from_numpy(val_labels))
    generator=torch.Generator().manual_seed(config["seed"])
    num_workers=config["training"]["num_workers"]
    loader=DataLoader(dataset,batch_size=config["training"]["batch_size"],
                      shuffle=True,generator=generator,
                      num_workers=num_workers,pin_memory=True,drop_last=False)
    valid_loader=DataLoader(valset,batch_size=config["training"]["batch_size"],
                            shuffle=False,num_workers=num_workers,pin_memory=True)
    model=SCVIDS(n_classes=len(categories),
                 dropout=config["model"]["dropout"],
                 patch=config["model"]["vit_patch_size"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["training"]["learning_rate"],
                                weight_decay=config["training"]["weight_decay"])
    loss_fn=MulticlassFocalLoss(gamma=2.0)
    history=[];best_f1=-1.0;best_loss=float("inf");start_epoch=1
    latest=weights/"scv_last_resume.pt"
    best=weights/"scv_best.pt"
    if args.resume and latest.is_file():
        restored=resume(latest,model,optimizer,generator,device)
        history=restored["history"];best_f1=restored["best_f1"]
        best_loss=restored["best_loss"];start_epoch=restored["epoch"]+1
        print("Resuming at epoch",start_epoch,flush=True)
    print("Training only: train windows",len(train_labels),
          "validation windows",len(val_labels),"fold",fold,
          "group intersection 0",flush=True)
    total_start=time.perf_counter()
    for epoch in tqdm(range(start_epoch,config["training"]["epochs"]+1),
                      desc="SCV-IDS epochs",unit="epochs"):
        model.train();epoch_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        sum_loss=0.0;sum_correct=0;sum_seen=0
        for xi,yi in tqdm(loader,desc=f"Epoch {epoch:02d} training",leave=False,
                          unit="batches"):
            x=xi.to(device,non_blocking=True).float().div_(255)
            y=yi.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=loss_fn(logits,y)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            sum_loss+=float(loss.detach())*len(y)
            sum_correct+=int((logits.argmax(1)==y).sum())
            sum_seen+=len(y)
        truths,preds,probs,vloss=calculate_validation(model,valid_loader,device,loss_fn)
        macro=float(f1_score(truths,preds,average="macro",zero_division=0))
        weighted=float(f1_score(truths,preds,average="weighted",zero_division=0))
        record={"epoch":epoch,"train_loss":sum_loss/sum_seen,
                "val_loss":vloss,"train_accuracy":sum_correct/sum_seen,
                "val_accuracy":float(accuracy_score(truths,preds)),
                "val_macro_f1":macro,"val_weighted_f1":weighted,
                "learning_rate":optimizer.param_groups[0]["lr"],
                "epoch_seconds":time.perf_counter()-epoch_start,
                "peak_gpu_memory_mb":torch.cuda.max_memory_allocated(device)/1048576}
        history.append(record)
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
        better=macro>best_f1+1e-12 or (abs(macro-best_f1)<1e-12 and vloss<best_loss)
        if better:
            best_f1,best_loss=macro,vloss
            checkpoint(best,model,optimizer,epoch,best_f1,best_loss,history,generator)
        checkpoint(latest,model,optimizer,epoch,best_f1,best_loss,history,generator)
        if epoch%config["training"]["checkpoint_every"]==0:
            checkpoint(weights/f"scv_epoch_{epoch:03d}.pt",model,optimizer,
                       epoch,best_f1,best_loss,history,generator)
        print(f"EPOCH {epoch:02d}/{config['training']['epochs']} "
              f"train_loss={record['train_loss']:.5f} val_loss={vloss:.5f} "
              f"val_macro_f1={macro:.5f} best={best_f1:.5f} "
              f"val_accuracy={record['val_accuracy']:.5f}",flush=True)

    if not best.is_file():raise RuntimeError("No best checkpoint generated")
    best_snapshot=torch.load(best,map_location=device,weights_only=False)
    model.load_state_dict(best_snapshot["model"])
    true,pred,prob,loss=calculate_validation(model,valid_loader,device,loss_fn)
    np.savez_compressed(out/"validation_predictions.npz",y_true=true.astype(np.uint8),
                        y_pred=pred.astype(np.uint8),probabilities=prob.astype(np.float32),
                        validation_window_end_indices=va_idx[window-1:].astype(np.int32))
    class_df,cm,ece,auc_info=metric_plot_sources(
        out,history,categories,train_labels,val_labels,true,pred,prob,
        train_images,val_images,model,device,val_values,config)

    # Copy raw history as a self-contained primary result artifact.
    metrics={
        "macro_f1":float(f1_score(true,pred,average="macro",zero_division=0)),
        "weighted_f1":float(f1_score(true,pred,average="weighted",zero_division=0)),
        "accuracy":float(accuracy_score(true,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(true,pred)),
        "focal_validation_loss":float(loss),"ece_15_bins":ece,
        "normal_to_attack_false_positive_rate":float(
            np.mean(pred[true==categories.index("Normal")]!=categories.index("Normal"))),
        "best_epoch":int(best_snapshot["epoch"]),
        "epochs_executed":len(history),
        "training_seconds_total":float(time.perf_counter()-total_start),
        "parameter_count":int(sum(p.numel() for p in model.parameters())),
        "validation_window_count":int(len(true)),
        "train_window_count":int(len(train_labels)),
        "label_classes":categories,"one_vs_rest_auc_ap":auc_info}
    save_json(metrics,out/"metrics_validation.json")
    # Save durable checkpoint- and data-level provenance after plotting.
    snapshots=sorted(weights.glob("scv_*.pt"))
    evidence={
        "experiment_id":config["experiment_id"],"status":"completed_unverified",
        "observed_utc":datetime.now(timezone.utc).isoformat(),
        "source_git_sha":source_git(),
        "data_source":str(dataset_path),"train_sha256":hash_file(dataset_path),
        "official_test_predictions":False,
        "official_test_sha256_reference_only":config["official_test_sha256_reference_only"],
        "split":{"seed":config["seed"],"fold":int(fold),
                 "grouping":"SHA256 exact 42 original parsed features",
                 "group_overlap":0,
                 "row_indices_sha256":split_indices_sha256(tr_idx,va_idx),
                 "train_rows":int(len(tr_idx)),"validation_rows":int(len(va_idx)),
                 "train_windows":int(len(train_labels)),
                 "validation_windows":int(len(val_labels))},
        "categorical_unknown_counts_note":"unknown categories index-encoded and clipped with train-only MinMax",
        "out_of_range_feature_values":{"train":int(train_out),"validation":int(val_out)},
        "software":{"python":platform.python_version(),"torch":torch.__version__,
                    "numpy":np.__version__,"pandas":pd.__version__,
                    "scikit_learn":__import__("sklearn").__version__,
                    "matplotlib":__import__("matplotlib").__version__,
                    "cuda":torch.version.cuda},
        "gpu":{"name":torch.cuda.get_device_name(device),
               "max_memory_allocated_mb":float(torch.cuda.max_memory_allocated(device)/1048576)},
        "artifacts":[{"file":str(p),"sha256":hash_file(p),"bytes":p.stat().st_size,
                      "storage":"Kaggle saved notebook output only"}
                     for p in snapshots+[out/"validation_predictions.npz",
                                          out/"split_indices.npz",
                                          out/"embedding_vectors.npz"]],
        "metric_scope":"internal group-disjoint validation ONLY",
        "paper_deviations":[
           "The paper does not fully specify normalized float-to-RGB 24-bit rounding; numpy.rint is used.",
           "The paper does not fully specify attention projection/classifier dimensions; these are documented independent reconstruction choices.",
           "20% internal validation was created with exact-feature group disjointness, versus unspecified paper validation split.",
           "CSV row-order windows are not verified chronological.",
           "Normalization/vocabulary fit on development training ONLY (no validation leakage).",
           "focal gamma=2 and no alpha class weights are a documented choice; paper does not identify these.",
           "No official test inference has been performed."
        ]
    }
    save_json(evidence,out/"artifact_manifest.json")
    from plot_scv_figures import Gallery
    gallery=Gallery(out)
    figmanifest=gallery.run()
    # Re-run manifest hashing after figures are completely finalized.
    plot_files=sorted((out/"plot_data").glob("*.csv"))
    rows=[{"path":str(p.relative_to(out)),"sha256":hash_file(p),"bytes":p.stat().st_size}
          for p in plot_files]
    evidence["plot_sources"]=rows
    evidence["figure_manifest_sha256"]=hash_file(out/"figure_manifest.json")
    evidence["figure_count"]=figmanifest["figure_count"]
    save_json(evidence,out/"artifact_manifest.json")
    report=["# SCV-IDS paper-aligned reconstruction: validation only","",
      "**This is not an official-test result, nor verified author-code reproduction.**",
      f"- Validation macro-F1: **{metrics['macro_f1']:.6f}**",
      f"- Validation accuracy: **{metrics['accuracy']:.6f}**",
      f"- Validation weighted-F1: **{metrics['weighted_f1']:.6f}**",
      f"- Best epoch: **{metrics['best_epoch']} / {len(history)}**",
      f"- Window-labelled development train/validation: **{len(train_labels):,} / {len(val_labels):,}**",
      f"- Shared original 42-feature groups across development split: **0**",
      f"- Model parameters: **{metrics['parameter_count']:,}**",
      f"- Figures generated: **{figmanifest['figure_count']}**, each in PNG, SVG and PDF",
      f"- Official test inference: **NO**",
      f"- Kaggle GPU: **{evidence['gpu']['name']}**",
      f"- Source Git SHA: \`{evidence['source_git_sha']}\`",
      f"- Checkpoint best SHA-256: \`{hash_file(best)}\`",
      "",
      "The full per-window validation probabilities, embedding vectors, split indices and",
      "replayable model checkpoints are saved under Kaggle working outputs.",
      "These binaries must be preserved by **Save Version > Save & Run All**.",
      "All editable plotting tables and the three publication formats are prepared for GitHub synchronization.",
      ""]
    (out/"run_report.md").write_text("\n".join(report),encoding="utf-8")
    print("\nSTEP 7 COMPLETE (VALIDATION ONLY)")
    print(json.dumps({k:v for k,v in metrics.items() if k in (
      "macro_f1","weighted_f1","accuracy","best_epoch","parameter_count","validation_window_count")},
      indent=2))
    print("Figures generated:",figmanifest["figure_count"],
          "formats: PNG, SVG, PDF; preserve Kaggle output checkpoints.")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoints",required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    main(args)
