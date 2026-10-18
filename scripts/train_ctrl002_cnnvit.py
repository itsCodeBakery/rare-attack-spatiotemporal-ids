"""CTRL-CNNVIT-V2-S42: original split-matched CNN activation repair control.

Only designated UNSW TRAIN CSV is opened. Original group-disjoint indices, class
IDs, vocab/scaler and validation-target sequence are reused byte-for-byte.
All optimizer checkpoints and predictions stay in Kaggle, never public Git.
The original CNN-ViT model is never loaded or modified by this training script.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score
from tqdm.auto import tqdm
from rareids.v2_model import SCVIDSLeakyFinalCNN
from rareids.scv_preprocessing import FittedPreprocessor,transform_preprocessor,make_windows
from rareids.scv_model import MulticlassFocalLoss
from rareids.baseline_utils import split_indices_sha256


def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(4194304),b""):h.update(chunk)
    return h.hexdigest()


def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def git_head(repo):
    return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
            text=True,stderr=subprocess.DEVNULL).strip()


def checkpoint(path,model,optimizer,epoch,history,best_f1,best_loss,generator):
    obj={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
         "epoch":epoch,"history":history,"best_macro_f1":best_f1,
         "best_validation_focal_loss":best_loss,
         "rng_numpy":np.random.get_state(),"rng_python":random.getstate(),
         "rng_torch":torch.get_rng_state(),
         "generator_state":generator.get_state()}
    if torch.cuda.is_available():obj["rng_cuda"]=torch.cuda.get_rng_state_all()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_suffix(".tmp")
    torch.save(obj,part);os.replace(part,path)


def restore(path,model,optimizer,generator,device):
    obj=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(obj["model"],strict=True)
    optimizer.load_state_dict(obj["optimizer"])
    np.random.set_state(obj["rng_numpy"])
    random.setstate(obj["rng_python"])
    torch.set_rng_state(obj["rng_torch"])
    generator.set_state(obj["generator_state"])
    if torch.cuda.is_available() and "rng_cuda" in obj:
        torch.cuda.set_rng_state_all(obj["rng_cuda"])
    return obj


def prepare(cfg,reference):
    train_source=Path(cfg["source_dataset"])
    if not train_source.is_file():raise FileNotFoundError(str(train_source))
    if digest(train_source)!=cfg["source_sha256"]:
        raise ValueError("Source SHA-256 mismatch; refuses alternate UNSW data")
    split_file=reference/"split_indices.npz"
    frozen_preds=reference/"validation_predictions.npz"
    if digest(split_file)!=cfg["reference_split_sha256"]:
        raise ValueError("Original train/validation indices do not match archived reference")
    if digest(frozen_preds)!=cfg["reference_predictions_sha256"]:
        raise ValueError("Original held-out validation predictions were altered")
    with np.load(split_file,allow_pickle=False) as splits:
        train_idx=splits["dev_train_indices"].astype(np.int64)
        val_idx=splits["validation_indices"].astype(np.int64)
        train_y=splits["dev_train_labels"].astype(np.int64)
        val_y=splits["validation_labels"].astype(np.int64)
        classes=[str(x) for x in splits["classes"].tolist()]
    with np.load(frozen_preds,allow_pickle=False) as preds:
        original_y=preds["y_true"].astype(np.int64)
        validation_end_rows=preds["validation_window_end_indices"].astype(np.int64)
    if len(classes)!=10 or len(train_idx)!=140332 or len(val_idx)!=35009:
        raise ValueError("Source benchmark split row counts/class vocabulary not as recorded")
    if len(np.intersect1d(train_idx,val_idx)):
        raise ValueError("Original development rows overlap")
    w=cfg["input_window"]
    if (w!=5 or len(original_y)!=cfg["reference_validation_windows"] or
        not np.array_equal(val_y[w-1:],original_y) or
        not np.array_equal(val_idx[w-1:],validation_end_rows)):
        raise ValueError("Validation-window target/lineage does not match original SCV run")
    frame=pd.read_csv(train_source,low_memory=False)
    if len(frame)!=175341 or len(frame.columns)!=45:
        raise ValueError("Expected UNSW 175341 x 45 official training CSV")
    mapped=frame["attack_cat"].astype(str).str.strip().map(
        {name:i for i,name in enumerate(classes)}).to_numpy()
    if not np.array_equal(mapped[train_idx],train_y) or not np.array_equal(mapped[val_idx],val_y):
        raise ValueError("Original target labels mismatch")
    state=json.loads((reference/"preprocessing.json").read_text(encoding="utf-8"))
    pp=FittedPreprocessor(**{k:state[k] for k in
                          ("columns","categorical","vocabs","min_values","max_values")})
    if len(pp.columns)!=42:raise ValueError("Missing original 42 input features")
    train_x,out_tr=transform_preprocessor(pp,frame.iloc[train_idx])
    valid_x,out_va=transform_preprocessor(pp,frame.iloc[val_idx])
    del frame
    images_train,target_train=make_windows(train_x,train_y,window=w,progress=tqdm)
    images_valid,target_valid=make_windows(valid_x,val_y,window=w,progress=tqdm)
    if not np.array_equal(target_valid,original_y):
        raise ValueError("Control and original window label targets differ")
    return images_train,target_train,images_valid,target_valid,classes,{
        "development_train_row_count":len(train_idx),
        "development_validation_row_count":len(val_idx),
        "train_window_count":len(target_train),"validation_window_count":len(target_valid),
        "frozen_reference_split_sha256":digest(split_file),
        "frozen_reference_predictions_sha256":digest(frozen_preds),
        "row_index_digest":split_indices_sha256(train_idx,val_idx),
        "train_clipped_values":out_tr,"validation_clipped_values":out_va,
        "window_target_equal_to_original":True,
        "validated_test_access":False}


def evaluate(model,loader,device,loss_fn):
    model.eval()
    true,pred,probs=[],[],[]
    loss_total=0.;n=0
    with torch.no_grad():
        for x,y in tqdm(loader,desc="Internal validation",leave=False,unit="batches"):
            xx=x.to(device,non_blocking=True).float().div_(255)
            yy=y.to(device,non_blocking=True)
            logits=model(xx)
            if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite logits")
            loss=loss_fn(logits,yy)
            prob=torch.softmax(logits,dim=1).cpu().numpy()
            true.append(y.numpy());pred.append(prob.argmax(1));probs.append(prob)
            n+=len(y);loss_total+=float(loss)*len(y)
    return np.concatenate(true),np.concatenate(pred),np.concatenate(probs),loss_total/n


def select_health_sample(labels,seed,size):
    """Deterministic, rare-class-aware internal validation subsample."""
    rng=np.random.default_rng(seed)
    selected=[]
    for cls in np.unique(labels):
        positions=np.flatnonzero(labels==cls)
        selected.extend(rng.choice(positions,size=min(24,len(positions)),
                                   replace=False).tolist())
    anchors=np.unique(np.asarray(selected,dtype=np.int64))
    remaining=np.setdiff1d(np.arange(len(labels),dtype=np.int64),anchors)
    extra=rng.choice(remaining,size=min(max(0,size-len(anchors)),len(remaining)),
                     replace=False)
    result=np.sort(np.concatenate([anchors,extra]))
    if len(result)!=size:raise ValueError("Cannot generate fixed health subset")
    return result


def cnn_health(model,validation_images,indices,device,batch_size):
    """Frozen-evaluation activation variability and zero-CNN intervention."""
    model.eval()
    tokens=[];pool=[];deltas=[];changes=0;seen=0
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            positions=indices[start:start+batch_size]
            x=torch.from_numpy(validation_images[positions]).to(device).float().div_(255.)
            y,c,v=model.forward_branches(x)
            mask,_,_=model.forward_branches(x,zero_cnn=True)
            tokens.append(c.cpu().numpy())
            pool.append(c.mean(1).cpu().numpy())
            deltas.extend(torch.linalg.vector_norm(y-mask,dim=1).cpu().numpy().tolist())
            changes+=int((y.argmax(1)!=mask.argmax(1)).sum().item())
            seen+=len(positions)
    enc=np.concatenate(tokens).astype(np.float64)
    pooled=np.concatenate(pool).astype(np.float64)
    variable=enc.reshape(seen,-1).std(axis=0)
    pstd=pooled.std(axis=0)
    result={
        "health_sample_size":int(seen),
        "cnn_token_nonzero_fraction":float(np.mean(np.abs(enc)>1e-12)),
        "cnn_token_between_sample_mean_std":float(variable.mean()),
        "cnn_constant_token_fraction":float(np.mean(variable<=1e-8)),
        "cnn_pooled_between_sample_mean_std":float(pstd.mean()),
        "cnn_pooled_constant_dimension_fraction":float(np.mean(pstd<=1e-8)),
        "cnn_zero_mask_mean_logit_l2":float(np.mean(deltas)),
        "cnn_zero_mask_changed_prediction_fraction":float(changes/seen),
        "cnn_token_mean_absolute":float(np.mean(np.abs(enc)))
    }
    if not all(np.isfinite(x) for x in result.values()):
        raise FloatingPointError("Nonfinite measured CNN health result")
    return result


def main(repo,config_path,out,weights):
    repo,out,weights=Path(repo),Path(out),Path(weights)
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("experiment_id")!="CTRL-CNNVIT-V2-S42" or cfg.get("official_test")!="FORBIDDEN":
        raise ValueError("Invalid experiment/test access configuration")
    if not torch.cuda.is_available():raise RuntimeError("Enable GPU for Kaggle CNN ViT control")
    seed=int(cfg["seed"])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda:0")
    out.mkdir(parents=True,exist_ok=True);weights.mkdir(parents=True,exist_ok=True)
    status_path=out/"status.json"
    if status_path.exists():
        status=json.loads(status_path.read_text())
        if status.get("completed"):
            print("RUN ALREADY FINISHED — no repeated training. Use finalize cell.")
            return
    reference=Path("/kaggle/working/research_outputs/SCV-IDS-REPRO-001")
    imgs_tr,y_tr,imgs_va,y_va,classes,lineage=prepare(cfg,reference)
    save_json(out/"config_effective.json",cfg)
    save_json(out/"data_lineage.json",lineage)
    # Freeze package versions during ORIGINAL new control training, not after recovery.
    try:
        freeze=subprocess.check_output(
            [os.sys.executable,"-m","pip","freeze"],
            text=True,stderr=subprocess.DEVNULL,timeout=90)
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        freeze="pip freeze unavailable"
    (out/"training_environment.txt").write_text(
        "Capture scope: ORIGINAL CTRL-CNNVIT-V2-S42 training\n"
        + "Python: "+platform.python_version()+"\n"
        + "Torch: "+str(torch.__version__)+"\n"
        + "CUDA: "+str(torch.version.cuda)+"\n"
        + "Git commit: "+git_head(repo)+"\n\n"
        + freeze+"\n",encoding="utf-8")
    (out/"source_code_commit.txt").write_text(git_head(repo)+"\n")
    selected_health=select_health_sample(y_va,int(cfg["health"]["sample_seed"]),
                   int(cfg["health"]["sample_size"]))
    np.savez_compressed(out/"health_sample_indices.npz",
                        window_indices=selected_health)
    tr_set=TensorDataset(torch.from_numpy(imgs_tr),torch.from_numpy(y_tr))
    va_set=TensorDataset(torch.from_numpy(imgs_va),torch.from_numpy(y_va))
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(tr_set,batch_size=cfg["batch_size"],shuffle=True,
       generator=generator,num_workers=2,pin_memory=True)
    val_loader=DataLoader(va_set,batch_size=cfg["batch_size"],shuffle=False,
       num_workers=2,pin_memory=True)
    model=SCVIDSLeakyFinalCNN(n_classes=10,dropout=cfg["dropout"],negative_slope=cfg["negative_slope"]).to(device)
    count=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),
                lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    loss_fn=MulticlassFocalLoss(gamma=2.)
    best=-1.;best_loss=float("inf")
    history=[];first=1
    latest=weights/"last_resume.pt"
    best_file=weights/"best_val_macro_f1.pt"
    if latest.is_file():
        saved=restore(latest,model,optimizer,generator,device)
        history=saved["history"];first=int(saved["epoch"])+1
        best=saved["best_macro_f1"];best_loss=saved["best_validation_focal_loss"]
        print("Resuming frozen control from epoch",first)
    for epoch in tqdm(range(first,cfg["epochs"]+1),desc="CNN-ViT LeakyReLU control epochs",unit="epochs"):
        start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        seen=0;loss_sum=0.;correct=0
        for xx,yy in tqdm(loader,desc=f"Epoch {epoch:02d} training",
                          leave=False,unit="batches"):
            x=xx.to(device,non_blocking=True).float().div_(255)
            y=yy.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=loss_fn(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite focal loss")
            loss.backward();optimizer.step()
            n=len(y);seen+=n
            correct+=int((logits.argmax(1)==y).sum().item())
            loss_sum+=float(loss.detach().item())*n
        true,pred,prob,vloss=evaluate(model,val_loader,device,loss_fn)
        h=cnn_health(model,imgs_va,selected_health,device,cfg["batch_size"])
        record={"epoch":epoch,"train_loss":loss_sum/seen,
             "val_loss":vloss,"train_accuracy":correct/seen,
             "val_accuracy":float(accuracy_score(true,pred)),
             "val_macro_f1":float(f1_score(true,pred,average="macro",zero_division=0)),
             "val_weighted_f1":float(f1_score(true,pred,average="weighted",zero_division=0)),
             "epoch_seconds":time.perf_counter()-start,
             "allocated_gpu_peak_mib":torch.cuda.max_memory_allocated(device)/1048576,
             "learning_rate":optimizer.param_groups[0]["lr"],**h}
        history.append(record)
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
        better=(record["val_macro_f1"]>best+1e-12 or
                (abs(record["val_macro_f1"]-best)<=1e-12 and vloss<best_loss))
        if better:
            best=record["val_macro_f1"];best_loss=vloss
            checkpoint(best_file,model,optimizer,epoch,history,best,best_loss,generator)
        checkpoint(latest,model,optimizer,epoch,history,best,best_loss,generator)
        if epoch%cfg["checkpoint_every"]==0:
            checkpoint(weights/f"epoch_{epoch:03d}.pt",model,optimizer,epoch,
                       history,best,best_loss,generator)
        print(f"CTRL-V2 EPOCH {epoch:02d}/{cfg['epochs']} "
              f"train_loss={record['train_loss']:.5f} "
              f"val_macro_f1={record['val_macro_f1']:.5f} "
              f"val_accuracy={record['val_accuracy']:.5f} "
              f"best_macro_f1={best:.5f} "
              f"cnn_nonzero={h['cnn_token_nonzero_fraction']:.4f} "
              f"cnn_variation={h['cnn_token_between_sample_mean_std']:.6f} "
              f"cnn_mask_L2={h['cnn_zero_mask_mean_logit_l2']:.5f}",flush=True)
    if not best_file.is_file():raise RuntimeError("Best checkpoint not generated")
    model.load_state_dict(torch.load(best_file,map_location=device,
                                    weights_only=False)["model"],strict=True)
    tr,pr,pb,vl=evaluate(model,val_loader,device,loss_fn)
    best_health=cnn_health(model,imgs_va,selected_health,device,cfg["batch_size"])
    healthy=(best_health["cnn_token_nonzero_fraction"]>=
              cfg["health"]["gate_min_nonzero_fraction"] and
             best_health["cnn_token_between_sample_mean_std"]>=
              cfg["health"]["gate_min_between_example_feature_std"])
    save_json(out/"health_summary.json",{
        "activation_health_pass":bool(healthy),**best_health,
        "note":"Predeclared health gate; failed health remains a valid, reported negative result."})
    if not np.array_equal(tr,y_va):raise ValueError("Validation targets drifted")
    np.savez_compressed(out/"best_validation_predictions.npz",
      y_true=tr.astype(np.uint8),y_pred=pr.astype(np.uint8),
      probabilities=pb.astype(np.float32))
    status={"completed":True,"epochs":len(history),"best_macro_f1":best,
            "best_checkpoint_epoch":int(torch.load(best_file,map_location="cpu",
                            weights_only=False)["epoch"]),
            "class_names":classes,"parameter_count":count,
            "validation_windows":len(tr),"train_windows":len(y_tr),
            "checkpoint_sha256":digest(best_file),
            "predictions_sha256":digest(out/"best_validation_predictions.npz"),
            "official_test_used":False,
            "activation_health_pass":bool(healthy),
            "best_cnn_health":best_health,
            "note":"Minimal CNN activation-only engineering control, single seed; official test locked"}
    save_json(status_path,status)
    print("CTRL-CNNVIT-V2 TRAINING COMPLETE — no official test")
    print(json.dumps(status,indent=2))
    print("NEXT: run finalize-only notebook cell, never repeat 50 epochs for plots.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("repo","config","output","checkpoints"):
        ap.add_argument("--"+name,required=True)
    a=ap.parse_args()
    main(a.repo,a.config,a.output,a.checkpoints)
