"""DIAG-001 — frozen CNN, attention, and branch-zeroing audit on internal validation.

No model training, gradients, fit(), official-test access, calibration or parameter
updates. Numeric summaries + provenance only; original model and full predictions
stay in the private draft GitHub Release archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from tqdm.auto import tqdm
from rareids.scv_model import SCVIDS
from rareids.scv_preprocessing import FittedPreprocessor,transform_preprocessor,windows_to_rgb


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(4194304),b""): h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def metric_rows(y,p,classes,mode):
    pr,re,f,s=precision_recall_fscore_support(y,p,labels=np.arange(10),zero_division=0)
    overall={"mode":mode,"count":int(len(y)),
             "accuracy":float(accuracy_score(y,p)),
             "macro_f1":float(f1_score(y,p,average="macro",zero_division=0))}
    detail=[{"mode":mode,"class":name,"support":int(s[i]),
             "precision":float(pr[i]),"recall":float(re[i]),"f1":float(f[i])}
             for i,name in enumerate(classes)]
    return overall,detail


class CNNHooks:
    def __init__(self,model):
        self.values={}
        self.active=True
        self.handles=[]
        for name,layer in model.cnn.layers.named_children():
            if isinstance(layer,(torch.nn.Conv2d,torch.nn.ReLU,torch.nn.MaxPool2d)):
                self.values[name]=[]
                self.handles.append(layer.register_forward_hook(self._hook(name)))
    def _hook(self,name):
        def record(module,inputs,out):
            if not self.active:return
            z=out.detach().float()
            self.values[name].append({
              "mean":z.mean((2,3)).cpu().numpy(),
              "spatial":z.std((2,3),unbiased=False).mean(0).cpu().numpy(),
              "zero":(z==0).float().mean((0,2,3)).cpu().numpy(),
              "n":len(z)})
        return record
    def close(self):
        for hook in self.handles:hook.remove()
    def summarize(self):
        layers,channels=[],[]
        for name,items in self.values.items():
            means=np.concatenate([x["mean"] for x in items])
            total=sum(x["n"] for x in items)
            sd=means.std(axis=0)
            spatial=sum((x["spatial"]*x["n"] for x in items))/total
            zeros=sum((x["zero"]*x["n"] for x in items))/total
            for i in range(len(sd)):
                channels.append({"layer":name,"channel":int(i),
                    "between_image_std":float(sd[i]),
                    "within_image_spatial_std":float(spatial[i]),
                    "zero_fraction":float(zeros[i])})
            layers.append({"layer":name,"channels":int(len(sd)),
                "between_image_mean_std":float(sd.mean()),
                "max_channel_std":float(sd.max()),
                "constant_fraction":float(np.mean(sd<=1e-8)),
                "spatial_std":float(spatial.mean()),
                "zero_fraction":float(zeros.mean())})
        return layers,channels


def fusion_logits(model,v,c):
    a,_=model.vit_to_cnn(v,c,c,need_weights=False)
    b,_=model.cnn_to_vit(c,v,v,need_weights=False)
    z=model.fusion(torch.cat([a.mean(1),b.mean(1)],dim=1))
    return model.classifier(z),a,b,z


def describe_embeddings(storage):
    rows=[]
    for name,chunks in storage.items():
        values=np.concatenate(chunks).astype(np.float64)
        spread=values.std(axis=0)
        rows.append({"branch":name,"dimensions":int(values.shape[1]),
            "examples":len(values),"mean_feature_std":float(spread.mean()),
            "max_feature_std":float(spread.max()),
            "constant_dimension_fraction":float(np.mean(spread<=1e-8)),
            "mean_vector_norm":float(np.linalg.norm(values,axis=1).mean())})
    return rows


def run(root,config_path,base,out):
    started=time.perf_counter()
    root,base,out=Path(root),Path(base),Path(out)
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg["mode"]!="frozen_validation_diagnostic_only" or cfg["official_test"]!="FORBIDDEN":
        raise ValueError("Unapproved experiment mode")
    ckpt=base/"checkpoints"/"scv_best.pt"
    old=base/"SCV-IDS-REPRO-001"
    required=[ckpt,old/"split_indices.npz",old/"validation_predictions.npz",
              old/"embedding_vectors.npz",old/"preprocessing.json"]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError("Original frozen checkpoint/NPZ files missing; restore draft Release.")
    if sha(ckpt)!=cfg["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match independently verified SHA-256")
    if sha(cfg["input"])!=cfg["input_sha256"]:
        raise ValueError("Designated official TRAIN CSV has changed")
    with np.load(old/"split_indices.npz",allow_pickle=False) as f:
        validation_indices=f["validation_indices"].astype(np.int64)
        validation_labels=f["validation_labels"].astype(np.int64)
        classes=[str(v) for v in f["classes"].tolist()]
    with np.load(old/"validation_predictions.npz",allow_pickle=False) as f:
        y=f["y_true"].astype(np.int64)
        ref_prob=f["probabilities"].astype(np.float64)
        ends=f["validation_window_end_indices"].astype(np.int64)
    with np.load(old/"embedding_vectors.npz",allow_pickle=False) as f:
        old_selected=f["sample_index"].astype(np.int64)
        old_cnn=f["cnn"].astype(np.float64)
    if len(classes)!=10 or len(y)!=len(validation_indices)-4:
        raise ValueError("Saved validation lineage has unexpected shape")
    if not np.array_equal(ends,validation_indices[4:]) or not np.array_equal(y,validation_labels[4:]):
        raise ValueError("Saved window targets do not match split lineage")
    rng=np.random.default_rng(cfg["seed"])
    random_add=rng.choice(len(y),size=min(cfg["additional_random_windows"],len(y)),
                          replace=False)
    selected=np.unique(np.concatenate([old_selected,random_add]))
    if len(selected)>cfg["max_total_windows"]:
        raise ValueError("Diagnostic sample limit exceeded")
    if (selected<0).any() or (selected>=len(y)).any():
        raise ValueError("Invalid frozen sample indices")
    if not (np.isfinite(ref_prob).all() and np.isfinite(old_cnn).all()):
        raise ValueError("Nonfinite original saved predictions or embedding vectors")
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"class":classes,"sample_support":[int((y[selected]==j).sum())
                for j in range(10)]}).to_csv(out/"sample_support.csv",index=False)

    # EXACT original fit parameters, not a new preprocessing fit.
    state=json.loads((old/"preprocessing.json").read_text(encoding="utf-8"))
    prep=FittedPreprocessor(**{k:state[k] for k in
               ("columns","categorical","vocabs","min_values","max_values")})
    original=pd.read_csv(cfg["input"],low_memory=False)
    if len(original)!=175341:raise ValueError("Train-source row count mismatch")
    validation_original=original.iloc[validation_indices]
    actual_labels=validation_original["attack_cat"].astype(str).str.strip().map(
        {name:i for i,name in enumerate(classes)}).to_numpy()
    if not np.array_equal(actual_labels,validation_labels):
        raise ValueError("Reconstructed validation labels differ from saved split")
    x_values,clipped=transform_preprocessor(prep,validation_original)
    del original,validation_original

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=SCVIDS(n_classes=10).to(device)
    weights=torch.load(ckpt,map_location=device,weights_only=False)
    model.load_state_dict(weights["model"],strict=True)
    model.eval()
    collector=CNNHooks(model)
    encodings={n:[] for n in ("cnn_pooled","cnn_tokens","vit_pooled",
                             "vit_to_cnn_attention","cnn_to_vit_attention","fused")}
    modes=cfg["ablations"]
    predictions={mode:[] for mode in modes}
    logits_delta={mode:[] for mode in modes if mode!="original"}
    replay_errors=[];replayed_cnn={}
    flips={"cnn":[],"vit":[]}
    rgb={"R":[],"G":[],"B":[]}

    try:
        with torch.no_grad():
            for pos in tqdm(range(0,len(selected),cfg["batch_size"]),
                       desc="Frozen CNN/attention probe",unit="batches"):
                ids=selected[pos:pos+cfg["batch_size"]]
                imgs=windows_to_rgb(x_values,ids+4,window=5)
                pixels=imgs.astype(np.float32)/255.
                for j,name in enumerate(("R","G","B")):
                    rgb[name].append(pixels[:,j].reshape(len(ids),-1))
                x=torch.from_numpy(imgs).to(device).float().div_(255)
                v=model.vit(x)
                c=model.cnn(x)
                z,vc,cv,fused=fusion_logits(model,v,c)
                prob=torch.softmax(z,dim=1).cpu().numpy()
                replay_errors.append(np.abs(prob-ref_prob[ids]))
                predictions["original"].append(prob.argmax(1))
                pooled=c.mean(1).cpu().numpy()
                for idx,value in zip(ids,pooled):replayed_cnn[int(idx)]=value
                values={"cnn_pooled":pooled,"cnn_tokens":c.flatten(1).cpu().numpy(),
                        "vit_pooled":v.mean(1).cpu().numpy(),
                        "vit_to_cnn_attention":vc.mean(1).cpu().numpy(),
                        "cnn_to_vit_attention":cv.mean(1).cpu().numpy(),
                        "fused":fused.cpu().numpy()}
                for name,item in values.items():encodings[name].append(item)
                for mode,vi,ci in (
                    ("cnn_zeroed",v,torch.zeros_like(c)),
                    ("vit_zeroed",torch.zeros_like(v),c),
                    ("both_zeroed",torch.zeros_like(v),torch.zeros_like(c))
                ):
                    result,_,_,_=fusion_logits(model,vi,ci)
                    predictions[mode].append(result.argmax(1).cpu().numpy())
                    logits_delta[mode].extend(torch.linalg.vector_norm(
                        result-z,dim=1).cpu().numpy().tolist())
                # Horizontal flip is a controlled input perturbation; not a network attack.
                collector.active=False
                changed=x.flip(-1)
                flips["cnn"].extend(torch.linalg.vector_norm(
                    model.cnn(changed).flatten(1)-c.flatten(1),dim=1).cpu().numpy().tolist())
                flips["vit"].extend(torch.linalg.vector_norm(
                    model.vit(changed).flatten(1)-v.flatten(1),dim=1).cpu().numpy().tolist())
                collector.active=True
    finally:
        collector.close()
    failures=np.concatenate(replay_errors)
    maxerr=float(failures.max())
    original_pred=np.concatenate(predictions["original"])
    ref_pred=ref_prob[selected].argmax(1)
    agreement=float(np.mean(original_pred==ref_pred))
    if maxerr>cfg["replay_max_absolute_probability_error"] or agreement<1.0:
        raise ValueError(f"Checkpoint replay mismatch: max |delta prob| {maxerr}; "
                         f"argmax agreement {agreement}. Do NOT interpret branch diagnostics.")
    prev_map={int(idx):old_cnn[j] for j,idx in enumerate(old_selected)}
    olddiff=np.stack([np.abs(replayed_cnn[int(idx)]-prev_map[int(idx)])
                      for idx in old_selected])
    if float(olddiff.max())>1e-4:
        raise ValueError("CNN embeddings do not reproduce original saved sample.")
    layers,channels=collector.summarize()
    representations=describe_embeddings(encodings)
    overall=[];per_class=[]
    label=y[selected]
    for mode in modes:
        p=np.concatenate(predictions[mode])
        o,detail=metric_rows(label,p,classes,mode)
        overall.append(o);per_class.extend(detail)
    baseline=predictions["original"]
    base_pred=np.concatenate(baseline)
    for m in overall:
        p=np.concatenate(predictions[m["mode"]])
        m["disagreement_with_original"]=float(np.mean(p!=base_pred))
        if m["mode"]=="original":m["mean_logit_l2_change"]=0.
        else:m["mean_logit_l2_change"]=float(np.mean(logits_delta[m["mode"]]))
    input_rows=[{
        "channel":c,"pixel_mean":float(np.concatenate(arr).mean()),
        "pixel_std":float(np.concatenate(arr).std()),
        "fraction_exact_zero":float(np.mean(np.concatenate(arr)==0))}
        for c,arr in rgb.items()]
    flip_rows=[{"branch":k,"mean_l2":float(np.mean(v)),
        "median_l2":float(np.median(v)),
        "p90_l2":float(np.percentile(v,90)),
        "fraction_near_zero":float(np.mean(np.asarray(v)<=1e-8))}
        for k,v in flips.items()]
    history_cnn=np.stack([prev_map[int(idx)] for idx in old_selected])
    prev_std=history_cnn.std(axis=0)
    report={
        "experiment_id":cfg["experiment_id"],
        "reference_experiment":"SCV-IDS-REPRO-001",
        "n_diagnostic_validation_windows":int(len(selected)),
        "reference_original_saved_cnn_windows":int(len(old_selected)),
        "reference_cnn_mean_feature_std":float(np.mean(prev_std)),
        "reference_cnn_zero_variance_dimensions":int(np.sum(prev_std<=1e-8)),
        "current_cnn_mean_feature_std":next(x["mean_feature_std"]
                                             for x in representations if x["branch"]=="cnn_pooled"),
        "prediction_replay_max_abs_probability_error":maxerr,
        "prediction_replay_argmax_agreement":agreement,
        "cnn_replay_max_abs_vector_error":float(olddiff.max()),
        "pass_prediction_replay":True,
        "checkpoint_sha256":sha(ckpt),
        "saved_predictions_sha256":sha(old/"validation_predictions.npz"),
        "saved_embeddings_sha256":sha(old/"embedding_vectors.npz"),
        "split_indices_sha256":sha(old/"split_indices.npz"),
        "training_source_sha256":cfg["input_sha256"],
        "original_training_environment_available":False,
        "diagnostic_code_git_sha":source_sha(root),
        "diagnostic_model_code_sha256":sha(root/"src"/"rareids"/"scv_model.py"),
        "sample_index_sha256":hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "sample_policy":"union of original saved stratified sample and fixed-seed random validation sample",
        "preprocessor":"replayed exactly from saved training-only fit",
        "preprocessing_clipped_feature_values":int(clipped),
        "gpu":torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU",
        "torch_version":str(torch.__version__),
        "python_version":platform.python_version(),
        "official_test_read":False,"training_or_optimizer_used":False,
        "elapsed_seconds":float(time.perf_counter()-started),
        "limitations":cfg["limitations"]
    }
    pd.DataFrame(layers).to_csv(out/"layer_statistics.csv",index=False)
    pd.DataFrame(channels).to_csv(out/"channel_statistics.csv",index=False)
    pd.DataFrame(representations).to_csv(out/"branch_variance.csv",index=False)
    pd.DataFrame(overall).to_csv(out/"ablation_overall.csv",index=False)
    pd.DataFrame(per_class).to_csv(out/"ablation_per_class.csv",index=False)
    pd.DataFrame(input_rows).to_csv(out/"rgb_channel_stats.csv",index=False)
    pd.DataFrame(flip_rows).to_csv(out/"flip_sensitivity.csv",index=False)
    pd.DataFrame(confusion_matrix(label,base_pred,labels=np.arange(10)),
             index=classes,columns=classes).rename_axis(
               "true_class").to_csv(out/"sampled_confusion.csv")
    write_json(out/"diagnostic_metrics.json",report)
    write_json(out/"config_effective.json",cfg)
    print("REPLAY VERIFIED; MAX PROBABILITY DIFFERENCE:",maxerr,
          "FROZEN CNN EMBEDDING DIFFERENCE:",float(olddiff.max()))
    print("Original CNN mean-feature standard deviation:",
          report["reference_cnn_mean_feature_std"])
    print("Diagnostic CNN mean-feature standard deviation:",
          report["current_cnn_mean_feature_std"])
    from plot_scv_cnn_diagnostic import make_figures
    generated=make_figures(out)
    if generated["figure_count"]!=cfg["output_figure_count"]:
        raise ValueError("Expected figure count differs from actual diagnostic outputs")
    report["figure_count"]=generated["figure_count"]
    report["figure_manifest_sha256"]=sha(out/"figure_manifest.json")
    write_json(out/"diagnostic_metrics.json",report)
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--config",required=True)
    ap.add_argument("--base",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    run(a.repo,a.config,a.base,a.output)
