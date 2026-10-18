"""Regenerate DIAG-001 figures from versioned aggregate CSVs, without model access.

Exports PNG (240dpi), editable SVG, PDF plus per-export hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C1="#236A97"; C2="#DD8243"; C3="#498A76"; C4="#8060AC"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1048576),b""):h.update(x)
    return h.hexdigest()

def make_figures(root):
    root=Path(root);folder=root/"figures";folder.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":120,
         "savefig.dpi":240,"pdf.fonttype":42,"svg.fonttype":"none",
         "svg.hashsalt":"SCV-CNN-DIAG-001","axes.spines.top":False,
         "axes.spines.right":False})
    results=[]
    def load(name):return pd.read_csv(root/name)
    def save(name,fig,inputs,caption):
        fig.tight_layout(pad=1.7)
        exports=[]
        for ext in ("png","svg","pdf"):
            file=folder/(name+"."+ext)
            meta=({"Creator":"SCV-CNN-DIAG-001",
                   "CreationDate":datetime(2000,1,1,tzinfo=timezone.utc)}
                  if ext=="pdf" else {"Date":"2000-01-01"} if ext=="svg" else None)
            fig.savefig(file,dpi=240,bbox_inches="tight",metadata=meta)
            exports.append({"path":str(file.relative_to(root)),
                            "sha256":sha(file),"bytes":file.stat().st_size})
        plt.close(fig)
        results.append({"stem":name,"source_files":inputs,
                        "caption":caption,"exports":exports})

    layer=load("layer_statistics.csv")
    channel=load("channel_statistics.csv")
    branch=load("branch_variance.csv")
    overall=load("ablation_overall.csv")
    detail=load("ablation_per_class.csv")
    rgb=load("rgb_channel_stats.csv")
    flip=load("flip_sensitivity.csv")
    confusion=load("sampled_confusion.csv")
    cls=detail[detail["mode"]=="original"]["class"].tolist()
    label=layer["layer"].astype(str).tolist()
    x=np.arange(len(label))

    def plot_stage(name,col,title,ylabel,log=False):
        fig,ax=plt.subplots(figsize=(10,4.7))
        vals=layer[col].to_numpy()
        ax.bar(x,vals,color=C1)
        if log:ax.set_yscale("symlog",linthresh=1e-9)
        ax.set_xticks(x,label)
        ax.set_ylabel(ylabel);ax.set_xlabel("CNN Sequential module index")
        ax.set_title(title)
        save(name,fig,["layer_statistics.csv"],title)
    plot_stage("01_layer_between_image_std","between_image_mean_std",
        "CNN stage variation between sampled validation images",
        "Mean between-image channel std",True)
    plot_stage("02_layer_zero_fraction","zero_fraction",
        "CNN layer fraction of exact zeros","Fraction zero")
    plot_stage("03_layer_constant_fraction","constant_fraction",
        "Fraction of constant CNN channels by layer","Fraction constant")
    plot_stage("04_layer_spatial_std","spatial_std",
        "CNN within-image spatial variation","Mean spatial std",True)

    fig,ax=plt.subplots(figsize=(10,4.5))
    xx=np.arange(len(branch))
    ax.bar(xx,branch["mean_feature_std"],color=C1,label="Mean per-feature std")
    ax.set_yscale("symlog",linthresh=1e-9)
    ax.set_xticks(xx,branch["branch"],rotation=35,ha="right")
    ax.set_ylabel("Std across validation examples")
    ax.set_title("Feature variation: CNN, ViT, attention and fusion")
    save("05_branch_variance",fig,["branch_variance.csv"],
         "Branch feature variance measured on frozen validation sample")

    fig,ax=plt.subplots(figsize=(8,4.5))
    xx=np.arange(len(overall))
    ax.bar(xx-.18,overall["macro_f1"],width=.35,color=C1,label="Macro-F1")
    ax.bar(xx+.18,overall["accuracy"],width=.35,color=C2,label="Accuracy")
    ax.set_xticks(xx,overall["mode"],rotation=20,ha="right")
    ax.set_ylim(0,1);ax.set_ylabel("Score")
    ax.set_title("Zeroed-branch sensitivity on a fixed validation subset")
    ax.legend(frameon=False)
    save("06_masked_branch_metrics",fig,["ablation_overall.csv"],
         "Inference-only masking, out-of-distribution, not causal training ablation")

    fig,ax=plt.subplots(figsize=(11,4.8))
    xx=np.arange(len(cls))
    modes=("original","cnn_zeroed","vit_zeroed")
    for k,(m,color) in enumerate(zip(modes,(C1,C2,C3))):
        values=detail[detail["mode"]==m].set_index("class").loc[cls,"recall"].to_numpy()
        ax.bar(xx+(k-1)*.25,values,width=.24,label=m,color=color)
    ax.set_xticks(xx,cls,rotation=40,ha="right")
    ax.set_ylim(0,1);ax.set_ylabel("Recall")
    ax.set_title("Per-class recall under frozen branch zeroing")
    ax.legend(frameon=False)
    save("07_class_recall_masking",fig,["ablation_per_class.csv"],
         "Validation subset recall; branch suppression is a diagnostic intervention")

    fig,ax=plt.subplots(figsize=(8,4.2))
    o=overall[overall["mode"]!="original"]
    xx=np.arange(len(o))
    ax.bar(xx-.19,o["mean_logit_l2_change"],width=.35,color=C1,label="Mean logits L2")
    ax.set_xticks(xx,o["mode"],rotation=25,ha="right")
    ax.set_ylabel("Mean logits L2 change")
    ax.set_title("Change in logits when a fusion input branch is zeroed")
    save("08_logit_masking",fig,["ablation_overall.csv"],"Paired sample logits change")

    fig,ax=plt.subplots(figsize=(6.5,4.4))
    xx=np.arange(len(rgb))
    ax.bar(xx-.18,rgb["pixel_mean"],width=.35,color=C1,label="Mean normalized pixel")
    ax.bar(xx+.18,rgb["pixel_std"],width=.35,color=C2,label="Pixel std")
    ax.set_xticks(xx,rgb["channel"])
    ax.set_ylabel("Normalized channel values")
    ax.set_title("Saved 24-bit serialized image channel variability")
    ax.legend(frameon=False)
    save("09_rgb_image_input",fig,["rgb_channel_stats.csv"],
         "Image-channel aggregates, not network-traffic raw records")

    fig,ax=plt.subplots(figsize=(6.5,4.4))
    xx=np.arange(len(flip))
    ax.bar(xx-.18,flip["median_l2"],width=.35,color=C1,label="Median")
    ax.bar(xx+.18,flip["p90_l2"],width=.35,color=C2,label="90th percentile")
    ax.set_xticks(xx,flip["branch"])
    ax.set_ylabel("Embedding L2 after horizontal image flip")
    ax.set_title("Frozen encoder response to controlled image perturbation")
    ax.legend(frameon=False)
    save("10_flip_sensitivity",fig,["flip_sensitivity.csv"],
         "Image-space perturbation control, NOT adversarial attack accuracy")

    fig,ax=plt.subplots(figsize=(9,4.5))
    names=channel["layer"].astype(str).unique().tolist()
    values=[channel[channel["layer"].astype(str)==l]["between_image_std"].values
            for l in names]
    ax.boxplot([np.maximum(v,1e-12) for v in values],tick_labels=names,
               showfliers=False)
    ax.set_yscale("log");ax.set_xlabel("CNN Sequential module index")
    ax.set_ylabel("Channel-wise between-image std (floor 1e-12)")
    ax.set_title("Channel-level variability across the CNN hierarchy")
    save("11_channel_variability",fig,["channel_statistics.csv"],
         "Channel activation variation; log-plot floor is display-only")

    cm=confusion[cls].to_numpy(dtype=float)
    if confusion["true_class"].tolist()!=cls:
        raise ValueError("Confusion class order differs from saved CSV")
    cm=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(8.4,7))
    im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(np.arange(len(cls)),cls,rotation=45,ha="right")
    ax.set_yticks(np.arange(len(cls)),cls)
    ax.set_ylabel("True class");ax.set_xlabel("Predicted class")
    ax.set_title("Frozen model confusion (diagnostic validation subset)")
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    save("12_sampled_confusion",fig,["sampled_confusion.csv"],
         "Counts row-normalized; diagnostic subset, NOT full test set")

    manifest={
        "experiment_id":"SCV-CNN-DIAG-001",
        "figure_count":len(results),
        "formats":["png","svg","pdf"],
        "plots":results,
        "sources":[{"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size}
            for p in sorted(root.glob("*.csv"))],
        "notes":"Figures generated solely from observed, frozen-checkpoint aggregate CSVs."}
    (root/"figure_manifest.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("GENERATED",len(results),"reproducible diagnostic figures × 3 formats")
    return manifest

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    args=ap.parse_args()
    make_figures(args.run)
