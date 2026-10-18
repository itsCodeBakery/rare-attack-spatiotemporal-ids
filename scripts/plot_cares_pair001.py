"""Render CARES-PAIR-001 observed numeric evidence without training or inference.

Twelve plot families in true PNG/SVG/PDF, each with a numerically versioned
source CSV. Journal-style editable vectors, immutable saved confusion counts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEYS=("baseline","dos_only","backdoor_only","analysis_only",
      "all_unrestricted","all_selective")
LABELS=("Baseline","DoS only","Backdoor only","Analysis only",
        "All unrestricted","All selective")

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(4194304),b""): h.update(block)
    return h.hexdigest()

def plot(root):
    root=Path(root)
    figure_dir=root/"figures"
    figure_dir.mkdir(exist_ok=True)
    scores=pd.read_csv(root/"comparison_scores.csv")
    if list(scores["variant"])!=list(KEYS):
        raise ValueError("Expected six saved ablation variants in fixed original order")
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if summary["experiment_id"]!="CARES-PAIR-001-S42":
        raise ValueError("Experiment lineage changed")
    classes=json.loads((root/"data_lineage.json").read_text())["class_names"]
    for name in KEYS:
        if not (root/name/"per_class.csv").exists() or not (root/name/"confusion.csv").exists():
            raise FileNotFoundError("Actual saved classwise metrics missing: "+name)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
         "figure.dpi":130,"savefig.dpi":240,"pdf.fonttype":42,
         "svg.fonttype":"none","axes.spines.top":False,
         "axes.spines.right":False})
    items=[]
    def save(fig,slug,title,source):
        fig.tight_layout(pad=1.5)
        exports=[]
        for ext in ("png","svg","pdf"):
            path=figure_dir/(slug+"."+ext)
            fig.savefig(path,bbox_inches="tight")
            exports.append({"file":str(path.relative_to(root)),
                            "bytes":path.stat().st_size,
                            "sha256":sha(path)})
        plt.close(fig)
        items.append({"name":slug,"title":title,"source":source,"exports":exports})
    def bars(slug,title,field,label,scale=1):
        fig,ax=plt.subplots(figsize=(9,4.6))
        values=scores[field].to_numpy(dtype=float)*scale
        bars=ax.bar(range(len(values)),values,color="#36769a",alpha=.92)
        bars[0].set_color("#747e87")
        ax.set_xticks(range(len(values)),LABELS,rotation=20,ha="right")
        ax.set_ylabel(label);ax.set_title(title)
        ax.grid(axis="y",alpha=.18)
        for bar,value in zip(bars,values):
            ax.annotate(f"{value:.2f}" if label.endswith("(%)") else f"{int(value):,}",
                        (bar.get_x()+bar.get_width()/2,bar.get_height()),
                        ha="center",va="bottom",fontsize=8,xytext=(0,3),
                        textcoords="offset points")
        upper=float(np.max(values))
        ax.set_ylim(0,upper*1.17 if upper>0 else 1)
        save(fig,slug,title,"comparison_scores.csv")
    bars("01_macro_f1","Actual 10-class macro-F1","macro_f1","Macro-F1 (%)",100)
    bars("02_accuracy","Accuracy after each fixed routing intervention","accuracy","Accuracy (%)",100)
    bars("03_balanced_accuracy","Macro-average recall (balanced accuracy)","balanced_accuracy","Balanced accuracy (%)",100)
    bars("04_changed_predictions","Argmax labels changed versus frozen baseline",
         "changed_predictions_vs_baseline","Changed predictions",1)
    bars("05_dos_correct","DoS attacks correctly identified","DoS_correct","Correct DoS examples",1)
    bars("06_backdoor_correct","Backdoor attacks correctly identified",
         "Backdoor_correct","Correct Backdoor examples",1)
    bars("07_dos_to_exploits","DoS incorrectly predicted as Exploits",
         "DoS_to_Exploits","DoS→Exploits errors",1)
    bars("08_backdoor_to_exploits","Backdoor incorrectly predicted as Exploits",
         "Backdoor_to_Exploits","Backdoor→Exploits errors",1)
    per0=pd.read_csv(root/"baseline"/"per_class.csv")
    vectors=[]
    for key in KEYS:
        per=pd.read_csv(root/key/"per_class.csv")
        if per["class"].tolist()!=classes:raise ValueError("Class order drift: "+key)
        vectors.append(per["f1"].to_numpy(dtype=float))
    delta=(np.stack(vectors)[1:]-np.asarray(vectors[0]))*100
    fig,ax=plt.subplots(figsize=(10.8,5))
    lim=max(.1,float(np.max(np.abs(delta))))
    im=ax.imshow(delta,aspect="auto",cmap="RdBu",vmin=-lim,vmax=lim)
    ax.set_xticks(range(10),classes,rotation=38,ha="right")
    ax.set_yticks(range(5),LABELS[1:])
    ax.set_title("Observed per-class F1 change relative to rare-weighted LightGBM (pp)")
    fig.colorbar(im,ax=ax,label="F1 difference (percentage points)",pad=.02)
    save(fig,"09_classwise_f1_delta","Per-class F1 differences to frozen baseline",
         "each_variant/per_class.csv")
    for position,key in enumerate(("baseline","dos_only","all_selective"),start=10):
        confusion=pd.read_csv(root/key/"confusion.csv",index_col=0)
        if confusion.index.tolist()!=classes or confusion.columns.tolist()!=classes:
            raise ValueError("Confusion classes/order changed")
        raw=confusion.to_numpy(dtype=float)
        normalized=raw/np.maximum(raw.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(8.2,6.7))
        im=ax.imshow(normalized,interpolation="nearest",vmin=0,vmax=1,cmap="Blues")
        ax.set_xticks(range(10),classes,rotation=46,ha="right")
        ax.set_yticks(range(10),classes)
        ax.set_xlabel("Predicted category");ax.set_ylabel("Ground-truth category")
        ax.set_title("Confusion (row normalized): "+key.replace("_"," "))
        fig.colorbar(im,ax=ax,label="Fraction of actual class",fraction=.042,pad=.025)
        save(fig,f"{position:02d}_confusion_{key}",
             "Row-normalized confusion matrix: "+key,key+"/confusion.csv")
    out={"experiment_id":"CARES-PAIR-001-S42",
         "figure_count":len(items),"export_count":sum(len(v["exports"]) for v in items),
         "numeric_sources":[{"file":str(p.relative_to(root)),
             "bytes":p.stat().st_size,"sha256":sha(p)}
             for p in sorted([root/"comparison_scores.csv"]+
                 [root/key/name for key in KEYS for name in ("per_class.csv","confusion.csv")])],
         "figures":items,"scope":"35,005 training-derived internal-validation examples, six genuine saved variants"}
    if out["figure_count"]!=12 or out["export_count"]!=36:
        raise ValueError("Expected twelve real plots × PNG/SVG/PDF")
    (root/"figure_manifest.json").write_text(
        json.dumps(out,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("CARES FIGURES: 12 × PNG/SVG/PDF (36) generated from actual saved numeric sources")
    return out

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--run",required=True)
    plot(a.parse_args().run)
