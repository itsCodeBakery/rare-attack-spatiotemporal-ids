"""CTRL-VIT-001: rebuild fourteen figure groups ONLY from saved numeric CSVs.

All 14 figures exported PNG/SVG/PDF with hashes; no model inference.
"""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE="#245E86";TEAL="#248C86";ORANGE="#CD784C"

def checksum(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def plot_run(root):
    root=Path(root)
    source=root/"plot_data"
    folder=root/"figures";folder.mkdir(exist_ok=True,parents=True)
    def csv(name):
        f=source/name
        if not f.is_file():raise FileNotFoundError(f)
        return pd.read_csv(f)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
       "figure.dpi":120,"savefig.dpi":240,"pdf.fonttype":42,"svg.fonttype":"none",
       "svg.hashsalt":"CTRL-VIT-001-S42","axes.spines.top":False,
       "axes.spines.right":False})
    entries=[]
    def save(name,fig,inputs,caption):
        fig.tight_layout(pad=1.4)
        exports=[]
        for ext in ("png","svg","pdf"):
            dest=folder/(name+"."+ext)
            metadata=({"Creator":"CTRL-VIT-001-S42",
              "CreationDate":datetime(2000,1,1,tzinfo=timezone.utc)}
              if ext=="pdf" else {"Date":"2000-01-01"} if ext=="svg" else None)
            fig.savefig(dest,bbox_inches="tight",metadata=metadata)
            exports.append({"file":str(dest.relative_to(root)),
                            "bytes":dest.stat().st_size,"sha256":checksum(dest)})
        plt.close(fig)
        entries.append({"name":name,"caption":caption,
                        "sources":["plot_data/"+s for s in inputs],"exports":exports})
    hist=csv("training_history.csv")
    pr=csv("per_class.csv")
    cm=csv("confusion_matrix.csv")
    counts=csv("split_support.csv")
    reliability=csv("reliability.csv")
    curves=csv("curves.csv")
    freq=csv("prediction_proportions.csv")
    classes=pr["class"].tolist()
    mat=cm[classes].to_numpy()
    if classes!=cm["true_class"].tolist():raise ValueError("Confusion label ordering mismatch")

    def line(stem,keys,title,ylabel):
        fig,ax=plt.subplots(figsize=(8,4.2))
        for j,(column,name) in enumerate(keys):
            ax.plot(hist["epoch"],hist[column],
                    lw=1.8,color=[BLUE,ORANGE,TEAL][j%3],label=name)
        ax.set_xlabel("Epoch");ax.set_ylabel(ylabel);ax.set_title(title)
        ax.grid(axis="y",alpha=.2)
        if len(keys)>1:ax.legend(frameon=False)
        save(stem,fig,["training_history.csv"],title)
    line("01_train_validation_focal_loss",
          [("train_loss","Train"),("val_loss","Validation")],
         "Focal loss across training","Focal loss")
    line("02_train_validation_accuracy",
         [("train_accuracy","Train"),("val_accuracy","Validation")],
         "Accuracy over 50 epochs","Accuracy")
    line("03_macro_f1",[("val_macro_f1","Validation macro-F1")],
         "Internal-validation macro-F1","Macro-F1")
    line("04_weighted_vs_macro_f1",
         [("val_weighted_f1","Weighted-F1"),("val_macro_f1","Macro-F1")],
         "Weighted versus macro-F1","F1")
    line("05_epoch_runtime",[("epoch_seconds","Epoch duration")],
         "Training duration per epoch","Seconds")

    fig,ax=plt.subplots(figsize=(10,4.4))
    x=np.arange(len(classes))
    ax.bar(x-.2,counts["train_windows"],width=.38,color=BLUE,label="Train windows")
    ax.bar(x+.2,counts["validation_windows"],width=.38,color=ORANGE,
           label="Validation windows")
    ax.set_xticks(x,classes,rotation=42,ha="right")
    ax.set_ylabel("Samples");ax.set_title("Same target-window support as original SCV-IDS")
    ax.legend(frameon=False)
    save("06_partition_support",fig,["split_support.csv"],
         "Train and validation class support, original partition/window labels")

    fig,ax=plt.subplots(figsize=(9,4.5))
    ordered=pr.sort_values("f1")
    ax.barh(ordered["class"],ordered["f1"],
            color=[ORANGE if v in {"Analysis","Backdoor","DoS","Worms"} else BLUE
                   for v in ordered["class"]])
    ax.set_xlim(0,1);ax.set_xlabel("F1")
    ax.set_title("Validation per-class F1")
    save("07_per_class_f1",fig,["per_class.csv"],"Best-checkpoint class-wise F1")

    fig,ax=plt.subplots(figsize=(11,4.4))
    ax.bar(x-.2,pr["precision"],width=.38,color=BLUE,label="Precision")
    ax.bar(x+.2,pr["recall"],width=.38,color=ORANGE,label="Recall")
    ax.set_xticks(x,classes,rotation=40,ha="right");ax.set_ylim(0,1)
    ax.set_title("Precision and recall by class");ax.legend(frameon=False)
    save("08_class_precision_recall",fig,["per_class.csv"],
         "Precision and recall calculated from frozen best-checkpoint predictions")

    for normalized in (False,True):
        plot=(mat/np.maximum(mat.sum(axis=1,keepdims=True),1)) if normalized else mat
        fig,ax=plt.subplots(figsize=(8.5,7.4))
        im=ax.imshow(plot,cmap="Blues",aspect="auto")
        ax.set_xticks(x,classes,rotation=45,ha="right");ax.set_yticks(x,classes)
        ax.set_xlabel("Predicted class");ax.set_ylabel("True class")
        ax.set_title("Validation confusion: "+("row-normalized" if normalized else "counts"))
        for i in range(len(classes)):
            for j in range(len(classes)):
                v=plot[i,j]
                ax.text(j,i,f"{v:.2f}" if normalized else str(int(v)),
                        color="white" if v>plot.max()*.6 else "black",
                        fontsize=6.3,ha="center",va="center")
        fig.colorbar(im,ax=ax,fraction=.036,pad=.03)
        save("10_confusion_normalized" if normalized else "09_confusion_counts",
             fig,["confusion_matrix.csv"],"Best-checkpoint confusion matrix")

    fig,ax=plt.subplots(figsize=(6.5,5))
    ax.plot([0,1],[0,1],"--",color="grey",label="Perfect calibration")
    ax.plot(reliability["mean_confidence"],reliability["observed_accuracy"],
            "o-",color=BLUE,label="Observed validation bins")
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel("Mean maximum probability");ax.set_ylabel("Accuracy")
    ax.set_title("Validation reliability diagram");ax.legend(frameon=False)
    save("11_reliability",fig,["reliability.csv"],
         "Calibration diagnostic using 15 equal-width confidence bins")

    for kind,tag,title in (("roc","12_roc_ovr","ROC"),
                           ("pr","13_precision_recall_ovr","Precision–recall")):
        fig,ax=plt.subplots(figsize=(7,6))
        for i,name in enumerate(classes):
            part=curves[(curves["kind"]==kind)&(curves["class"]==name)]
            ax.plot(part["x"],part["y"],label=name,lw=1.3)
        if kind=="roc":ax.plot([0,1],[0,1],"--",color="gray")
        ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_xlabel("False positive rate" if kind=="roc" else "Recall")
        ax.set_ylabel("True positive rate" if kind=="roc" else "Precision")
        ax.set_title("One-vs-rest "+title+" (validation)")
        ax.legend(frameon=False,fontsize=7,ncol=2)
        save(tag,fig,["curves.csv"],"Full probabilities used for scores; curve points downsampled for display")

    fig,ax=plt.subplots(figsize=(10,4.3))
    ax.bar(x-.2,freq["actual_proportion"],width=.38,label="True",color=BLUE)
    ax.bar(x+.2,freq["predicted_proportion"],width=.38,label="Predicted",color=ORANGE)
    ax.set_xticks(x,classes,rotation=42,ha="right")
    ax.set_ylabel("Fraction of validation windows")
    ax.set_title("Actual vs predicted class frequencies");ax.legend(frameon=False)
    save("14_class_frequency_shift",fig,["prediction_proportions.csv"],
         "Prediction distribution shift, best checkpoint")

    result={"experiment_id":"CTRL-VIT-001-S42","figure_count":len(entries),
            "plots":entries,"formats":["png","svg","pdf"],
            "plot_sources":[{"file":"plot_data/"+p.name,"sha256":checksum(p),
                             "bytes":p.stat().st_size}
                            for p in sorted(source.glob("*.csv"))],
            "note":"These are actual values only after Kaggle inference; data-independent CI tests use synthetic fixtures."}
    (root/"figure_manifest.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("REPRODUCIBLE FIGURES:",len(entries),"x 3 formats")
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    plot_run(args.run)
