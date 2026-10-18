"""Publication figures rebuilt exclusively from versioned numeric plot sources.

Input: results/SCV-IDS-REPRO-001/plot_data/*.csv and run metadata.
Output: figures/{png,svg,pdf} and figure_manifest.json.
No metrics fabricated; plotting code never accesses the official test partition.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE=["#195A91","#27A29A","#DE7841","#715CB0","#D2A43B",
         "#4D7A70","#A33A66","#5D92C7","#BE7D55","#8C7298"]
RARE={"Analysis","Backdoor","Shellcode","Worms","DoS"}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

class Gallery:
    def __init__(self, root):
        self.root=Path(root)
        self.data=self.root/"plot_data"
        self.figdir=self.root/"figures"
        self.figdir.mkdir(parents=True,exist_ok=True)
        self.files=[]
        plt.rcParams.update({
            "font.family":"DejaVu Sans","font.size":10,
            "axes.titlesize":12,"axes.labelsize":10,
            "figure.dpi":120,"savefig.dpi":240,
            "axes.spines.top":False,"axes.spines.right":False,
            "axes.grid":False,"pdf.fonttype":42,"svg.fonttype":"none", "svg.hashsalt":"SCV-IDS-REPRO-001"
        })

    def csv(self,name):
        p=self.data/name
        if not p.is_file(): raise FileNotFoundError(str(p))
        return pd.read_csv(p)

    def store(self,name,fig,sources,caption):
        files=[]
        fig.tight_layout(pad=1.7)
        for ext in ("png","svg","pdf"):
            dest=self.figdir/(name+"."+ext)
            metadata=(
                {"Creator":"rare-attack-spatiotemporal-ids",
                 "CreationDate":datetime(2000,1,1,tzinfo=timezone.utc)}
                if ext=="pdf" else
                {"Date":"2000-01-01"} if ext=="svg" else None
            )
            fig.savefig(dest,dpi=240,bbox_inches="tight",metadata=metadata)
            files.append({"path":str(dest.relative_to(self.root)),"sha256":sha256(dest),"bytes":dest.stat().st_size})
        plt.close(fig)
        self.files.append({"stem":name,"caption":caption,
                           "source_files":sources,"exports":files})

    def run(self):
        h=self.csv("training_history.csv")
        supports=self.csv("split_class_counts.csv")
        cls=self.csv("per_class_validation.csv")
        conf=self.csv("confusion_validation.csv")
        bins=self.csv("reliability_bins.csv")
        hist=self.csv("confidence_histogram.csv")
        curves=self.csv("roc_pr_curves.csv")
        pixels=self.csv("rgb_sample_pixels.csv")
        coordinates=self.csv("embedding_coordinates.csv")
        classes=cls["class"].tolist()
        n=len(classes); colors=[PALETTE[i%len(PALETTE)] for i in range(n)]
        x=np.arange(n)

        # 01 / 02 — supports + imbalance
        fig,ax=plt.subplots(figsize=(11,4.4))
        ax.bar(x-0.2,supports["train_count"],width=.39,label="Development train",color="#195A91")
        ax.bar(x+0.2,supports["validation_count"],width=.39,label="Validation",color="#DE7841")
        ax.set_xticks(x,classes,rotation=40,ha="right")
        ax.set_ylabel("Window-labelled examples");ax.set_title("Partition class support")
        ax.legend(frameon=False);self.store("01_class_support",fig,["split_class_counts.csv"],"Train versus validation window-label counts")
        fig,ax=plt.subplots(figsize=(10,4.4))
        ordered=supports.sort_values("train_count",ascending=True)
        ax.barh(ordered["class"],ordered["train_count"],color="#195A91")
        ax.set_xscale("log");ax.set_xlabel("Train windows (log scale)")
        ax.set_title("Training class imbalance")
        self.store("02_class_imbalance_log",fig,["split_class_counts.csv"],"Logarithmic class-imbalance overview")

        # 03 - representative RGB traffic image per class, derived from train only
        fig,axes=plt.subplots(2,5,figsize=(11,5.3))
        for ax,name in zip(axes.flat,classes):
            r=pixels[pixels["class"]==name].sort_values("pixel")
            if len(r)!=256: raise ValueError("Expected 256 recorded pixels per RGB class")
            im=r[["r","g","b"]].to_numpy(dtype=np.uint8).reshape(16,16,3)
            ax.imshow(im,interpolation="nearest")
            ax.set_title(name,fontsize=10);ax.axis("off")
        self.store("03_rgb_examples",fig,["rgb_sample_pixels.csv"],"Actual generated train-split RGB images, one per class")
        fig,ax=plt.subplots(figsize=(10,4))
        for name,color in zip(classes,colors):
            r=pixels[pixels["class"]==name]
            luminance=(.2126*r["r"]+.7152*r["g"]+.0722*r["b"])/255
            counts,edges=np.histogram(luminance,bins=np.linspace(0,1,31))
            ax.plot((edges[1:]+edges[:-1])/2,counts/len(luminance),color=color,label=name,linewidth=1.3)
        ax.set_xlabel("RGB luminance");ax.set_ylabel("Fraction of 256 pixels")
        ax.set_title("Serialized example pixel-intensity distributions")
        ax.legend(ncol=5,fontsize=7,frameon=False)
        self.store("04_rgb_luminance",fig,["rgb_sample_pixels.csv"],"Intensity distributions of one example from each class")

        def timeseries(stem,cols,title,ylabel,colors_sel=None):
            fig,ax=plt.subplots(figsize=(8,4.3))
            for j,(c,label) in enumerate(cols):
                if c not in h: raise ValueError("History missing "+c)
                ax.plot(h["epoch"],h[c],label=label,linewidth=1.8,
                        color=(colors_sel or PALETTE)[j])
            ax.set_xlabel("Epoch");ax.set_ylabel(ylabel);ax.set_title(title)
            if len(cols)>1: ax.legend(frameon=False)
            ax.grid(axis="y",alpha=.15)
            self.store(stem,fig,["training_history.csv"],title)

        timeseries("05_train_validation_loss",[("train_loss","Train"),("val_loss","Validation")],
                   "Focal loss over training","Focal loss")
        timeseries("06_training_accuracy",[("train_accuracy","Train accuracy")],
                   "Training accuracy by epoch","Accuracy")
        timeseries("07_validation_accuracy",[("val_accuracy","Validation accuracy")],
                   "Internal-validation accuracy","Accuracy")
        timeseries("08_validation_macrof1",[("val_macro_f1","Validation macro-F1")],
                   "Internal-validation macro-F1","Macro-F1")
        timeseries("09_validation_weightedf1",[("val_weighted_f1","Weighted F1"),
                                                 ("val_macro_f1","Macro F1")],
                   "Weighted versus macro-F1","F1")
        timeseries("10_learning_rate",[("learning_rate","Learning rate")],
                   "Recorded optimizer learning rate","Learning rate")
        timeseries("11_epoch_seconds",[("epoch_seconds","Epoch duration")],
                   "Training time per epoch","Seconds")
        timeseries("12_gpu_peak_memory",[("peak_gpu_memory_mb","Peak allocated GPU memory")],
                   "PyTorch allocated GPU peak per epoch","MiB")
        timeseries("13_train_validation_accuracy",[("train_accuracy","Train"),("val_accuracy","Validation")],
                   "Training versus validation accuracy","Accuracy")

        # 14 / 15 confusion matrix
        matrix=conf[classes].to_numpy(dtype=np.int64)
        if matrix.shape!=(n,n) or conf["true_class"].tolist()!=classes:
            raise ValueError("Confusion label order mismatch")
        for normalized in (False,True):
            numbers=matrix/(matrix.sum(axis=1,keepdims=True).clip(min=1)) if normalized else matrix
            fig,ax=plt.subplots(figsize=(8.4,7.4))
            im=ax.imshow(numbers,cmap="Blues",aspect="auto")
            ax.set_xticks(x,classes,rotation=50,ha="right")
            ax.set_yticks(x,classes)
            ax.set_xlabel("Predicted class");ax.set_ylabel("True class")
            ax.set_title("Internal validation confusion matrix — "+("row-normalized" if normalized else "counts"))
            fmt=".2f" if normalized else "d"
            for i in range(n):
                for j in range(n):
                    ax.text(j,i,format(numbers[i,j],fmt),fontsize=6.2,
                            ha="center",va="center",
                            color="white" if numbers[i,j]>numbers.max()*.6 else "black")
            fig.colorbar(im,ax=ax,fraction=.038,pad=.02)
            self.store("15_confusion_normalized" if normalized else "14_confusion_counts",fig,
                       ["confusion_validation.csv"],"Validation confusion matrix; explicitly labelled metric")

        # 16, 17, 18, 19 class breakdowns
        fig,ax=plt.subplots(figsize=(11,4.3))
        for offset,field,col in zip((-.25,0,.25),("precision","recall","f1"),
                                    ("#195A91","#27A29A","#DE7841")):
            ax.bar(x+offset,cls[field],width=.23,label=field.title(),color=col)
        ax.set_xticks(x,classes,rotation=45,ha="right");ax.set_ylim(0,1.05)
        ax.set_ylabel("Score");ax.set_title("Precision, recall and F1 by class")
        ax.legend(frameon=False);self.store("16_per_class_prf",fig,["per_class_validation.csv"],"Class-wise validation precision, recall and F1")
        fig,ax=plt.subplots(figsize=(10,4.4))
        order=cls.sort_values("f1")
        ax.barh(order["class"],order["f1"],color=[("#DE7841" if c in RARE else "#195A91") for c in order["class"]])
        ax.set_xlim(0,1);ax.set_xlabel("F1");ax.set_title("Per-class F1: weakest categories highlighted")
        self.store("17_class_f1_ranked",fig,["per_class_validation.csv"],"Ranked class-wise validation F1")
        fig,ax=plt.subplots(figsize=(9,4.4))
        rare=cls[cls["class"].isin(RARE)].sort_values("recall")
        ax.barh(rare["class"],rare["recall"],color="#DE7841")
        ax.set_xlim(0,1);ax.set_xlabel("Recall")
        ax.set_title("Low-support and high-confusion attack recall")
        self.store("18_rare_class_recall",fig,["per_class_validation.csv"],"Selected difficult-attack recalls; selection specified a priori")
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.scatter(cls["support"].clip(lower=1),cls["f1"],s=60,color="#195A91")
        for _,r in cls.iterrows(): ax.annotate(r["class"],(r["support"],r["f1"]),
                                            xytext=(4,3),textcoords="offset points",fontsize=8)
        ax.set_xscale("log");ax.set_ylim(0,1.04)
        ax.set_xlabel("Validation support (log)");ax.set_ylabel("F1")
        ax.set_title("F1 versus class support")
        self.store("19_support_vs_f1",fig,["per_class_validation.csv"],"Score relative to sample support")
        fig,ax=plt.subplots(figsize=(10,4.4))
        true=matrix.sum(axis=1); pred=matrix.sum(axis=0)
        ax.bar(x-.2,true/true.sum(),width=.39,color="#195A91",label="Ground truth")
        ax.bar(x+.2,pred/pred.sum(),width=.39,color="#DE7841",label="Predictions")
        ax.set_xticks(x,classes,rotation=45,ha="right")
        ax.set_ylabel("Fraction of window labels");ax.set_title("True and predicted class proportions")
        ax.legend(frameon=False)
        self.store("20_class_prediction_shift",fig,["confusion_validation.csv"],"Predicted class imbalance relative to ground truth")

        # 21 22 reliability and confidence
        fig,ax=plt.subplots(figsize=(6.5,5))
        ax.plot([0,1],[0,1],ls="--",color="gray",label="Perfect")
        ax.plot(bins["mean_confidence"],bins["fraction_correct"],"o-",
                color="#195A91",label="Observed bins")
        ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_xlabel("Mean maximum class probability");ax.set_ylabel("Fraction correct")
        ax.set_title("Validation reliability diagram")
        ax.legend(frameon=False)
        self.store("21_reliability",fig,["reliability_bins.csv"],"Maximum-probability calibration bins on validation")
        fig,ax=plt.subplots(figsize=(8,4))
        ax.bar(hist["bin_center"],hist["correct"],width=.047,color="#195A91",alpha=.8,label="Correct")
        ax.bar(hist["bin_center"],hist["incorrect"],bottom=hist["correct"],width=.047,
               color="#DE7841",alpha=.9,label="Incorrect")
        ax.set_xlabel("Maximum predicted class probability");ax.set_ylabel("Validation predictions")
        ax.set_title("Confidence distribution by prediction correctness")
        ax.legend(frameon=False)
        self.store("22_confidence_counts",fig,["confidence_histogram.csv"],"Correct vs incorrect confidence histogram")

        # 23 24 one-versus-rest discriminative curves
        for kind,name in (("roc","ROC"),("pr","Precision–Recall")):
            fig,ax=plt.subplots(figsize=(7.3,6))
            for label,color in zip(classes,colors):
                r=curves[(curves["class"]==label)&(curves["kind"]==kind)]
                if r.empty: continue
                ax.plot(r["x"],r["y"],color=color,linewidth=1.4,label=label)
            if kind=="roc":ax.plot([0,1],[0,1],ls="--",color="gray",linewidth=1)
            ax.set_xlim(0,1);ax.set_ylim(0,1)
            ax.set_xlabel("False-positive rate" if kind=="roc" else "Recall")
            ax.set_ylabel("True-positive rate" if kind=="roc" else "Precision")
            ax.set_title("Internal validation one-vs-rest "+name+" curves")
            ax.legend(fontsize=7,frameon=False,loc="lower right" if kind=="roc" else "lower left")
            self.store("23_roc_ovr" if kind=="roc" else "24_pr_ovr",fig,
                       ["roc_pr_curves.csv"],"Validation one-vs-rest "+name+"; plotted points downsampled for SVG clarity")

        # 25-28 frozen PCA projections; no embedding optimizer or retraining.
        for i,view in enumerate(("raw","cnn","vit","fused"),start=25):
            subset=coordinates[coordinates["representation"]==view]
            if subset.empty: raise ValueError("Missing frozen embedding projection: "+view)
            if "projection" in subset and not subset["projection"].eq("PCA").all():
                raise ValueError("Projection labels are inconsistent for "+view)
            status=(subset["projection_status"].iloc[0]
                    if "projection_status" in subset else "ok")
            fig,ax=plt.subplots(figsize=(8,6.6))
            if status=="zero_variance":
                ax.text(.5,.5,"No informative feature variance\\n"
                             "All sampled embeddings are identical.",
                             transform=ax.transAxes,ha="center",va="center",fontsize=11)
                ax.set_xlim(0,1);ax.set_ylim(0,1)
            else:
                for name,color in zip(classes,colors):
                    cls_subset=subset[subset["class"]==name]
                    ax.scatter(cls_subset["x"],cls_subset["y"],s=12,alpha=.75,
                               color=color,label=name,linewidths=0)
                ax.legend(fontsize=7,frameon=False,ncol=2)
            ax.set_xlabel("PCA component 1");ax.set_ylabel("PCA component 2")
            ax.set_title("Stratified validation PCA — "+view.upper())
            self.store(f"{i:02d}_pca_{view}",fig,["embedding_coordinates.csv"],
                       "Exploratory PCA from frozen validation embeddings; "+status)

        manifest={"status":"generated_from_observed_sources",
                  "figure_count":len(self.files),
                  "source_data":[{"file":"plot_data/"+p.name,
                    "sha256":sha256(p),"bytes":p.stat().st_size}
                    for p in sorted(self.data.glob("*.csv"))],
                  "figures":self.files,"formats":["png","svg","pdf"],
                  "note":"Figures generated from saved data; SCV-IDS author-implementation equivalence not claimed. No official-test predictions."}
        (self.root/"figure_manifest.json").write_text(
            json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        # If figures are rebuilt later without model retraining, refresh provenance.
        evidence_path=self.root/"artifact_manifest.json"
        if evidence_path.is_file():
            evidence=json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["figure_manifest_sha256"]=sha256(self.root/"figure_manifest.json")
            evidence["figure_count"]=len(self.files)
            evidence_path.write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n",
                                     encoding="utf-8")
        print(f"Generated {len(self.files)} scientifically defined figures in PNG/SVG/PDF")
        return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    args=p.parse_args()
    Gallery(args.run).run()
