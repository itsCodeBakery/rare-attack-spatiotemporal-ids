"""Generate V2 fourteen matched validation figures plus four activation-health figures."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
# Support both import forms:
#  - pytest / package: scripts.plot_ctrl002_cnnvit
#  - Kaggle direct entry point: python scripts/finalize_ctrl002_cnnvit.py
if __package__:
    from scripts.plot_ctrl001_vit import plot_run as base_plot
else:
    from plot_ctrl001_vit import plot_run as base_plot

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()

def plot_run(folder):
    folder=Path(folder)
    health_file=folder/"plot_data"/"health_history.csv"
    if not health_file.exists():
        raise FileNotFoundError(health_file)
    history=pd.read_csv(health_file)
    checks=["epoch","cnn_token_nonzero_fraction",
            "cnn_token_between_sample_mean_std",
            "cnn_zero_mask_mean_logit_l2",
            "cnn_zero_mask_changed_prediction_fraction"]
    if len(history)!=50 or not set(checks).issubset(history.columns):
        raise ValueError("Expected the complete 50-epoch CNN-health numeric record")
    old=base_plot(folder)
    old["experiment_id"]="CTRL-CNNVIT-V2-S42"
    old["note"]="Original 14 figure recipes plus four observed activation-health figures."
    plt.rcParams.update({"pdf.fonttype":42,"svg.fonttype":"none",
                         "svg.hashsalt":"CTRL-CNNVIT-V2-S42"})
    spec=[
        ("15_cnn_token_nonzero","cnn_token_nonzero_fraction",
         "Nonzero CNN tokens by training epoch",0.05),
        ("16_cnn_token_variance","cnn_token_between_sample_mean_std",
         "CNN token variance on a fixed internal-validation sample",0.00001),
        ("17_cnn_zero_mask_logits","cnn_zero_mask_mean_logit_l2",
         "Effect of CNN token suppression on logits",None),
        ("18_cnn_zero_mask_predictions","cnn_zero_mask_changed_prediction_fraction",
         "Prediction disagreement under CNN suppression",None)
    ]
    for stem,field,title,threshold in spec:
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.plot(history["epoch"],history[field],marker=".",linewidth=1.8)
        if threshold is not None:
            ax.axhline(threshold,linestyle="--",label="Predeclared health threshold")
            ax.legend(frameon=False)
        ax.set(xlabel="Epoch",ylabel=field.replace("_"," "),title=title)
        ax.grid(axis="y",alpha=.15)
        fig.tight_layout()
        outs=[]
        for ext in ("png","svg","pdf"):
            target=folder/"figures"/(stem+"."+ext)
            meta=({"Creator":"CTRL-CNNVIT-V2-S42",
                   "CreationDate":datetime(2000,1,1,tzinfo=timezone.utc)}
                  if ext=="pdf" else {"Date":"2000-01-01"} if ext=="svg" else None)
            fig.savefig(target,dpi=240,bbox_inches="tight",metadata=meta)
            outs.append({"file":str(target.relative_to(folder)),
                         "sha256":sha(target),"bytes":target.stat().st_size})
        plt.close(fig)
        old["plots"].append({"name":stem,"caption":title,
                             "sources":["plot_data/health_history.csv"],
                             "exports":outs})
    # The shared base renderer already discovers all CSV files, including
    # health_history.csv. Do not duplicate its source manifest entry.
    health_entries=[x for x in old["plot_sources"]
                    if x["file"]=="plot_data/health_history.csv"]
    if len(health_entries)!=1 or health_entries[0]["sha256"]!=sha(health_file):
        raise ValueError("Health-plot source must appear once with the true SHA-256")
    old["figure_count"]=len(old["plots"])
    if old["figure_count"]!=18:raise AssertionError("Must produce 18 figures")
    (folder/"figure_manifest.json").write_text(
        json.dumps(old,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("V2 FIGURES:",old["figure_count"],"each in PNG/SVG/PDF")
    return old

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    args=p.parse_args()
    plot_run(args.run)
