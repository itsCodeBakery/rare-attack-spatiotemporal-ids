"""End-to-end *synthetic-only* smoke test for figure regeneration.

No synthetic figures or scores are published as benchmark findings.
All 28 plotting recipes must run offline using small saved CSVs.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.plot_scv_figures import Gallery

CLASSES=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
         "Normal","Reconnaissance","Shellcode","Worms"]

def test_all_figures_reproduce_from_numeric_csv_sources(tmp_path):
    source=tmp_path/"plot_data"
    source.mkdir()
    n=len(CLASSES)
    history=[]
    for epoch in (1,2,3):
        history.append({"epoch":epoch,"train_loss":1.1/epoch,
             "val_loss":1.2/epoch,"train_accuracy":.3+.1*epoch,
             "val_accuracy":.35+.05*epoch,
             "val_macro_f1":.2+.05*epoch,
             "val_weighted_f1":.3+.05*epoch,
             "learning_rate":.001,"epoch_seconds":1.0+epoch,
             "peak_gpu_memory_mb":200.0})
    pd.DataFrame(history).to_csv(source/"training_history.csv",index=False)
    counts=np.arange(5,5+n)
    pd.DataFrame({"class":CLASSES,
                  "train_count":counts*4,
                  "validation_count":counts}).to_csv(
                      source/"split_class_counts.csv",index=False)
    pd.DataFrame({"class":CLASSES,"support":counts,
                  "precision":np.linspace(.1,.9,n),
                  "recall":np.linspace(.1,.8,n),
                  "f1":np.linspace(.15,.85,n)}).to_csv(
                      source/"per_class_validation.csv",index=False)
    mat=np.eye(n,dtype=int)*counts
    pd.DataFrame(mat,index=CLASSES,columns=CLASSES).rename_axis(
        "true_class").to_csv(source/"confusion_validation.csv")
    pd.DataFrame({"mean_confidence":[.2,.5,.8],
                  "fraction_correct":[.1,.5,.9]}).to_csv(
                      source/"reliability_bins.csv",index=False)
    pd.DataFrame({"bin_center":[.15,.45,.85],
                  "correct":[1,10,12],
                  "incorrect":[5,4,1]}).to_csv(
                      source/"confidence_histogram.csv",index=False)
    curves=[]
    for name in CLASSES:
        for kind in ("roc","pr"):
            for j in range(10):
                curves.append({"class":name,"kind":kind,
                               "x":j/9,"y":j/9 if kind=="roc" else 1-j/18})
    pd.DataFrame(curves).to_csv(source/"roc_pr_curves.csv",index=False)
    pixels=[]
    for ci,name in enumerate(CLASSES):
        for pix in range(256):
            pixels.append({"class":name,"pixel":pix,
                           "r":ci*20,"g":pix%255,"b":128})
    pd.DataFrame(pixels).to_csv(source/"rgb_sample_pixels.csv",index=False)
    coords=[]
    for view in ("raw","cnn","vit","fused"):
        for i,name in enumerate(CLASSES):
            for j in range(5):
                coords.append({"representation":view,"class":name,
                               "sample_id":i*10+j,
                               "x":i+j*.1,"y":(i%3)+j*.2})
    pd.DataFrame(coords).to_csv(source/"embedding_coordinates.csv",index=False)
    manifest=Gallery(tmp_path).run()
    assert manifest["figure_count"]==28
    for item in manifest["figures"]:
        assert len(item["exports"])==3
        for entry in item["exports"]:
            assert (tmp_path/entry["path"]).is_file()
    checked=json.loads((tmp_path/"figure_manifest.json").read_text())
    assert checked["figure_count"]==28
