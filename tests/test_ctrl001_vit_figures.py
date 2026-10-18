"""Offline CTRL-001 figures and matched-target protocol smoke tests (synthetic ONLY)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.plot_ctrl001_vit import plot_run

CLASSES=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
         "Normal","Reconnaissance","Shellcode","Worms"]

def test_rebuild_fourteen_figures_without_model_or_predictions(tmp_path):
    data=tmp_path/"plot_data";data.mkdir()
    hist=pd.DataFrame([
       {"epoch":i,"train_loss":.2/i,"val_loss":.3/i,
        "train_accuracy":.8,"val_accuracy":.75,
        "val_macro_f1":.5,"val_weighted_f1":.7,
        "epoch_seconds":2.+i/20} for i in range(1,4)])
    hist.to_csv(data/"training_history.csv",index=False)
    pd.DataFrame({"class":CLASSES,
        "support":[10+i for i in range(10)],
        "precision":[.5]*10,"recall":[.4]*10,"f1":[.44]*10
    }).to_csv(data/"per_class.csv",index=False)
    m=np.eye(10,dtype=int)*10
    pd.DataFrame(m,index=CLASSES,columns=CLASSES).rename_axis(
        "true_class").to_csv(data/"confusion_matrix.csv")
    pd.DataFrame({"class":CLASSES,"train_windows":[40]*10,
        "validation_windows":[10]*10}).to_csv(data/"split_support.csv",index=False)
    pd.DataFrame({"class":CLASSES,"actual_proportion":[.1]*10,
        "predicted_proportion":[.1]*10}).to_csv(data/"prediction_proportions.csv",index=False)
    pd.DataFrame({"bin":[1,2,3],"n":[10,10,10],
        "mean_confidence":[.2,.5,.8],
        "observed_accuracy":[.1,.4,.9]}).to_csv(data/"reliability.csv",index=False)
    rows=[]
    for name in CLASSES:
        for kind in ("roc","pr"):
            for t in range(10):
                rows.append({"class":name,"kind":kind,
                    "x":t/9,"y":t/9 if kind=="roc" else 1-t/18})
    pd.DataFrame(rows).to_csv(data/"curves.csv",index=False)
    manifest=plot_run(tmp_path)
    assert manifest["figure_count"]==14
    assert len(manifest["plots"])==14
    assert len(manifest["plot_sources"])==7
    for ext in ("png","svg","pdf"):
        assert len(list((tmp_path/"figures").glob("*."+ext)))==14
    loaded=json.loads((tmp_path/"figure_manifest.json").read_text())
    assert all(len(entry["exports"])==3 for entry in loaded["plots"])
    sources={entry["file"] for entry in loaded["plot_sources"]}
    assert all(set(item["sources"]).issubset(sources) for item in loaded["plots"])


def test_no_official_test_references_in_control_training_code():
    code=Path("scripts/train_ctrl001_vit.py").read_text()
    assert 'cfg.get("official_test")!="FORBIDDEN"' in code
    assert "reference_split_sha256" in code
    assert "reference_predictions_sha256" in code
    assert "window_target_equal_to_original" in code
    assert "validation_predictions.npz" in code
    assert "do_not_read" not in code.lower()
    cfg=json.loads(Path("configs/ctrl001_vit_only_seed42.json").read_text())
    assert cfg["official_test"]=="FORBIDDEN"
    assert cfg["reference_validation_windows"]==35005
    assert cfg["seed"]==42
    assert cfg["epochs"]==50
    assert cfg["figures_expected"]==14


def test_ctrl_finalization_no_model_inference():
    text=Path("scripts/finalize_ctrl001_vit.py").read_text()
    assert "torch.load" not in text
    assert ".backward(" not in text
    assert "optimizer.step(" not in text
    assert "plot_run(root)" in text
