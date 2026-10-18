"""Fast tabular baseline: numerical correctness, no-label-leakage contracts."""
import ast
import json
from pathlib import Path
import numpy as np
from rareids.tabular_eval import score_predictions,class_weights

CLASSES=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Normal",
         "Reconnaissance","Shellcode","Worms"]

def test_confusion_exact_f1_and_highest_class_weights():
    y=np.repeat(np.arange(10),np.arange(1,11))
    p=np.full((len(y),10),1e-8)
    p[np.arange(len(y)),y]=.9999
    info,cm,pred=score_predictions(y,p,CLASSES)
    assert info["count"]==55 and info["correct"]==55
    assert info["macro_f1"]==1.0 and info["accuracy"]==1.0
    assert info["DoS_to_Exploits"]==0
    assert np.trace(cm)==55
    w=class_weights(y,power=0.5,max_weight=12)
    assert len(w)==10 and w[0]>w[9] and w.min()>=1 and w.max()<=12

def test_pre_registered_same_targets_and_excluded_label_columns():
    cfg=json.loads(Path("configs/ctrl003_fast_tabular.json").read_text())
    assert cfg["train_target_rows"]==140328
    assert cfg["validation_target_rows"]==35005
    assert cfg["official_test"]=="FORBIDDEN"
    assert cfg["features"]==42
    assert set(cfg["categorical_columns"])=={"proto","service","state"}
    assert len(cfg["candidates"])==2
    script=Path("scripts/run_ctrl003_tabular_fast.py").read_text()
    assert "idx_va[4:]" in script and "idx_tr[4:]" in script
    assert 'set(cols)&{"id","label","attack_cat"}' in script
    assert "split_indices.npz" in script
    assert "validation_predictions.npz" in script
    ast.parse(script)
