"""E6 preregistered training-only, source-order and nonoverlap controls tests."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

def raw_features_synthetic(n=20):
    x={}
    for lag in (4,3,2,1,0):
        x[f"lag{lag}__dur"]=np.arange(n,dtype=np.float32)+lag*100
        x[f"lag{lag}__rate"]=np.arange(n,dtype=np.float32)+lag*100+3
        x[f"lag{lag}__proto"]=pd.Categorical(
            ["tcp","udp","tcp","icmp"]*(n//4))
    return pd.DataFrame(x)

def test_whole_historical_block_shuffling_preserves_target_records_and_lag_relations():
    from scripts.run_e6_order_robustness import history_shuffle
    original=raw_features_synthetic()
    changed=history_shuffle(original,["dur","rate","proto"],6302)
    for feature in ("dur","rate","proto"):
        assert changed[f"lag0__{feature}"].equals(original[f"lag0__{feature}"])
    for lag in (1,2,3,4):
        np.testing.assert_array_equal(np.sort(changed[f"lag{lag}__dur"]),
                                      np.sort(original[f"lag{lag}__dur"]))
        np.testing.assert_allclose(changed[f"lag{lag}__rate"]-
                                   changed[f"lag{lag}__dur"],3)
        np.testing.assert_allclose(changed[f"lag{lag}__dur"]-
                                   changed["lag1__dur"],(lag-1)*100)
    assert not original["lag4__dur"].equals(changed["lag4__dur"])
    assert isinstance(changed["lag4__proto"].dtype,pd.CategoricalDtype)

def test_target_and_history_inputs_have_matching_labels_and_nonoverlap_indices():
    from scripts.run_e6_order_robustness import make_raw,metrics
    names=["dur","rate","proto"]
    frame=pd.DataFrame({"dur":np.arange(20,dtype=np.float32),
                        "rate":np.arange(20,dtype=np.float32)+7,
                        "proto":["tcp","udp","icmp","tcp"]*5})
    raw,vocab,cats=make_raw(frame,names)
    val,levels,check=make_raw(frame.iloc[:15],names,levels=vocab)
    assert raw.shape==(16,15) and val.shape==(11,15)
    assert raw["lag0__dur"].tolist()==list(range(4,20))
    assert raw["lag4__dur"].tolist()==list(range(0,16))
    assert raw["lag0__dur"].tolist()[::5]==[4.,9.,14.,19.]
    assert vocab==levels and cats==check
    assert cats==[f"lag{lag}__proto" for lag in (4,3,2,1,0)]
    y=np.arange(10)
    probs=np.eye(10,dtype=np.float32)
    score,cm,per,pred=metrics(y,probs)
    assert score["accuracy"]==1 and score["macro_f1"]==1
    assert cm.shape==(10,10) and len(per)==10
    np.testing.assert_array_equal(pred,y)

def test_config_fixes_five_alternative_controls_and_58_rounds():
    cfg=json.loads(Path("configs/e6_order_robustness_001.json").read_text())
    assert cfg["official_test"]=="FORBIDDEN"
    assert cfg["split_seed"]==2026 and cfg["model_seed"]==42
    assert cfg["fixed_num_boost_round"]==58
    assert cfg["no_hyperparameter_selection_from_e6_validation"] is True
    assert cfg["total_feature_count"]==444
    assert cfg["stats_feature_count"]==234
    assert [r["id"] for r in cfg["primary_conditions"]]==[
        "target_full","intact_full","shuffled_full","target_nonoverlap","intact_nonoverlap"]
    assert cfg["label_independent_historical_block_shuffle_seed_train"]!=(
        cfg["label_independent_historical_block_shuffle_seed_validation"])
    assert "macro_f1" in cfg["evaluation_metrics"]
    assert cfg["window_size"]==5

def test_no_official_test_access_and_private_archive_only():
    script=Path("scripts/run_e6_order_robustness.py").read_text()
    archival=Path("scripts/archive_e6_order.py").read_text()
    ast.parse(script)
    ast.parse(archival)
    assert "lgb.train(" in script
    assert 'num_boost_round=cfg["fixed_num_boost_round"]' in script
    assert "group_disjoint_stratified_split" in script
    assert "np.arange(0,len(yv),5" in script
    assert "history_shuffle(raw_x" in script
    assert "archive_candidate(output,name,token)" in script
    assert "restore_private_candidate(output,name,token)" in script
    assert "official_test_predictions_PRIVATE.npz" not in script
    assert "UNSW_NB15_testing-set.csv" not in script
    assert '"draft":True' in archival
    assert 'if not rel["draft"]' in archival
    assert 'if not file.is_file()' in archival or 'not file.is_file()' in archival
    assert "predictions_PRIVATE.npz" in archival
    notebook=json.loads(Path("notebooks/23_e6_order_robustness_five_controls_autobackup.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            code="".join(cell["source"])
            ast.parse(code)
            assert 'get_secret("pushRare")' in code
            assert "run_e6_order_robustness.py" in code
            assert "--auto-archive" in code
            assert "run_official_001.py" not in code
