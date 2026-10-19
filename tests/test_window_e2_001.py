"""WINDOW-E2-001: real split-local five-record feature contract and no test."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd

def test_exact_oldest_to_target_order_and_partition_edges():
    from scripts.run_window_e2 import build_split_windows
    cols=["rate","proto"]
    cats=["proto"]
    train=pd.DataFrame({"rate":[10,20,30,40,50,60],
                        "proto":["tcp","udp","tcp","icmp","tcp","udp"]})
    val=pd.DataFrame({"rate":[100,200,300,400,500],
                      "proto":["icmp","icmp","udp","tcp","tcp"]})
    X=build_split_windows(train,cols,cats,5)
    V=build_split_windows(val,cols,cats,5)
    assert X.shape==(2,10) and V.shape==(1,10)
    assert list(X.columns)==[
        "lag4__rate","lag4__proto","lag3__rate","lag3__proto",
        "lag2__rate","lag2__proto","lag1__rate","lag1__proto",
        "lag0__rate","lag0__proto"]
    assert list(X["lag4__rate"])==[10.,20.]
    assert list(X["lag0__rate"])==[50.,60.]
    assert V.iloc[0]["lag4__rate"]==100.
    assert V.iloc[0]["lag0__rate"]==500.
    assert not (V.to_numpy()=="10").any()
    assert X["lag3__proto"].tolist()==["udp","tcp"]

def test_frozen_protocol_and_archive_safeguards():
    cfg=json.loads(Path("configs/window_e2_001_seed42.json").read_text())
    ref=json.loads(Path("configs/ctrl003_fast_tabular.json").read_text())
    assert cfg["official_test"]=="FORBIDDEN"
    assert cfg["expected_flat_features"]==5*ref["features"]==210
    assert cfg["train_target_rows"]==140328
    assert cfg["validation_target_rows"]==35005
    for k in ("train_csv_sha256","reference_split_sha256","reference_pred_sha256"):
        assert len(cfg[k])==64 and cfg[k]==ref[k]
    for p in ["scripts/run_window_e2.py","scripts/archive_window_e2.py"]:
        ast.parse(Path(p).read_text())
    script=Path("scripts/run_window_e2.py").read_text()
    assert "archive_and_sync(output,key" in script
    assert "restore_prior_candidate(output,key" in script
    assert "pd.read_csv(raw,usecols=list(cols)+[\"attack_cat\"]" in script
    private=Path("scripts/archive_window_e2.py").read_text()
    assert '"draft":True' in private
    assert 'if not release["draft"]' in private
    assert '"predictions_PRIVATE.npz"' in private
    nb=json.loads(Path("notebooks/17_window_e2_five_record_lightgbm_autobackup.ipynb").read_text())
    cell="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    ast.parse(cell)
    assert "--auto-archive" in cell
    assert 'get_secret("pushRare")' in cell
