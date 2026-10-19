"""E3 scientific contract: window statistics and shuffled negative control.

All tests use synthetic data; they never open UNSW official training/test files.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

def synthetic():
    frame={}
    for lag in (4,3,2,1,0):
        frame[f"lag{lag}__dur"]=np.array([lag+10,lag+20,lag+30,lag+40],dtype=np.float32)
        frame[f"lag{lag}__rate"]=np.array([lag*2+10,lag*2+20,lag*2+30,lag*2+40],dtype=np.float32)
        frame[f"lag{lag}__proto"]=pd.Categorical(["tcp","udp","icmp","tcp"])
    return pd.DataFrame(frame)

def test_window_statistics_match_manual_5row_values():
    from scripts.run_window_e3 import append_stats
    f=synthetic()
    out,cols=append_stats(f,["dur","rate"])
    assert out.shape==(4,27) and len(cols)==12
    np.testing.assert_allclose(out["stats__dur__five_mean"],[12,22,32,42])
    np.testing.assert_allclose(out["stats__dur__five_std"],np.sqrt(2))
    np.testing.assert_allclose(out["stats__dur__five_min"],[10,20,30,40])
    np.testing.assert_allclose(out["stats__dur__five_max"],[14,24,34,44])
    np.testing.assert_allclose(out["stats__dur__oldest_to_target_diff"],[-4,-4,-4,-4])
    np.testing.assert_allclose(out["stats__dur__previous_four_mean_to_target_diff"],[-2.5]*4)
    assert out["lag0__dur"].equals(f["lag0__dur"])
    assert list(out.columns[:len(f.columns)])==list(f.columns)

def test_history_control_preserves_target_and_record_coherence():
    from scripts.run_window_e3 import shuffled_history
    f=synthetic()
    control=shuffled_history(f,["dur","rate","proto"],1701)
    changed=False
    for lag in (4,3,2,1):
        a=control[f"lag{lag}__dur"].to_numpy()
        b=control[f"lag{lag}__rate"].to_numpy()
        # Joint permutation must preserve the within-record relation.
        np.testing.assert_allclose(b-a,lag*np.ones(len(a)))
        changed|=not np.array_equal(a,f[f"lag{lag}__dur"].to_numpy())
    assert changed
    for feature in ("dur","rate","proto"):
        assert control[f"lag0__{feature}"].equals(f[f"lag0__{feature}"])
    assert str(control["lag4__proto"].dtype)=="category"
    np.testing.assert_array_equal(
        sorted(control["lag3__rate"].to_numpy()),
        sorted(f["lag3__rate"].to_numpy()))

def test_e3_protocol_and_notebook_no_test_access():
    cfg=json.loads(Path("configs/window_e3_001_seed42.json").read_text())
    e2=json.loads(Path("configs/window_e2_001_seed42.json").read_text())
    assert cfg["official_test"]=="FORBIDDEN"
    for key in ("train_csv_sha256","reference_split_sha256","reference_pred_sha256"):
        assert cfg[key]==e2[key] and len(cfg[key])==64
    assert cfg["validation_target_rows"]==35005
    assert cfg["total_statistical_features"]==444
    assert cfg["derived_statistic_count"]==234
    assert len(cfg["candidates"])==3
    assert cfg["shuffling_seed_training"]!=cfg["shuffling_seed_validation"]
    for file in ("scripts/run_window_e3.py","scripts/archive_window_e3.py"):
        ast.parse(Path(file).read_text(encoding="utf-8"))
    runner=Path("scripts/run_window_e3.py").read_text(encoding="utf-8")
    assert "archive_and_sync(out,name" in runner
    assert "restore_prior_candidate(out,name" in runner
    assert "target_record_preserved_in_shuffling" in runner
    archiver=Path("scripts/archive_window_e3.py").read_text()
    assert 'if not release["draft"]' in archiver
    assert '"predictions_PRIVATE.npz"' in archiver
    notebook=json.loads(Path("notebooks/18_window_e3_stats_shuffled_history_autobackup.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            src="".join(cell["source"])
            ast.parse(src)
            assert "restore_prior_candidate(E2,model,token)" in src
            assert "--auto-archive" in src
