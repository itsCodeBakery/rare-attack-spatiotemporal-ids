"""E4 preregistration tests: fair history destruction and identical 444 input."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd

def frame():
    data={}
    for lag in (4,3,2,1,0):
        for name,mult in (("n0",1),("n1",2)):
            data[f"lag{lag}__{name}"]=(100*np.arange(8)+lag+mult).astype(np.float32)
        data[f"lag{lag}__proto"]=pd.Categorical(["tcp","udp","icmp","tcp",
                                                    "icmp","udp","tcp","udp"])
    return pd.DataFrame(data)

def test_e4_control_preserves_current_records_and_class_coherence():
    from scripts.run_window_e4 import perturb,seed_perm
    f=frame()
    cols=["n0","n1","proto"]
    block=perturb(f,cols,seed_perm(42,False,"block_shuffled"),"block_shuffled")
    independent=perturb(f,cols,seed_perm(42,False,"independent_shuffled"),"independent_shuffled")
    for control in (block,independent):
        for col in cols:
            assert control[f"lag0__{col}"].equals(f[f"lag0__{col}"])
        for lag in (1,2,3,4):
            expected=(control[f"lag{lag}__n1"]-control[f"lag{lag}__n0"])
            np.testing.assert_allclose(expected,1)
    # Whole history block shuffle keeps the relation of original lag4 with lag1.
    np.testing.assert_allclose(block["lag4__n0"]-block["lag1__n0"],3)
    assert seed_perm(42,False,"block_shuffled")!=seed_perm(42,True,"block_shuffled")
    assert seed_perm(42,False,"block_shuffled")!=seed_perm(42,False,"independent_shuffled")
    assert not block.equals(f)

def test_e4_protocol_no_official_test_and_exact_seed42():
    c=json.loads(Path("configs/window_e4_replication_001.json").read_text())
    e3=json.loads(Path("results/WINDOW-E3-001-S42/stats_rare_sqrt/metrics.json").read_text())
    assert c["official_test"]=="FORBIDDEN"
    assert c["seeds"]==[42,43,44,45,46]
    assert c["original_seed42_intact_model_sha256"]==e3["checkpoint_sha256"]
    assert c["original_seed42_intact_probabilities_sha256"]==e3["private_predictions_sha256"]
    assert c["engineered_total_columns"]==444
    assert c["frozen_original_val_targets"]==35005
    assert set(x["id"] for x in c["candidate_families"])=={
        "stats_rare_sqrt","block_shuffled_rare_sqrt","independent_shuffled_rare_sqrt"}
    for name in ("scripts/run_window_e4.py","scripts/archive_window_e4.py"):
        ast.parse(Path(name).read_text())
    nb=json.loads(Path("notebooks/19_window_e4_5seed_block_independent_shuffled_autobackup.ipynb").read_text())
    code="\n".join("".join(cell["source"]) for cell in nb["cells"]
                   if cell["cell_type"]=="code")
    ast.parse(code)
    assert "--auto-archive" in code
    assert 'get_secret("pushRare")' in code
    script=Path("scripts/run_window_e4.py").read_text()
    assert "restore_prior_candidate(root,seed,family" in script
    assert "archive_and_sync(root,seed,family" in script
    archive=Path("scripts/archive_window_e4.py").read_text()
    assert 'if not release["draft"]' in archive
    assert '"predictions_PRIVATE.npz"' in archive
