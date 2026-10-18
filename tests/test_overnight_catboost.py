"""Overnight saved-version notebook integrity; synthetic manifests only, no training."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

def test_clean_kaggle_notebook_and_seeds():
    notebook=json.loads(Path("notebooks/13_overnight_catboost_5seeds_autobackup.ipynb").read_text())
    source="\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code")
    ast.parse(source)
    assert "restore_scv_release.py" in source
    assert "restore_ctrl003_tabular.py" in source
    assert "run_overnight_catboost.py" in source
    assert 'get_secret("pushRare")' in source
    assert "--max-hours" in source
    assert "SCV-IDS-REPRO-001" in source
    assert "catboost>=1.2.8" in source
    runner=Path("scripts/run_overnight_catboost.py").read_text()
    assert "SEEDS=(42,43,44,45,46)" in runner
    assert "progress_summary(root,SEEDS)" in runner
    assert "archive_sync_catboost_overnight.py" in runner

def test_archival_protocol_never_versions_model_binaries_publicly():
    from scripts.archive_sync_catboost_overnight import SEEDS,PREFIX,digest
    assert SEEDS==(42,43,44,45,46)
    assert PREFIX=="results/CTRL-CAT-OVERNIGHT"
    code=Path("scripts/archive_sync_catboost_overnight.py").read_text()
    assert 'if any(p.name in ("predictions_PRIVATE.npz","trained_model.cbm")' in code
    assert 'if not release["draft"]' in code
    assert "score[\"macro_f1\"]" in code
    assert "score[\"accuracy\"]" in code
    assert "official_test_used" in code

def test_restorer_pins_original_frozen_lightgbm_sha():
    from scripts.restore_ctrl003_tabular import EXPECTED_SHA,EXPECTED_SIZE,ASSET_ID
    from scripts.restore_scv_release import EXPECTED as SCV_SHA
    cfg=json.loads(Path("configs/ctrl004_catboost_pareto_seed42.json").read_text())
    assert EXPECTED_SHA=="6867b1d6805d5aa26d94de61f63e2e0758d2d29d9f9518f6cc7bde6ab997f29d"
    assert EXPECTED_SIZE==21207040
    assert ASSET_ID==653965961
    assert len(SCV_SHA)==64
    assert cfg["official_test"]=="LOCKED"

def test_summary_recaps_fixed_seed_scores_only(tmp_path):
    from scripts.run_overnight_catboost import progress_summary
    keys=["catboost_standard","catboost_balanced"]+[f"mix_{i}" for i in range(8)]
    for seed in (42,43):
        p=tmp_path/f"S{seed}"
        p.mkdir()
        rows=[{"candidate":k,"accuracy":.83,"macro_f1":.61,
             "weighted_precision":.85,"weighted_recall":.83,
             "weighted_f1":.82,"balanced_accuracy":.60,
             "backdoor_TP":10,"analysis_TP":20,"doS_TP":100}
             for k in keys]
        pd.DataFrame(rows).to_csv(p/"summary_scores.csv",index=False)
    assert progress_summary(tmp_path,(42,43,44,45,46))==20
    merged=pd.read_csv(tmp_path/"ACROSS_SEED_SUMMARY.csv")
    assert (merged["completed_seeds"]==2).all()
    assert set(merged["candidate"])==set(keys)
