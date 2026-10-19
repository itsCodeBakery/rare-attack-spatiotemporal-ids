"""WINDOW-OFFICIAL-001 frozen benchmark protocol: synthetic, no official test data."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

def test_pretest_method_freeze_matches_five_seed_e3():
    cfg=json.loads(Path("configs/window_official_001_frozen.json").read_text())
    trials=[int(v) for v in cfg["frozen_e3_five_seed_best_iterations"].values()]
    assert sorted(trials)==[36,48,58,61,74]
    assert cfg["final_train_iterations"]==int(np.median(trials))==58
    assert cfg["window_size"]==5
    assert cfg["full_training_window_count"]==175337
    assert cfg["official_test_window_count"]==82328
    assert cfg["original_training_rows"]==175341
    assert cfg["original_testing_rows"]==82332
    assert cfg["seed"]==42
    assert cfg["frozen_before_official_test"] is True
    assert cfg["no_additional_training_variants"] is True
    assert cfg["use_test_for_training"] is False
    assert cfg["use_official_test_for_early_stopping"] is False
    assert cfg["frozen_e3_feature_manifest_git_blob_sha"]=="590d0991ad56ab07b140618f0dd32e6cc9c3a792"

def test_true_original_444_features_and_train_only_categorical_encoding():
    from scripts.run_official_001 import (
        paths_and_schema,prepare_partition,CATEGORIES,CLASS_ORDER
    )
    cfg=json.loads(Path("configs/window_official_001_frozen.json").read_text())
    names,numerical,raw,stats,cat=paths_and_schema(Path("."),cfg)
    assert len(names)==42 and len(numerical)==39
    assert len(raw)==210 and len(stats)==234 and len(cat)==15
    assert len(CLASS_ORDER)==10
    frame={}
    for col in names:
        if col in CATEGORIES:
            frame[col]=["tcp","udp","tcp","icmp","tcp","udp","tcp","icmp"]
        else:
            frame[col]=np.arange(8,dtype=np.float32)+0.1*names.index(col)
    training=pd.DataFrame(frame)
    A,categoricals,vocabs=prepare_partition(training,names,numerical,raw,stats)
    assert A.shape==(4,444)
    assert list(A.columns)==raw+stats
    test=training.copy()
    test["proto"]=["unseen"]*8
    B,c2,_=prepare_partition(test,names,numerical,raw,stats,vocabs)
    assert B.shape==(4,444)
    assert c2==categoricals
    assert B["lag0__proto"].isna().all()
    assert "attack_cat" not in A.columns and "label" not in A.columns

def test_single_test_reporting_metrics():
    from scripts.run_official_001 import all_metrics
    labels=np.arange(10)
    p=np.eye(10,dtype=np.float32)*.99+.001
    m,cm,per,pred=all_metrics(labels,p)
    assert m["n"]==10 and m["accuracy"]==1
    assert m["macro_f1"]==1 and m["weighted_f1"]==1
    assert m["precision_macro"]==1 and m["precision_weighted"]==1
    assert m["binary_attack_recall"]==1
    assert cm.shape==(10,10) and len(per)==10
    np.testing.assert_array_equal(pred,labels)

def test_archive_freeze_precedes_any_test_read_and_safe_output():
    path=Path("scripts/run_official_001.py")
    code=path.read_text()
    assert code.index("archive_pretest(out,os.environ.get") < code.index("test_frame=pd.read_csv(test_file")
    assert code.index("model.save_model(str(model_file)") < code.index("test_frame=pd.read_csv(test_file")
    assert 'out.joinpath("official_evaluation_complete.json").exists()' in code
    assert '"official_test_used":True' in code
    assert "num_boost_round=cfg[\"final_train_iterations\"]" in code
    archive=Path("scripts/archive_official_001.py").read_text()
    assert 'if not current["draft"]' in archive
    assert '"official_test_predictions_PRIVATE.npz"' in archive
    assert "if score[\"official_predictions_sha256\"]!=sha(npz)" in archive
    for p in (path,Path("scripts/archive_official_001.py")):
        ast.parse(p.read_text())

def test_official_one_cell_notebook_syntax():
    nb=json.loads(Path("notebooks/20_official_unsw_locked_single_test_autobackup.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"]=="code":
            body="".join(cell["source"])
            ast.parse(body)
            assert "--auto-archive" in body
            assert "get_secret(\"pushRare\")" in body
            assert "UNSW_NB15_testing-set.csv" in body
