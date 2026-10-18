"""CTRL-CAT-001 configuration/lineage regression tests (no GPU or CatBoost needed)."""
import ast
import json
from pathlib import Path

def test_targets_and_frozen_protocol():
    cfg=json.loads(Path("configs/ctrl004_catboost_pareto_seed42.json").read_text())
    assert cfg["experiment_id"]=="CTRL-CAT-001-S42"
    assert cfg["official_test"]=="LOCKED"
    assert cfg["paper_thresholds"]=={
        "accuracy":.8577,
        "precision_aggregate":.8516,
        "recall_aggregate":.8577,
        "f1_aggregate":.8459,
        "macro_f1":.4687}
    assert len(cfg["train_csv_sha256"])==64
    assert len(cfg["split_sha256"])==64
    assert len(cfg["models"])==2
    assert len(cfg["fixed_ensembles"])==8
    assert cfg["current_development_baseline"]["validation_examples"]==35005
    assert cfg["catboost"]["seed"]==42
    assert cfg["catboost"]["task_type"]=="CPU"
    assert cfg["catboost"]["selection_metric"]=="MultiClass logarithmic loss"
    for mix in cfg["fixed_ensembles"]:
        assert len(mix["parents"])==len(mix["weights"])
        assert all(w>=0 for w in mix["weights"])
        assert abs(sum(mix["weights"])-1)<1e-12
    script=Path("scripts/run_ctrl004_catboost_pareto.py").read_text()
    assert "def verify_baselines(" in script
    assert "source_code_commit.txt" in script
    assert "official_test_used" in script
    assert "NO_OFFICIAL_SUPERIORITY_CLAIM" in script
    assert "np.array_equal" in script
    ast.parse(script)

def test_real_source_checksum_matches_original_v0():
    original=json.loads(Path("configs/step07_scv_ids_repro_001.json").read_text())
    candidate=json.loads(Path("configs/ctrl004_catboost_pareto_seed42.json").read_text())
    frozen=json.loads(Path("configs/ctrl003_fast_tabular.json").read_text())
    assert (candidate["train_csv_sha256"]==original["official_training_sha256"]
            ==frozen["train_csv_sha256"])
    assert candidate["split_sha256"]==frozen["reference_split_sha256"]
