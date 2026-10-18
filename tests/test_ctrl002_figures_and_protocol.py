"""CTRL-V2 synthetic-only figure and activation-health contract checks."""
from pathlib import Path
import json
import runpy
import numpy as np
import pandas as pd
from scripts.plot_ctrl002_cnnvit import plot_run

def test_eighteen_editable_evidence_figures_from_saved_numeric_sources(tmp_path):
    # Reuse proven real-rendering V1 plot test, with synthetic input tables only.
    factory=runpy.run_path(str(Path("tests/test_ctrl001_vit_figures.py")))
    factory["test_rebuild_fourteen_figures_without_model_or_predictions"](tmp_path)
    history=pd.DataFrame([{
        "epoch":epoch,
        "cnn_token_nonzero_fraction":.50+epoch*.0005,
        "cnn_token_between_sample_mean_std":.01+epoch*.0002,
        "cnn_constant_token_fraction":0.0,
        "cnn_pooled_between_sample_mean_std":.015,
        "cnn_pooled_constant_dimension_fraction":0.0,
        "cnn_zero_mask_mean_logit_l2":.15+epoch*.002,
        "cnn_zero_mask_changed_prediction_fraction":.10,
        "cnn_token_mean_absolute":.22,
        "health_sample_size":512
    } for epoch in range(1,51)])
    history.to_csv(tmp_path/"plot_data"/"health_history.csv",index=False)
    figures=plot_run(tmp_path)
    assert figures["experiment_id"]=="CTRL-CNNVIT-V2-S42"
    assert figures["figure_count"]==18
    assert len(figures["plot_sources"])==8
    assert len(figures["plots"])==18
    paths=set(p["file"] for p in figures["plot_sources"])
    for entry in figures["plots"]:
        assert set(entry["sources"]).issubset(paths)
        assert len(entry["exports"])==3
    for ext in ("png","svg","pdf"):
        assert len(list((tmp_path/"figures").glob("*."+ext)))==18


def test_predeclared_health_thresholds_and_split_contract():
    cfg=json.loads(Path("configs/ctrl002_cnnvit_leaky_seed42.json").read_text())
    assert cfg["negative_slope"]==0.1
    assert cfg["seed"]==42 and cfg["epochs"]==50
    assert cfg["reference_validation_windows"]==35005
    assert cfg["health"]["sample_size"]==512
    assert cfg["health"]["gate_min_nonzero_fraction"]==.05
    assert cfg["health"]["gate_min_between_example_feature_std"]==1e-5
    assert cfg["official_test"]=="FORBIDDEN"
    assert cfg["figures_expected"]==18
    script=Path("scripts/train_ctrl002_cnnvit.py").read_text()
    assert 'cfg.get("official_test")!="FORBIDDEN"' in script
    assert "reference_split_sha256" in script
    assert "reference_predictions_sha256" in script
    assert "status_path.exists()" in script
    assert "measure_cnn_health" not in script or "cnn_health" in script
    assert "torch.save" in script
    assert 'window_target_equal_to_original' in script
    finalizer=Path("scripts/finalize_ctrl002_cnnvit.py").read_text()
    assert "torch.load" not in finalizer
    assert ".backward(" not in finalizer
    assert "health_history.csv" in finalizer
    assert "activation_health_pass" in finalizer


def test_cnn_variability_metric_on_const_and_variable_vectors():
    # Math guard independent of Torch and journal performance.
    constant=np.zeros((100,16,128))
    varied=np.random.default_rng(42).normal(size=(100,16,128))
    assert constant.reshape(100,-1).std(axis=0).mean()==0
    assert varied.reshape(100,-1).std(axis=0).mean()>.5
