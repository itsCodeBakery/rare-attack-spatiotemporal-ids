"""Offline smoke test: recover 28 plots from only synthetic frozen artifacts.

No synthetic observations or figures are published as benchmark results.
"""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from scripts.recover_scv_figures import recovery

CLASSES=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
         "Normal","Reconnaissance","Shellcode","Worms"]

def test_recover_all_50_epoch_evidence_without_any_training(tmp_path):
    plot_test=runpy.run_path(str(Path("tests/test_scv_figure_regeneration.py")))
    plot_test["test_all_figures_reproduce_from_numeric_csv_sources"](tmp_path)
    out=tmp_path
    weights=tmp_path/"checkpoints"
    weights.mkdir()
    for name in ("scv_best.pt","scv_last_resume.pt"):
        (weights/name).write_bytes(b"synthetic checkpoint marker; no model weights")
    y=np.repeat(np.arange(10,dtype=np.uint8),np.arange(5,15))
    n=len(y)
    p=np.full((n,10),.01,dtype=np.float32)
    p[np.arange(n),y]=.91
    p/=p.sum(axis=1,keepdims=True)
    idx=np.arange(1000,1000+n+4,dtype=np.int64)
    np.savez_compressed(out/"validation_predictions.npz",y_true=y,y_pred=y,
                        probabilities=p,
                        validation_window_end_indices=idx[4:])
    dev=np.arange(60,dtype=np.int64)
    np.savez_compressed(out/"split_indices.npz",
                        dev_train_indices=dev,validation_indices=idx,
                        dev_train_labels=np.arange(60,dtype=np.int64)%10,
                        validation_labels=np.concatenate([[0,1,2,3],y]),
                        classes=np.asarray(CLASSES))
    sampled=np.arange(min(n,40))
    rng=np.random.default_rng(77)
    np.savez_compressed(out/"embedding_vectors.npz",
                        sample_index=sampled,y=y[sampled],
                        raw=rng.normal(size=(len(sampled),42)).astype(np.float32),
                        cnn=rng.normal(size=(len(sampled),128)).astype(np.float32),
                        vit=np.zeros((len(sampled),128),dtype=np.float32),
                        fused=rng.normal(size=(len(sampled),128)).astype(np.float32))
    history=pd.DataFrame([{
        "epoch":i,"train_loss":.5,"val_loss":.3,"train_accuracy":.9,
        "val_accuracy":1.0,"val_macro_f1":1.0,"val_weighted_f1":1.0,
        "learning_rate":.001,"epoch_seconds":2.0,
        "peak_gpu_memory_mb":100.0} for i in range(1,51)])
    history.to_csv(out/"training_history.csv",index=False)
    history.to_csv(out/"plot_data"/"training_history.csv",index=False)
    (out/"config_effective.json").write_text(json.dumps({
        "experiment_id":"SCV-IDS-REPRO-001",
        "official_test_csv":"DO_NOT_READ_DURING_THIS_EXPERIMENT",
        "official_training_sha256":"synthetic-test-hash",
        "training":{"epochs":50},"preprocessing":{"window":5}}))
    (out/"preprocessing.json").write_text("{}")
    (out/"environment.txt").write_text("Git commit: original-training-sha\n")
    result=recovery(out,weights,Path("."))
    assert result["original_training_finished_epochs"]==50
    assert result["figure_count"]==28
    assert result["official_test_used_for_training_validation_or_recovery"] is False
    assert len(list((out/"figures").glob("*.svg")))==28
    assert len(list((out/"figures").glob("*.pdf")))==28
    assert len(list((out/"figures").glob("*.png")))==28
    assert result["pca_diagnostics"]["vit"]["status"]=="zero_variance"
    assert abs(result["validation_performance_from_frozen_probabilities"]["macro_f1"]-1.0)<1e-9
    assert (out/"figures"/"27_pca_vit.svg").is_file()
    assert result["original_training_source_git_sha"]=="original-training-sha"
    assert len(result["heavyweight_files"])==5
