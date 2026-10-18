"""DIAG-001 synthetic-only figure and no-training contract tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from scripts.plot_scv_cnn_diagnostic import make_figures

CLASSES=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
         "Normal","Reconnaissance","Shellcode","Worms"]


def test_plot_reproducibility_for_all_twelve_figures(tmp_path):
    layers=["0","1","2","3","4","5","6","7","8","9","10","11","12"]
    pd.DataFrame([{
        "layer":k,"channels":32,"between_image_mean_std":i*.001,
        "max_channel_std":i*.002,"constant_fraction":0.0 if i else 1.0,
        "spatial_std":.01*i,"zero_fraction":.05*i}
        for i,k in enumerate(layers)]).to_csv(tmp_path/"layer_statistics.csv",index=False)
    pd.DataFrame([{
        "layer":k,"channel":c,"between_image_std":.001*i+c*.0001,
        "within_image_spatial_std":i*.003,
        "zero_fraction":.05*i}
        for i,k in enumerate(layers) for c in range(6)]
    ).to_csv(tmp_path/"channel_statistics.csv",index=False)
    names=["cnn_pooled","cnn_tokens","vit_pooled",
           "vit_to_cnn_attention","cnn_to_vit_attention","fused"]
    pd.DataFrame([{"branch":name,"dimensions":128,
       "examples":120,"mean_feature_std":.001*i,
       "max_feature_std":.003*i,"constant_dimension_fraction":0.,
       "mean_vector_norm":.2*i} for i,name in enumerate(names)]
    ).to_csv(tmp_path/"branch_variance.csv",index=False)
    modes=["original","cnn_zeroed","vit_zeroed","both_zeroed"]
    pd.DataFrame([{"mode":k,"count":150,"accuracy":.8-.1*i,
       "macro_f1":.6-.1*i,"disagreement_with_original":i*.05,
       "mean_logit_l2_change":i*.5} for i,k in enumerate(modes)]
    ).to_csv(tmp_path/"ablation_overall.csv",index=False)
    pd.DataFrame([{"mode":k,"class":c,"support":15,
       "precision":.8-i*.1,"recall":.7-i*.1,
       "f1":.75-i*.1} for i,k in enumerate(modes) for c in CLASSES]
    ).to_csv(tmp_path/"ablation_per_class.csv",index=False)
    pd.DataFrame([{"channel":c,"pixel_mean":.25+i*.1,
                   "pixel_std":.17+i*.04,"fraction_exact_zero":.1*i}
                  for i,c in enumerate(("R","G","B"))]
    ).to_csv(tmp_path/"rgb_channel_stats.csv",index=False)
    pd.DataFrame([{"branch":c,"median_l2":.1+i,
                   "mean_l2":.2+i,"p90_l2":.3+i,
                   "fraction_near_zero":0.} for i,c in enumerate(("cnn","vit"))]
    ).to_csv(tmp_path/"flip_sensitivity.csv",index=False)
    pd.DataFrame(np.diag([15]*10),columns=CLASSES,index=CLASSES).rename_axis(
        "true_class").to_csv(tmp_path/"sampled_confusion.csv")
    pd.DataFrame({"class":CLASSES,"sample_support":[15]*10}).to_csv(
        tmp_path/"sample_support.csv",index=False)
    m=make_figures(tmp_path)
    assert m["figure_count"]==12
    assert len(m["plots"])==12
    assert len(list((tmp_path/"figures").glob("*.png")))==12
    assert len(list((tmp_path/"figures").glob("*.svg")))==12
    assert len(list((tmp_path/"figures").glob("*.pdf")))==12
    saved=json.loads((tmp_path/"figure_manifest.json").read_text())
    assert saved["figure_count"]==12
    assert all(len(item["exports"])==3 for item in saved["plots"])
    assert all(item["source_files"] for item in saved["plots"])


def test_diag_source_forbids_model_fitting_and_official_test():
    text=Path("scripts/run_scv_cnn_diagnostic.py").read_text()
    assert "torch.no_grad()" in text
    assert 'cfg["official_test"]!="FORBIDDEN"' in text
    assert "model.load_state_dict" in text
    assert "torch.load" in text
    assert "model.fit(" not in text
    assert "optimizer.step(" not in text
    assert "loss.backward(" not in text
    assert "validation_predictions.npz" in text
