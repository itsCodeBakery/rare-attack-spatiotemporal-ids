"""CARES first-run preservation smoke tests. Synthetic numerical fixtures ONLY."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def test_generated_real_source_figures_from_synthetic_fixture(tmp_path):
    from scripts.plot_cares_pair001 import plot
    keys=("baseline","dos_only","backdoor_only","analysis_only",
          "all_unrestricted","all_selective")
    names=("Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic",
           "Normal","Reconnaissance","Shellcode","Worms")
    (tmp_path/"data_lineage.json").write_text(json.dumps({"class_names":names}))
    (tmp_path/"summary.json").write_text(json.dumps({"experiment_id":"CARES-PAIR-001-S42"}))
    scores=[]
    for i,key in enumerate(keys):
        folder=tmp_path/key;folder.mkdir()
        f1=np.linspace(.15,.90,10)+(i/300)
        pd.DataFrame({"class":names,"support":np.full(10,5),
                      "precision":f1,"recall":f1,"f1":f1}).to_csv(
                      folder/"per_class.csv",index=False)
        mat=np.eye(10,dtype=int)*5
        pd.DataFrame(mat,index=names,columns=names).rename_axis(
            "true_class").to_csv(folder/"confusion.csv")
        scores.append({"variant":key,"accuracy":.75+i/100,
             "macro_f1":.61+i/100,"balanced_accuracy":.6+i/100,
             "changed_predictions_vs_baseline":i*20,
             "DoS_correct":22-i,
             "Backdoor_correct":4+i,"DoS_to_Exploits":i*10,
             "Backdoor_to_Exploits":i+1})
    pd.DataFrame(scores).to_csv(tmp_path/"comparison_scores.csv",index=False)
    result=plot(tmp_path)
    assert result["figure_count"]==12
    assert result["export_count"]==36
    assert len(result["numeric_sources"])==13
    for ext in ("png","svg","pdf"):
        assert len(list((tmp_path/"figures").glob("*."+ext)))==12
    assert (tmp_path/"figure_manifest.json").is_file()

def test_all_six_variants_archiving_remains_private():
    from scripts.archive_cares_pair001 import CANDIDATES,SPECIALISTS,upload
    assert len(CANDIDATES)==6 and len(SPECIALISTS)==3
    assert upload.TAG=="cares-pair-001-s42-full-artifacts"
    assert "draft" in Path("scripts/archive_ctrl003_tabular.py").read_text()
    code=Path("scripts/sync_cares_pair001.py").read_text()
    assert "predictions_PRIVATE.npz" in code
    assert "specialist.txt" in code
    assert 'if any(p.name in ("specialist.txt","predictions_PRIVATE.npz",' in code
    assert "official_test_used" in Path("scripts/archive_cares_pair001.py").read_text()
