"""Synthetic fixture; does not claim real UNSW-NB15 performance or counts."""
import pandas as pd
from scripts.audit_unsw import audit

def test_audit_synthetic_split(tmp_path):
    cols={f"feature_{i:02d}":[1,2,2] for i in range(42)}
    tr=pd.DataFrame({"id":[1,2,3], **cols, "attack_cat":["Normal","Analysis","Analysis"],"label":[0,1,1]})
    te=pd.DataFrame({"id":[4,5],**{f"feature_{i:02d}":[2,3] for i in range(42)},
                    "attack_cat":["Analysis","Normal"],"label":[1,0]})
    a,b=tmp_path/"train.csv",tmp_path/"test.csv"
    tr.to_csv(a,index=False);te.to_csv(b,index=False)
    r=audit(a,b,tmp_path/"out")
    assert r["schema"]["common_feature_count"]==42
    assert r["classes"]["train"]["attack_category"]["Analysis"]==2
    assert r["duplicates"]["test_rows_sharing_train_fingerprint"]==1
    assert r["training_performed"] is False
