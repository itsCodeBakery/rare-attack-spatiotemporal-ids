"""Tests for CARES-OOF-002 training-only risk gate; synthetic data only."""
import json
import ast
from pathlib import Path
import numpy as np
def test_protocol_and_notebook():
 c=json.loads(Path("configs/cares_oof002_seed42.json").read_text())
 assert c["official_test"]=="FORBIDDEN"
 assert c["folds"]==3 and c["reference_validation_examples"]==35005
 assert c["variants"]==["frozen_base","simple_average","meta_full","risk_gated"]
 assert len(c["frozen_standard_predictions_sha256"])==64
 assert len(c["frozen_rare_predictions_sha256"])==64
 for name in ["scripts/run_cares_oof002.py","scripts/archive_sync_cares_oof002.py"]:
  ast.parse(Path(name).read_text())
 notebook=json.loads(Path("notebooks/14_cares_oof002_proposed_method.ipynb").read_text())
 for cell in notebook["cells"]:
  if cell["cell_type"]=="code":ast.parse("".join(cell["source"]))
def test_gate_selects_identity_on_nondiscriminating_candidate():
 from scripts.run_cares_oof002 import find_gate,gated,normalize
 rng=np.random.default_rng(9)
 p=rng.dirichlet(np.ones(10),size=200)
 labels=p.argmax(1)
 conf={"candidate_alpha":[.25,.5,1.0],
       "candidate_gate_threshold":[0,.1],
       "max_allowed_meta_gate_accuracy_loss":0,
       "min_meta_gate_macro_f1_gain":.001}
 cl=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Normal","Reconnaissance","Shellcode","Worms"]
 chosen,records,base=find_gate(labels,p,p,cl,conf)
 assert chosen["alpha"]==0
 assert len(records)==6
 corrected,accept=gated(p,p,chosen["alpha"],chosen["threshold"])
 np.testing.assert_allclose(corrected,normalize(p))
 assert not np.any(accept)
def test_routing_never_opens_test_set():
 s=Path("scripts/run_cares_oof002.py").read_text()
 assert "prepare(source,ref)" in s
 assert "gate_ids" in s
 assert "train_test_split" in s
 assert "restore_parent" in s
 assert "source_validation_untouched_for_meta_training" in s
 assert "official_test_used" in s
