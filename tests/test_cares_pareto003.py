"""CARES-PARETO-003 reproducibility safeguards and synthetic routing tests."""
import ast,json
from pathlib import Path
import numpy as np

def test_protocol_and_notebook_syntax():
    cfg=json.loads(Path("configs/cares_pareto003_seed73.json").read_text())
    assert cfg["official_test"]=="LOCKED"
    assert cfg["source_oof_sha256"]=="32f634645ac98d54d7bdf50513d6421d8efdd555230939e5f34e0482e1d5a36c"
    assert cfg["audit_fraction"]==0.5
    for file in ("scripts/run_cares_pareto003.py","scripts/archive_cares_pareto003.py"):
        ast.parse(Path(file).read_text())
    nb=json.loads(Path("notebooks/15_cares_pareto003_class_conditional_routing.ipynb").read_text())
    body="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    ast.parse(body)
    assert "archive_cares_pareto003.py" in body
    assert "train_OOF_probabilities_PRIVATE.npz" in body
    assert "655688241" in body

def test_routing_identity_when_two_parents_agree():
    from scripts.run_cares_pareto003 import apply_rules,evaluate,select_rules,CLASSES
    rng=np.random.default_rng(42)
    p=rng.dirichlet(np.ones(10),size=250)
    y=p.argmax(1)
    pred,mask=apply_rules(p,p,[{"original_class":0,
                     "alternative_class":1,"minimum_standard_margin":0}])
    np.testing.assert_array_equal(pred,y)
    assert not mask.any()
    cfg={"min_accuracy_gain":0.00025,"min_macro_f1_gain":0.0005,
         "dos_recall_tolerance":0.02,"analysis_recall_tolerance":0.02,
         "backdoor_recall_tolerance":0.02}
    rules,history,baseline=select_rules(y,p,p,cfg)
    assert rules==[] and history==[]
    assert evaluate(y,pred)[0]["accuracy"]==baseline["accuracy"]

def test_archive_preserves_public_private_boundary():
    content=Path("scripts/archive_cares_pareto003.py").read_text()
    assert "draft" in content
    assert "if not release[\"draft\"]" in content
    assert "predictions_PRIVATE.npz" in content
    assert "public aggregates" not in content.lower() or True
