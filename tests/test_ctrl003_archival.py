"""Public repo must never expose LightGBM weights or per-record predictions."""
from pathlib import Path
import importlib.util

def test_tabular_public_sync_import_and_private_paths():
    script=Path("scripts/sync_ctrl003_tabular.py")
    spec=importlib.util.spec_from_file_location("tab_sync",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PREFIX=="results/CTRL-TAB-001-S42"
    assert len(module.IDS)==2
    assert len(module.PER)==4
    assert not any("model.txt" in p or "predictions_PRIVATE" in p for p in module.PER)
    assert not any("model.txt" in p or "predictions_PRIVATE" in p for p in module.TOP)

def test_private_archiver_isolation_and_tag():
    text=Path("scripts/archive_ctrl003_tabular.py").read_text()
    assert 'TAG="ctrl-tab-001-s42-full-artifacts"' in text
    assert '"draft":True' in text
    assert '"private_artifact":True' in text
    assert 'predictions_PRIVATE.npz' in text
    assert 'model.txt' in text
    assert "weights.rglob" not in text
