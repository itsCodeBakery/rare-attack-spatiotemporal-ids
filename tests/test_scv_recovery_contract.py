"""Post-crash recovery: source-level guard that preserves training evidence."""
from pathlib import Path

def test_recovery_does_not_contain_training_loop():
    text=Path("scripts/recover_scv_figures.py").read_text(encoding="utf-8")
    assert "project_saved_embeddings" in text
    assert "validation_predictions.npz" in text
    assert "scv_best.pt" in text
    assert "training_history.csv" in text
    assert ".fit(" not in text
    assert "optimizer.step(" not in text
    assert "model.fit(" not in text
