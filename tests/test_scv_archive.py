"""Archive is specified to include binary model/prediction backups, no data source CSV."""
from pathlib import Path
import json
from scripts.archive_scv_outputs import main

def test_complete_backup_archive_synthetic(tmp_path):
    import tarfile
    base=tmp_path/"research_outputs"
    run=base/"SCV-IDS-REPRO-001"
    weights=base/"checkpoints"
    run.mkdir(parents=True);weights.mkdir(parents=True)
    for name in ("validation_predictions.npz","split_indices.npz","embedding_vectors.npz",
                 "metrics_validation.json","figure_manifest.json","artifact_manifest.json","environment.txt"):
        (run/name).write_bytes(b"synthetic fixture only")
    for name in ("scv_best.pt","scv_last_resume.pt"):
        (weights/name).write_bytes(b"synthetic fixture checkpoint")
    archive=tmp_path/"backup.tar"
    result=main(base,archive)
    assert result["source_file_count"]==9
    assert archive.stat().st_size>0
    with tarfile.open(archive,"r") as stream:
        names=stream.getnames()
        assert "checkpoints/scv_best.pt" in names
        assert "SCV-IDS-REPRO-001/validation_predictions.npz" in names
        assert "SCV-IDS-REPRO-001_ARCHIVE_MANIFEST.json" in names
        contents=json.load(stream.extractfile("SCV-IDS-REPRO-001_ARCHIVE_MANIFEST.json"))
        assert len(contents["files"])==9
