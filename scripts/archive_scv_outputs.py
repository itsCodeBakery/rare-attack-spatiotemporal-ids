"""Create a downloadable complete SCV run archive from existing Kaggle artifacts.

No retraining, no raw UNSW dataset, no tokens. Not uploaded to GitHub.
Archive preserves checkpoint weights, original prediction probabilities,
frozen embedding vectors, split indices, source CSV tables, every figure,
metric and source-provenance file. Verify with SHA-256 manifest.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from tqdm.auto import tqdm

RUN="SCV-IDS-REPRO-001"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def main(base,output):
    base=Path(base)
    run=base/RUN
    check=base/"checkpoints"
    if not run.is_dir() or not check.is_dir():
        raise FileNotFoundError("Run/checkpoint data not found; do not retrain automatically")
    must=[run/"validation_predictions.npz",run/"split_indices.npz",
          run/"embedding_vectors.npz",run/"metrics_validation.json",
          run/"figure_manifest.json",run/"artifact_manifest.json",
          run/"environment.txt",check/"scv_best.pt",check/"scv_last_resume.pt"]
    if any(not f.is_file() for f in must):
        raise FileNotFoundError("Required results missing: "+
            ", ".join(str(f) for f in must if not f.is_file()))
    paths=sorted(p for p in (list(run.rglob("*"))+list(check.rglob("*"))) if p.is_file())
    if any(p.is_symlink() for p in paths):
        raise ValueError("Refuse symlinks in research output archive")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    entries=[]
    for p in tqdm(paths,desc="Hash original reproducibility files",unit="files"):
        entries.append({"relative_path":str(p.relative_to(base)),
                        "size_bytes":p.stat().st_size,"sha256":digest(p)})
    manifest={
        "archive_type":"SCV-IDS full reproducibility backup; NO raw dataset",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "files":entries,
        "source_training_status":"50 epochs completed before postprocessing crash",
        "training_repeated_for_this_archive":False,
        "official_test_evaluated":False,
        "note":"Archive is a separate preservation copy; public GitHub stores only compact numeric sources and figures."}
    blob=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode("utf-8")
    temp=output.with_suffix(".partial")
    if temp.exists():temp.unlink()
    try:
        with tarfile.open(temp,"w") as archive:
            for p in tqdm(paths,desc="Archive complete experimental evidence",unit="files"):
                entry=archive.gettarinfo(str(p),arcname=str(p.relative_to(base)))
                entry.uid=entry.gid=0
                entry.uname=entry.gname=""
                with p.open("rb") as f:
                    archive.addfile(entry,f)
            info=tarfile.TarInfo("SCV-IDS-REPRO-001_ARCHIVE_MANIFEST.json")
            info.size=len(blob)
            info.mtime=0
            info.uid=info.gid=0
            archive.addfile(info,io.BytesIO(blob))
        os.replace(temp,output)
    finally:
        if temp.exists():temp.unlink()
    summary={"file":str(output),"size_bytes":output.stat().st_size,
             "sha256":digest(output),"source_file_count":len(entries),
             "download_and_store_separately":True,
             "note":"Save this tar file outside the temporary Kaggle session, or publish as a private Kaggle dataset."}
    (output.parent/(output.stem+"_checksum.json")).write_text(
         json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("FULL RUN ARCHIVE COMPLETE: "+str(output),flush=True)
    print("ARCHIVE SHA-256: "+summary["sha256"],flush=True)
    print("DOWNLOAD OR VERSION THE ARCHIVE BEFORE KAGGLE SESSION EXPIRES",flush=True)
    return summary

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--base",required=True)
    a.add_argument("--archive",required=True)
    x=a.parse_args()
    main(x.base,x.archive)
