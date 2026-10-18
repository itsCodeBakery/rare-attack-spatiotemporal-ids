"""Restore pinned frozen CTRL-TAB baseline from private GitHub Release.

For a Kaggle Save & Run All job, /kaggle/working starts CLEAN; downloading only
the existing, hash-pinned 20.22 MiB trained-model archive prevents retraining.
No designated official test and no raw dataset in the archived bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
ASSET_ID=653965961
EXPECTED_SIZE=21207040
EXPECTED_SHA="6867b1d6805d5aa26d94de61f63e2e0758d2d29d9f9518f6cc7bde6ab997f29d"
RUN="CTRL-TAB-001-S42"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4194304),b""):
            h.update(chunk)
    return h.hexdigest()

def restore(base,archive,token):
    base=Path(base);archive=Path(archive)
    dest=base/RUN
    must=[dest/"summary.json",dest/"lgbm_unweighted"/"model.txt",
          dest/"lgbm_rare_sqrt"/"model.txt",
          dest/"lgbm_unweighted"/"predictions_PRIVATE.npz",
          dest/"lgbm_rare_sqrt"/"predictions_PRIVATE.npz"]
    if all(p.is_file() for p in must):
        for model in ("lgbm_unweighted","lgbm_rare_sqrt"):
            info=json.loads((dest/model/"metrics.json").read_text())
            if (info["checkpoint_sha256"]!=digest(dest/model/"model.txt") or
                info["private_predictions_sha256"]!=digest(dest/model/"predictions_PRIVATE.npz")):
                raise ValueError("Existing frozen baseline hashes failed")
        print("PINNED LIGHTGBM ARCHIVE ALREADY RESTORED",flush=True)
        return dest
    if not archive.is_file() or archive.stat().st_size!=EXPECTED_SIZE or digest(archive)!=EXPECTED_SHA:
        if not token:
            raise RuntimeError("Kaggle pushRare is needed to fetch private original baseline")
        url=f"https://api.github.com/repos/{REPO}/releases/assets/{ASSET_ID}"
        archive.parent.mkdir(parents=True,exist_ok=True)
        temp=archive.with_suffix(".partial_download")
        try:
            with requests.get(url,headers={
                "Authorization":"Bearer "+token,
                "Accept":"application/octet-stream",
                "X-GitHub-Api-Version":"2022-11-28"},
                stream=True,timeout=(45,300)) as res:
                res.raise_for_status()
                if "json" in res.headers.get("Content-Type",""):
                    raise RuntimeError("GitHub returned metadata, not baseline archive bytes")
                with temp.open("wb") as f,tqdm(total=EXPECTED_SIZE,unit="B",
                    unit_scale=True,desc="Restore frozen LightGBM") as bar:
                    for chunk in res.iter_content(2097152):
                        if chunk:f.write(chunk);bar.update(len(chunk))
            if temp.stat().st_size!=EXPECTED_SIZE or digest(temp)!=EXPECTED_SHA:
                raise ValueError("Downloaded LightGBM archive hash mismatch")
            temp.replace(archive)
        finally:
            temp.unlink(missing_ok=True)
    with tarfile.open(archive,"r") as tar:
        members={m.name:m for m in tar.getmembers()}
        if "PRIVATE_ARTIFACT_CATALOGUE.json" not in members:
            raise ValueError("Private baseline archive has no catalog")
        manifest=json.load(tar.extractfile(members["PRIVATE_ARTIFACT_CATALOGUE.json"]))
        if manifest.get("experiment_id")!=RUN or manifest.get("test_partition_used") is not False:
            raise ValueError("Unrecognized baseline archive provenance")
        for item in tqdm(manifest["files"],desc="Verify restored baseline",unit="files"):
            rel=Path(item["path"])
            if (rel.parts[0]!="run" or rel.is_absolute() or
                ".." in rel.parts or len(rel.parts)<2):
                raise ValueError("Archive path traversal blocked")
            member=members.get(str(rel))
            if member is None or not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Missing or unsafe archived file: "+str(rel))
            dest_path=dest/Path(*rel.parts[1:])
            dest_path.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            with tar.extractfile(member) as src,dest_path.open("wb") as target:
                for buf in iter(lambda:src.read(4194304),b""):
                    target.write(buf);h.update(buf);count+=len(buf)
            if h.hexdigest()!=item["sha256"] or count!=item["bytes"]:
                dest_path.unlink(missing_ok=True)
                raise ValueError("Archived item checksum invalid: "+str(rel))
    assert all(p.is_file() for p in must)
    print("PINNED 20.22 MiB LIGHTGBM MODEL ARCHIVE VERIFIED — NO RETRAINING",flush=True)
    return dest

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--base",required=True)
    parser.add_argument("--archive",required=True)
    args=parser.parse_args()
    restore(args.base,args.archive,os.environ.get("GITHUB_TOKEN"))
