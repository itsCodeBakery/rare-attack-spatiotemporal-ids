"""Restore SCV checkpoints from pinned GitHub draft Release, no training or test data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
ASSET_ID=653575961
SIZE=136366080
EXPECTED="644629dd24e142906569ac771ec8c9ec4b86b47c6605f01cf96c307b8fc3bd00"
NEEDED=("checkpoints/scv_best.pt","SCV-IDS-REPRO-001/split_indices.npz",
        "SCV-IDS-REPRO-001/validation_predictions.npz",
        "SCV-IDS-REPRO-001/embedding_vectors.npz",
        "SCV-IDS-REPRO-001/preprocessing.json")

def checksum(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4194304),b""):h.update(chunk)
    return h.hexdigest()

def restore(base,archive,token):
    base=Path(base);archive=Path(archive)
    if all((base/p).is_file() for p in NEEDED):
        print("Frozen SCV artifacts found; no download or training necessary.")
        return
    if not archive.exists() or archive.stat().st_size!=SIZE or checksum(archive)!=EXPECTED:
        if not token:raise RuntimeError("Grant Kaggle access to pushRare to restore archived experiment.")
        url=f"https://api.github.com/repos/{REPO}/releases/assets/{ASSET_ID}"
        archive.parent.mkdir(parents=True,exist_ok=True)
        with requests.get(url,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/octet-stream",
            "X-GitHub-Api-Version":"2022-11-28"
        },stream=True,timeout=(45,300)) as response:
            response.raise_for_status()
            if "json" in response.headers.get("Content-Type",""):
                raise RuntimeError("GitHub returned draft asset metadata instead of archive bytes.")
            partial=archive.with_suffix(".download")
            with partial.open("wb") as dest,tqdm(total=SIZE,desc="Download frozen artifacts",
                         unit="B",unit_scale=True) as progress:
                for chunk in response.iter_content(2097152):
                    if chunk:dest.write(chunk);progress.update(len(chunk))
        if partial.stat().st_size!=SIZE or checksum(partial)!=EXPECTED:
            partial.unlink(missing_ok=True)
            raise ValueError("Downloaded archive checksum differs from verified GitHub digest.")
        partial.replace(archive)
    print("Pinned 130.05 MiB archive SHA-256 validated.")
    with tarfile.open(archive,"r") as tar:
        members={}
        for m in tar.getmembers():
            if m.isdir():continue
            if not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError("Unsafe archive member")
            if not (m.name.startswith("checkpoints/") or
                    m.name.startswith("SCV-IDS-REPRO-001/") or
                    m.name=="SCV-IDS-REPRO-001_ARCHIVE_MANIFEST.json"):
                raise ValueError("Unexpected archive member")
            members[m.name]=m
        manifest=json.load(tar.extractfile(members["SCV-IDS-REPRO-001_ARCHIVE_MANIFEST.json"]))
        for item in tqdm(manifest["files"],desc="Verify restored files",unit="files"):
            rel=item["relative_path"]
            if rel not in members:raise ValueError("Missing archived file: "+rel)
            target=base/rel;target.parent.mkdir(parents=True,exist_ok=True)
            sha=hashlib.sha256();size=0
            with tar.extractfile(members[rel]) as inp,target.open("wb") as out:
                for block in iter(lambda:inp.read(2097152),b""):
                    out.write(block);sha.update(block);size+=len(block)
            if sha.hexdigest()!=item["sha256"] or size!=item["size_bytes"]:
                target.unlink(missing_ok=True)
                raise ValueError("Stored file digest mismatch: "+rel)
    assert all((base/p).is_file() for p in NEEDED)
    print("ALL FROZEN RESEARCH ARTIFACTS RESTORED — NO TRAINING")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--archive",required=True)
    args=ap.parse_args()
    restore(args.base,args.archive,os.environ.get("GITHUB_TOKEN"))
