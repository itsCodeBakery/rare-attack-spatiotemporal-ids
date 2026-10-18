"""Publish only certified aggregate CARES-PAIR-001 results and plots to GitHub.

Binary expert checkpoints and per-example model probabilities remain exclusively in
the unpublished, checksum-verified private release.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import requests
from tqdm.auto import tqdm
if __package__:
    from scripts.archive_cares_pair001 import validated_outputs, CANDIDATES, SPECIALISTS
else:
    from archive_cares_pair001 import validated_outputs, CANDIDATES, SPECIALISTS

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX="results/CARES-PAIR-001-S42"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as file:
        for part in iter(lambda:file.read(4194304),b""):
            h.update(part)
    return h.hexdigest()

def select_public(root):
    root=Path(root)
    validated_outputs(root)
    fig=json.loads((root/"figure_manifest.json").read_text())
    if (fig["figure_count"]!=12 or fig["export_count"]!=36 or
        fig["experiment_id"]!="CARES-PAIR-001-S42"):
        raise ValueError("CARES figure manifest absent or not complete")
    files=[root/p for p in (
        "summary.json","comparison_scores.csv","config_effective.json",
        "data_lineage.json","source_commit.txt","figure_manifest.json")]
    for name in CANDIDATES:
        files.extend(root/name/p for p in ("metrics.json","confusion.csv","per_class.csv"))
    for name in SPECIALISTS:
        files.extend(root/("specialist_"+name)/p
                     for p in ("specialist_info.json","earlystop_history.csv"))
    for src in fig["numeric_sources"]:
        p=root/src["file"]
        if not p.is_file() or p.stat().st_size!=src["bytes"] or sha(p)!=src["sha256"]:
            raise ValueError("Figure numeric source mismatch: "+src["file"])
    for figure in fig["figures"]:
        if len(figure["exports"])!=3:
            raise ValueError("One or more scientific figure exports missing")
        for exported in figure["exports"]:
            p=root/exported["file"]
            if p.stat().st_size!=exported["bytes"] or sha(p)!=exported["sha256"]:
                raise ValueError("Figure binary SHA-256 mismatch: "+str(p))
            files.append(p)
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise FileNotFoundError("Required aggregate or plot source not found")
    if len(files)!=len({str(x) for x in files}):
        raise ValueError("Duplicate public artifact")
    if sum(p.stat().st_size for p in files)>75*1024*1024:
        raise ValueError("Public figure footprint exceeds expected maximum")
    if any(p.name in ("specialist.txt","predictions_PRIVATE.npz",
                       "validation_specialist_PRIVATE.npz") for p in files):
        raise ValueError("Private per-record data selected for public GitHub")
    return files

def sync(root,token):
    if not token:
        raise RuntimeError("Kaggle pushRare secret must be enabled to sync")
    root=Path(root);files=select_public(root)
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"})
        def req(method,route,**kwargs):
            r=session.request(method,api+"/"+route,timeout=180,**kwargs)
            r.raise_for_status()
            return r.json()
        head=req("GET","git/ref/heads/main")["object"]["sha"]
        base=req("GET","git/commits/"+head)["tree"]["sha"]
        changes=[]
        for p in tqdm(files,desc="Synchronize certified CARES metrics and figures",unit="files"):
            rel=PREFIX+"/"+str(p.relative_to(root)).replace(os.sep,"/")
            blob=p.read_bytes()
            if p.suffix in (".png",".pdf"):
                obj=req("POST","git/blobs",json={
                    "content":base64.b64encode(blob).decode(),
                    "encoding":"base64"})["sha"]
                changes.append({"path":rel,"mode":"100644","type":"blob","sha":obj})
            else:
                changes.append({"path":rel,"mode":"100644","type":"blob",
                                "content":blob.decode("utf-8")})
        tree=req("POST","git/trees",json={
            "base_tree":base,"tree":changes})["sha"]
        if tree==base:
            print("CARES AGGREGATE UNCHANGED: https://github.com/"+REPO+"/commit/"+head)
            return head
        commit=req("POST","git/commits",json={
            "message":"results(cares001): actual six-variant ablation, classwise metrics, 36 figures; private model archive",
            "tree":tree,"parents":[head]})["sha"]
        req("PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    print("VERIFIED CARES RESULTS SYNC COMMIT: https://github.com/"+REPO+"/commit/"+commit)
    return commit

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    sync(args.run,os.environ.get("GITHUB_TOKEN"))
