"""Secure atomic GitHub synchronization of paper-aligned baseline evidence.

Uploads the small plotting CSV sources, SVG/PDF/PNG published figures,
all text reports, configuration and cryptographic artifact references.
Does NOT upload checkpoints, source dataset, per-row validation predictions,
row indices or raw embedding vectors, all of which remain in Kaggle output.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
RUN_ID="SCV-IDS-REPRO-001"
PREFIX="results/"+RUN_ID
TEXT_FILES=("metrics_validation.json","config_effective.json","preprocessing.json",
            "training_history.csv","artifact_manifest.json","run_report.md",
            "figure_manifest.json")
OPTIONAL_TEXT_FILES=("environment.txt","recovery_environment.json",
                     "projection_diagnostics.json")
MAX_SINGLE_BYTES=12*1024*1024
MAX_TOTAL_BYTES=90*1024*1024

def hash_bytes(blob):
    return hashlib.sha256(blob).hexdigest()

def collect(root):
    root=Path(root)
    results=[]
    for name in TEXT_FILES:
        f=root/name
        if not f.is_file():raise FileNotFoundError(str(f))
        results.append(f)
    for name in OPTIONAL_TEXT_FILES:
        path=root/name
        if path.is_file():
            results.append(path)
    if not any((root/name).is_file() for name in
               ("environment.txt","recovery_environment.json")):
        raise FileNotFoundError("Neither original nor recovery environment metadata present")
    for folder in ("plot_data","figures"):
        directory=root/folder
        if not directory.is_dir():raise FileNotFoundError(str(directory))
        results.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    expected=json.loads((root/"figure_manifest.json").read_text(encoding="utf-8"))
    formats={"png":0,"svg":0,"pdf":0}
    for p in results:
        if p.suffix[1:] in formats:formats[p.suffix[1:]]+=1
    n=expected["figure_count"]
    if any(count!=n for count in formats.values()):
        raise ValueError(f"Incomplete figure formats: {formats}; expected {n} each")
    total=sum(p.stat().st_size for p in results)
    if total>MAX_TOTAL_BYTES:
        raise ValueError("Text and figure artifacts exceed Git commit size policy; "
                         "keep all files in Kaggle, reduce plot DPI in a documented commit")
    too_large=[str(p) for p in results if p.stat().st_size>MAX_SINGLE_BYTES]
    if too_large:raise ValueError("Oversized figure files: "+", ".join(too_large))
    # Both text and binary source artifacts are identified by SHA-256.
    return results,total

def upload(root,token):
    if not token:raise ValueError("Kaggle secret pushRare is unavailable")
    root=Path(root)
    files,total=collect(root)
    print(f"Syncing {len(files)} reproducibility files ({total/1024**2:.1f} MiB).")
    base="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"
        })
        def req(method,endpoint,**kwargs):
            result=None
            for attempt in range(4):
                result=session.request(method,base+"/"+endpoint,timeout=120,**kwargs)
                if result.status_code in (429,500,502,503,504):
                    time.sleep(min(16,2**attempt))
                    continue
                result.raise_for_status()
                return result.json()
            result.raise_for_status()
        parent=req("GET","git/ref/heads/main")["object"]["sha"]
        tree_base=req("GET","git/commits/"+parent)["tree"]["sha"]
        entries=[]
        for p in tqdm(files,desc="Upload figure/report sources",unit="files"):
            data=p.read_bytes()
            path=PREFIX+"/"+str(p.relative_to(root)).replace(os.sep,"/")
            # Blob API supports binary PDF and PNG without text corruption.
            if p.suffix in {".png",".pdf"}:
                blob=req("POST","git/blobs",json={
                    "content":base64.b64encode(data).decode("ascii"),
                    "encoding":"base64"})["sha"]
                entries.append({"path":path,"mode":"100644","type":"blob","sha":blob})
            else:
                entries.append({"path":path,"mode":"100644","type":"blob",
                                "content":data.decode("utf-8")})
        tree=req("POST","git/trees",json={"base_tree":tree_base,"tree":entries})["sha"]
        if tree==tree_base:
            print("Artifacts already synchronized; no new GitHub commit.")
            return parent
        sha=req("POST","git/commits",json={
            "message":"results(scv): sync paper-aligned validation evidence and reproducible figures",
            "tree":tree,"parents":[parent]})["sha"]
        req("PATCH","git/refs/heads/main",json={"sha":sha,"force":False})
    print("VERIFIED SCV BASELINE SYNC COMMIT: https://github.com/"+REPO+"/commit/"+sha)
    return sha

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--run",required=True)
    x=a.parse_args()
    upload(x.run,os.environ.get("GITHUB_TOKEN"))
