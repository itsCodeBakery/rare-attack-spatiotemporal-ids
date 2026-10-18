"""Atomically sync compact ViT control metrics and figures; never weights or predictions."""
import argparse
import base64
import json
import os
from pathlib import Path
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX="results/CTRL-VIT-001-S42"
REQUIRED=("config_effective.json","data_lineage.json","status.json",
          "metrics_validation.json","artifact_manifest.json","run_report.md",
          "figure_manifest.json","source_code_commit.txt","training_history.csv")
MAX_TOTAL=75*1024*1024

def sync(root,token):
    if not token:raise RuntimeError("Enable Kaggle secret pushRare")
    root=Path(root)
    fig=json.loads((root/"figure_manifest.json").read_text(encoding="utf-8"))
    if fig["figure_count"]!=14:raise ValueError("Expected 14 versioned figures")
    files=[root/x for x in REQUIRED]
    files += sorted((root/"plot_data").glob("*.csv"))
    for entry in fig["plots"]:
        if len(entry["exports"])!=3:raise ValueError("Missing one of PNG SVG PDF")
        files.extend(root/e["file"] for e in entry["exports"])
    if any(not p.is_file() for p in files):raise FileNotFoundError("Missing experiment file")
    if len({str(p) for p in files})!=len(files):raise ValueError("Duplicate manifest paths")
    if sum(p.stat().st_size for p in files)>MAX_TOTAL:
        raise ValueError("Compact public figures exceed upload budget")
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"})
        def req(method,end,**kw):
            r=session.request(method,api+"/"+end,timeout=180,**kw)
            r.raise_for_status()
            return r.json()
        head=req("GET","git/ref/heads/main")["object"]["sha"]
        treebase=req("GET","git/commits/"+head)["tree"]["sha"]
        entries=[]
        for path in tqdm(files,desc="Version ViT metrics and publication figures",unit="files"):
            blob=path.read_bytes()
            p=PREFIX+"/"+str(path.relative_to(root)).replace(os.sep,"/")
            if path.suffix in (".png",".pdf"):
                h=req("POST","git/blobs",json={
                    "content":base64.b64encode(blob).decode(),
                    "encoding":"base64"})["sha"]
                entries.append({"path":p,"mode":"100644","type":"blob","sha":h})
            else:
                entries.append({"path":p,"mode":"100644","type":"blob",
                                "content":blob.decode("utf-8")})
        tree=req("POST","git/trees",json={"base_tree":treebase,"tree":entries})["sha"]
        if tree==treebase:
            print("CONTROL VIT FIGURES UNCHANGED: https://github.com/"+REPO+"/commit/"+head)
            return head
        commit=req("POST","git/commits",json={
            "message":"results(ctrl001): standalone ViT same-split validation and 42 reproducible figures",
            "tree":tree,"parents":[head]})["sha"]
        req("PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    print("VERIFIED CTRL-VIT SYNC COMMIT: https://github.com/"+REPO+"/commit/"+commit)
    return commit

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    a=p.parse_args()
    sync(a.run,os.environ.get("GITHUB_TOKEN"))
