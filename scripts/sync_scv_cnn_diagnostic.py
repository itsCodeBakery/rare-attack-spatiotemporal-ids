"""Atomically sync only aggregated frozen-checkpoint diagnostic evidence to GitHub."""
import argparse
import base64
import json
import os
from pathlib import Path
import time
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX="results/SCV-CNN-DIAG-001"
CSV_FILES=("sample_support.csv","layer_statistics.csv","channel_statistics.csv",
           "branch_variance.csv","ablation_overall.csv","ablation_per_class.csv",
           "rgb_channel_stats.csv","flip_sensitivity.csv","sampled_confusion.csv")
DOC_FILES=("diagnostic_metrics.json","config_effective.json","figure_manifest.json")
MAX_TOTAL=50*1024*1024

def sync(folder,token):
    folder=Path(folder)
    if not token:raise RuntimeError("Kaggle pushRare secret must be authorized")
    files=[folder/p for p in CSV_FILES+DOC_FILES]
    m=json.loads((folder/"figure_manifest.json").read_text())
    if m["figure_count"]!=12:raise ValueError("Twelve plot definitions required")
    for item in m["plots"]:
        if len(item["exports"])!=3:raise ValueError("PNG/SVG/PDF source exports incomplete")
        files.extend(folder/e["path"] for e in item["exports"])
    if any(not f.is_file() for f in files):
        raise FileNotFoundError("One or more required aggregate outputs are missing")
    if len({str(f) for f in files})!=len(files):
        raise ValueError("Duplicate output paths in figure manifest")
    total=sum(f.stat().st_size for f in files)
    if total>MAX_TOTAL:raise ValueError("Diagnostic outputs too large for public sync")
    print(f"Synchronizing {len(files)} files ({total/1024**2:.2f} MiB).")
    url="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({
          "Authorization":"Bearer "+token,"Accept":"application/vnd.github+json",
          "X-GitHub-Api-Version":"2022-11-28"})
        def req(method,endpoint,**kwargs):
            response=None
            for attempt in range(4):
                response=session.request(method,url+"/"+endpoint,timeout=120,**kwargs)
                if response.status_code in (429,500,502,503,504):
                    time.sleep(min(16,2**attempt));continue
                response.raise_for_status()
                return response.json()
            response.raise_for_status()
        head=req("GET","git/ref/heads/main")["object"]["sha"]
        base_tree=req("GET","git/commits/"+head)["tree"]["sha"]
        entries=[]
        for file in tqdm(files,desc="Upload scientific figure source/outputs",unit="files"):
            relative=str(file.relative_to(folder)).replace(os.sep,"/")
            content=file.read_bytes()
            if file.suffix.lower() in (".png",".pdf"):
                digest=req("POST","git/blobs",json={
                    "content":base64.b64encode(content).decode("ascii"),
                    "encoding":"base64"})["sha"]
                entries.append({"path":PREFIX+"/"+relative,"mode":"100644",
                                "type":"blob","sha":digest})
            else:
                entries.append({"path":PREFIX+"/"+relative,"mode":"100644",
                                "type":"blob","content":content.decode("utf-8")})
        tree=req("POST","git/trees",json={"base_tree":base_tree,"tree":entries})["sha"]
        if tree==base_tree:
            print("Diagnostic already synced: https://github.com/"+REPO+"/commit/"+head)
            return head
        commit=req("POST","git/commits",json={
           "message":"results(diag): sync frozen CNN activation evidence and 12 figure sets",
           "tree":tree,"parents":[head]})["sha"]
        req("PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    print("VERIFIED CNN DIAGNOSTIC SYNC COMMIT: https://github.com/"+REPO+"/commit/"+commit)
    return commit

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    a=p.parse_args()
    sync(a.run,os.environ.get("GITHUB_TOKEN"))
