"""Synchronize authentic CTRL-TAB-001 aggregate LightGBM results. Never upload
private per-example probabilities, model weights or Python environment paths.

Inputs: frozen local run and GitHub token supplied by Kaggle pushRare secret.
Requires all two scored models, true confusion/F1 parity, 35005 saved targets.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX="results/CTRL-TAB-001-S42"
IDS=("lgbm_unweighted","lgbm_rare_sqrt")
TOP=("config_effective.json","data_lineage.json","summary.json",
     "aggregate_model_scores.csv","source_commit.txt","figure_manifest.json")
PER=("metrics.json","confusion.csv","per_class.csv","learning.csv")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4194304),b""):
            h.update(chunk)
    return h.hexdigest()

def verify_run(root):
    root=Path(root)
    summary=json.loads((root/"summary.json").read_text())
    lineage=json.loads((root/"data_lineage.json").read_text())
    fig=json.loads((root/"figure_manifest.json").read_text())
    if (summary.get("official_test_used") is not False or
        lineage.get("test_dataset_accessed") is not False or
        summary.get("reference_validation_target_count")!=35005 or
        lineage.get("validation_target_rows")!=35005 or
        lineage.get("validation_target_sequence_matches_original") is not True):
        raise ValueError("Locked original train-derived validation provenance missing")
    if len(fig["exports"])!=15:
        raise ValueError("Require all five actual plots × PNG/SVG/PDF")
    public=[root/x for x in TOP]
    results=summary["model_results"]
    if set(results)!=set(IDS):
        raise ValueError("Both original LightGBM candidates required")
    for model in IDS:
        d=root/model
        public.extend(d/p for p in PER)
        metrics=json.loads((d/"metrics.json").read_text())
        reference=results[model]["metrics"]
        if metrics!=reference:
            raise ValueError("Run summary vs candidate result mismatch for "+model)
        if (metrics.get("count")!=35005 or
            metrics.get("model_id")!=model or
            metrics.get("internal_validation_only") is not True):
            raise ValueError("Unexpected candidate target scope")
        if (metrics["checkpoint_sha256"]!=sha(d/"model.txt") or
            metrics["private_predictions_sha256"]!=sha(d/"predictions_PRIVATE.npz")):
            raise ValueError("Saved private candidate hash mismatch for "+model)
        with (d/"confusion.csv").open(newline="") as f:
            rows=list(csv.reader(f))
        if len(rows)!=11 or any(len(row)!=11 for row in rows):
            raise ValueError("Expected exactly ten-class confusion matrix")
        cm=[[int(v) for v in row[1:]] for row in rows[1:]]
        truth=[sum(x) for x in cm]
        predicted=[sum(cm[i][j] for i in range(10)) for j in range(10)]
        n=sum(truth)
        correct=sum(cm[i][i] for i in range(10))
        macro=sum((2*cm[i][i]/(truth[i]+predicted[i])
                   if truth[i]+predicted[i] else 0) for i in range(10))/10
        if (n!=35005 or correct!=metrics["correct"] or
            abs(correct/n-metrics["accuracy"])>1e-12 or
            abs(macro-metrics["macro_f1"])>1e-12):
            raise ValueError("Confusion-derived metrics contradict frozen reported scores")
        with (d/"per_class.csv").open(newline="") as f:
            per=list(csv.DictReader(f))
        if len(per)!=10 or any(int(row["support"])!=truth[i] for i,row in enumerate(per)):
            raise ValueError("Saved per-class support differs from actual confusion")
        print(f"VERIFIED {model}: macro-F1={100*macro:.3f}%, accuracy={100*correct/n:.3f}%")
    seen=set()
    for plot in fig["exports"]:
        rel=plot["file"]
        if rel in seen or not rel.startswith("figures/"):
            raise ValueError("Duplicate or unexpected figure entry")
        seen.add(rel)
        p=root/rel
        if p.stat().st_size!=plot["bytes"] or sha(p)!=plot["sha256"]:
            raise ValueError("Figure SHA-256 mismatch: "+rel)
        if plot["source"] not in {"aggregate_model_scores.csv",*[
           model+"/"+part for model in IDS for part in ("per_class.csv","confusion.csv")
        ]}:
            raise ValueError("Unknown numeric plot source: "+plot["source"])
        public.append(p)
    if len({str(p) for p in public})!=len(public):
        raise ValueError("Repeated public artifact paths")
    if any(not p.is_file() or p.is_symlink() for p in public):
        raise FileNotFoundError("Missing or symlinked public figure/metrics input")
    if sum(p.stat().st_size for p in public)>50*1024*1024:
        raise ValueError("Unexpectedly oversized public aggregate output")
    if any(("predictions_PRIVATE" in str(p) or p.name=="model.txt" or
             "environment" in p.name) for p in public):
        raise ValueError("Private artifact accidentally selected for public upload")
    return public

def sync(root,token):
    if not token:
        raise RuntimeError("Enable Kaggle secret pushRare; no token printed")
    root=Path(root)
    public=verify_run(root)
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"})
        def request(method,route,**kw):
            r=session.request(method,api+"/"+route,timeout=180,**kw)
            r.raise_for_status()
            return r.json()
        head=request("GET","git/ref/heads/main")["object"]["sha"]
        treebase=request("GET","git/commits/"+head)["tree"]["sha"]
        entries=[]
        for path in tqdm(public,desc="Version only verified tabular aggregate figures",unit="files"):
            blob=path.read_bytes()
            remote=PREFIX+"/"+str(path.relative_to(root)).replace(os.sep,"/")
            if path.suffix in (".png",".pdf"):
                s=request("POST","git/blobs",json={
                   "content":base64.b64encode(blob).decode("ascii"),
                   "encoding":"base64"})["sha"]
                entries.append({"path":remote,"mode":"100644","type":"blob","sha":s})
            else:
                entries.append({"path":remote,"mode":"100644","type":"blob",
                                "content":blob.decode("utf-8")})
        tree=request("POST","git/trees",json={
            "base_tree":treebase,"tree":entries})["sha"]
        if tree==treebase:
            print("TABULAR FIGURES UNCHANGED: https://github.com/"+REPO+"/commit/"+head)
            return head
        commit=request("POST","git/commits",json={
            "message":"results(ctrl003): authentic dual LightGBM internal validation; models private",
            "tree":tree,"parents":[head]})["sha"]
        request("PATCH","git/refs/heads/main",json={
            "sha":commit,"force":False})
    print("VERIFIED CTRL-TAB SYNC COMMIT: https://github.com/"+REPO+"/commit/"+commit)
    return commit

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    args=p.parse_args()
    sync(args.run,os.environ.get("GITHUB_TOKEN"))
