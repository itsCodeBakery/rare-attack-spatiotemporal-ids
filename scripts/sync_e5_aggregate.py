"""Allowlist-only public E5 aggregate artifact publisher.

No raw training features, per-record labels/IDs/predictions, model checkpoints,
original E3 booster, private token, official-test CSV or official test scores
may be committed from the E5 job.
"""
import json
from pathlib import Path
import requests

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
EXPERIMENT="E5-CONTEXT-DIAGNOSTIC-001"
ALLOWLIST=(
    "config_effective.json","source_order_summary.json",
    "source_transitions.csv","window_purity.csv","position_gaps.csv",
    "class_confusion.csv","classwise_errors.csv",
    "top_misclassifications.csv","confidence_bins.csv",
    "importance_by_group.csv","top_features.csv","position_ablation.csv",
    "summary.json","provenance.json"
)

def api(session,method,path,**kwargs):
    response=session.request(method,f"https://api.github.com/repos/{REPO}/{path}",
                             timeout=(45,180),**kwargs)
    response.raise_for_status()
    return response.json()

def push_public_e5(output,cfg,token):
    if not token:
        raise RuntimeError("Authorize Kaggle secret pushRare to save E5 public aggregate results")
    folder=Path(output)
    report=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    lineage=json.loads((folder/"provenance.json").read_text(encoding="utf-8"))
    if (cfg["official_test"]!="FORBIDDEN" or
        report["experiment_id"]!=EXPERIMENT or
        report["official_test_accessed"] is not False or
        report["trained_new_models"]!=0 or
        lineage["official_test_opened"] is not False or
        lineage["trained_boosters"]!=0):
        raise ValueError("Only true zero-training training-only E5 aggregate results can be published")
    manifest=[]
    for name in ALLOWLIST:
        file=folder/name
        if not file.is_file() or file.is_symlink():
            raise FileNotFoundError("Missing scientific public report: "+str(file))
        if file.stat().st_size>1_500_000:
            raise ValueError("Unexpected oversized E5 report: "+str(file))
        manifest.append({"path":f"results/{EXPERIMENT}/{name}",
            "mode":"100644","type":"blob","content":file.read_text(encoding="utf-8")})
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
          "Accept":"application/vnd.github+json",
          "X-GitHub-Api-Version":"2022-11-28"})
        head=api(session,"GET","git/ref/heads/main")["object"]["sha"]
        root=api(session,"GET","git/commits/"+head)["tree"]["sha"]
        newtree=api(session,"POST","git/trees",json={"base_tree":root,"tree":manifest})["sha"]
        if root==newtree:
            commit=head
        else:
            commit=api(session,"POST","git/commits",json={
                "message":"results(e5): train-only source-order, error and read-only E3 feature attribution audit",
                "tree":newtree,"parents":[head]})["sha"]
            api(session,"PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    print("E5 PUBLIC AGGREGATE RESEARCH COMMIT:",
          f"https://github.com/{REPO}/commit/{commit}",flush=True)
    print("E5 REPORTS (NO RAW DATA, NO TEST):",
          f"https://github.com/{REPO}/tree/main/results/{EXPERIMENT}",flush=True)
