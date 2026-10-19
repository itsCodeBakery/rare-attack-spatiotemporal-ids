"""Official UNSW evaluation: pre-test model notarization and one-time outcome archive.

Save booster privately before official test is opened and publish the frozen
provenance hash; publish actual *aggregate* test results only after independent
recomputation, keeping per-row official predictions in unpublished releases.
"""
import csv
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import f1_score,confusion_matrix
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
EXPERIMENT="WINDOW-OFFICIAL-001-S42"
MODEL_TAG="window-official-001-s42-model-frozen-pretest"
RESULT_TAG="window-official-001-s42-test-evidence-private"
MODEL_FILE="WINDOW-OFFICIAL-001-S42_MODEL_PRETEST_PRIVATE.tar"
RESULT_FILE="WINDOW-OFFICIAL-001-S42_TEST_EVIDENCE_PRIVATE.tar"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def gh(token):
    if not token:
        raise RuntimeError("Enable existing Kaggle GitHub secret pushRare for private audit trail")
    s=requests.Session()
    s.headers.update({"Authorization":"Bearer "+token,
       "Accept":"application/vnd.github+json",
       "X-GitHub-Api-Version":"2022-11-28"})
    return s

def api(s,method,path,**kw):
    response=s.request(method,f"https://api.github.com/repos/{REPO}/{path}",
                       timeout=(45,180),**kw)
    response.raise_for_status()
    return response.json()

def release(s,tag,title,model_hash):
    exists=api(s,"GET","releases",params={"per_page":100})
    current=next((x for x in exists if x["tag_name"]==tag),None)
    if current is None:
        current=api(s,"POST","releases",json={
             "tag_name":tag,"target_commitish":"main","draft":True,
             "prerelease":True,"name":title,
             "body":"Private UNSW-NB15 experiment artifact; model frozen before test. Model SHA256 "+model_hash})
    if not current["draft"]:
        raise ValueError("Private model/probabilities may not be published")
    return current

def tar_manifest(destination,folder,filenames,stage):
    folder,destination=Path(folder),Path(destination)
    selected=[folder/f for f in filenames]
    for file in selected:
        if not file.is_file() or file.is_symlink():
            raise FileNotFoundError("Missing private stage file: "+str(file))
    catalogue=[{"name":file.name,"size":file.stat().st_size,
                "sha256":sha(file)} for file in selected]
    if destination.exists():
        # Existing same-name local immutable archive must reproduce contents.
        with tarfile.open(destination,"r") as tar:
            old=json.load(tar.extractfile("PRIVATE_ARTIFACT_CATALOGUE.json"))
        if (old.get("stage")!=stage or old.get("files")!=catalogue):
            raise ValueError("Existing immutable stage archive differs from exact files")
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest=(json.dumps({"experiment_id":EXPERIMENT,"stage":stage,
                          "official_test_used":stage=="test",
                          "private_artifact":True,"files":catalogue},
                         indent=2,sort_keys=True)+"\n").encode()
    temp=destination.with_suffix(".partial")
    try:
        with tarfile.open(temp,"w") as tar:
            for p in tqdm(selected,desc="Freeze "+stage+" evidence",unit="files"):
                info=tar.gettarinfo(str(p),arcname="run/"+p.name)
                info.uid=0;info.gid=0;info.uname=info.gname=""
                with p.open("rb") as handle:
                    tar.addfile(info,handle)
            info=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
            info.size=len(manifest)
            tar.addfile(info,io.BytesIO(manifest))
        os.replace(temp,destination)
    finally:
        temp.unlink(missing_ok=True)

def upload_release(s,tag,title,model_sha,files):
    published=release(s,tag,title,model_sha)
    byname={x["name"]:x for x in published.get("assets",[])}
    for file in files:
        expected="sha256:"+sha(file)
        saved=byname.get(file.name)
        if saved:
            if saved["size"]!=file.stat().st_size or saved.get("digest")!=expected:
                raise ValueError("Previously frozen GitHub draft archive differs: "+file.name)
            continue
        class Progress:
            def __init__(self,path):
                self.file=path.open("rb")
                self.bar=tqdm(desc="Private GitHub upload "+path.name,
                              total=path.stat().st_size,unit="B",unit_scale=True)
            def read(self,n=-1):
                part=self.file.read(n)
                self.bar.update(len(part))
                return part
            def close(self):
                self.file.close()
                self.bar.close()
        data=Progress(file)
        try:
            response=s.post(published["upload_url"].split("{")[0],
                params={"name":file.name},
                headers={"Content-Type":"application/x-tar" if file.suffix==".tar" else "text/plain",
                         "Content-Length":str(file.stat().st_size)},
                data=data,timeout=(45,7200))
            response.raise_for_status()
            result=response.json()
        finally:
            data.close()
        if (result.get("state")!="uploaded" or
            result["size"]!=file.stat().st_size or
            result.get("digest") not in (None,expected)):
            raise ValueError("Private draft asset verification failed: "+file.name)
    return published

def commit_public(s,folder,filenames,commit_message):
    folder=Path(folder)
    paths=[]
    for name in filenames:
        file=folder/name
        if not file.is_file() or file.is_symlink() or file.name.endswith("_PRIVATE.npz") or "_PRIVATE" in file.name:
            raise ValueError("Never commit private model/data: "+str(file))
        paths.append(file)
    head=api(s,"GET","git/ref/heads/main")["object"]["sha"]
    root=api(s,"GET","git/commits/"+head)["tree"]["sha"]
    tree=[{"path":f"results/{EXPERIMENT}/{p.name}",
          "mode":"100644","type":"blob","content":p.read_text(encoding="utf-8")}
          for p in paths]
    newtree=api(s,"POST","git/trees",json={"base_tree":root,"tree":tree})["sha"]
    if newtree==root:return head
    commit=api(s,"POST","git/commits",json={
          "tree":newtree,"parents":[head],"message":commit_message})["sha"]
    api(s,"PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    return commit

def archive_pretest(folder,token):
    folder=Path(folder)
    frozen=read(folder/"PRETEST_FROZEN_CHECKPOINT.json")
    meta=read(folder/"data_lineage.json")
    cfg=read(folder/"config_effective.json")
    if (frozen["experiment_id"]!=EXPERIMENT or
        frozen["official_test_opened"] is not False or
        frozen["selected_before_test"] is not True or
        cfg["status"]!="PRE_REGISTERED_BEFORE_OPENING_TEST" or
        meta["official_test_opened_before_checkpoint"] is not False or
        meta["official_test_used_for_training"] is not False or
        frozen["model_checkpoint_sha256"]!=sha(folder/"final_model_PRIVATE.txt")):
        raise ValueError("Pre-test model isn't securely frozen")
    candidates=("final_model_PRIVATE.txt","training_category_vocab_PRIVATE.json",
       "config_effective.json","PRETEST_FROZEN_CHECKPOINT.json",
       "data_lineage.json","feature_manifest.json",
       "source_code_commit.txt","training_environment.txt")
    destination=folder.parent/"official_archives"/MODEL_FILE
    tar_manifest(destination,folder,candidates,"pretest")
    side=destination.with_suffix(".tar.sha256")
    if side.exists() and side.read_text().split()[0]!=sha(destination):
        raise ValueError("Frozen pretest full archive mismatch")
    side.write_text(f"{sha(destination)}  {destination.name}\n")
    with gh(token) as session:
        rel=upload_release(session,MODEL_TAG,
             "Frozen BEFORE official test — 444-feature full train",
             frozen["model_checkpoint_sha256"],(destination,side))
        commit=commit_public(session,folder,(
             "config_effective.json","PRETEST_FROZEN_CHECKPOINT.json",
             "data_lineage.json","feature_manifest.json",
             "source_code_commit.txt","training_environment.txt"),
             "protocol(official001): notarize frozen full-training model before opening official test")
    print("OFFICIAL PRE-TEST MODEL DRAFT:",rel["html_url"],flush=True)
    print("OFFICIAL PRE-TEST PUBLIC FREEZE COMMIT:",
          f"https://github.com/{REPO}/commit/{commit}",flush=True)
    print("OFFICIAL PRE-TEST ARCHIVE SHA256:",sha(destination),flush=True)

def audit_test(folder):
    folder=Path(folder)
    cfg=read(folder/"config_effective.json")
    check=read(folder/"PRETEST_FROZEN_CHECKPOINT.json")
    score=read(folder/"official_metrics.json")
    completed=read(folder/"official_evaluation_complete.json")
    if (check["model_checkpoint_sha256"]!=sha(folder/"final_model_PRIVATE.txt") or
        score["model_checkpoint_sha256"]!=check["model_checkpoint_sha256"] or
        completed["model_checkpoint_sha256"]!=check["model_checkpoint_sha256"] or
        score["official_test_used"] is not True or
        score["used_for_tuning"] is not False or
        score["used_for_training"] is not False or
        score["n"]!=cfg["official_test_window_count"] or
        score["fixed_training_rounds"]!=cfg["final_train_iterations"]):
        raise ValueError("Official test results violate frozen pre-test checkpoint")
    npz=folder/"official_test_predictions_PRIVATE.npz"
    if score["official_predictions_sha256"]!=sha(npz):
        raise ValueError("Raw test probability archive SHA256 drift")
    with np.load(npz,allow_pickle=False) as d:
        y=d["y_true"].astype(int)
        p=d["probabilities"].astype(float)
        yp=d["y_pred"].astype(int)
        indices=d["official_test_row_indices"].astype(int)
    if (p.shape!=(cfg["official_test_window_count"],10) or
        not np.array_equal(p.argmax(1),yp) or
        not np.array_equal(indices,np.arange(4,cfg["original_testing_rows"]))):
        raise ValueError("Original official target indexing differs")
    cm=confusion_matrix(y,yp,labels=np.arange(10))
    acc=float(np.trace(cm)/len(y))
    macro=float(f1_score(y,yp,average="macro",labels=np.arange(10),zero_division=0))
    weighted=float(f1_score(y,yp,average="weighted",labels=np.arange(10),zero_division=0))
    if any(abs(v-score[k])>1e-12 for k,v in (
        ("accuracy",acc),("macro_f1",macro),("weighted_f1",weighted))):
        raise ValueError("Independent actual official-test accuracy/F1 recomputation failed")
    with (folder/"official_confusion.csv").open(newline="") as f:
        rows=list(csv.reader(f))
    classes=read(folder/"data_lineage.json")["class_names"]
    if (len(rows)!=11 or rows[0][1:]!=classes or
        not np.array_equal(cm,[[int(n) for n in row[1:]] for row in rows[1:]])):
        raise ValueError("Recorded official class confusion differs from actual predictions")
    return score

def archive_final(folder,token):
    folder=Path(folder)
    score=audit_test(folder)
    wanted=("official_test_predictions_PRIVATE.npz",
        "official_metrics.json","official_confusion.csv",
        "official_per_class.csv","paper_comparison.json",
        "official_evaluation_complete.json",
        "PRETEST_FROZEN_CHECKPOINT.json","config_effective.json")
    destination=folder.parent/"official_archives"/RESULT_FILE
    tar_manifest(destination,folder,wanted,"test")
    side=destination.with_suffix(".tar.sha256")
    if side.exists() and side.read_text().split()[0]!=sha(destination):
        raise ValueError("Previously frozen private test archive mismatch")
    side.write_text(f"{sha(destination)}  {destination.name}\n")
    with gh(token) as session:
        rel=upload_release(session,RESULT_TAG,
             "Private raw official-test evidence — single frozen E3 model",
             score["model_checkpoint_sha256"],(destination,side))
        pub=commit_public(session,folder,(
            "official_metrics.json","official_confusion.csv",
            "official_per_class.csv","paper_comparison.json",
            "official_evaluation_complete.json"),
            "results(official001): independent raw-probability-audited one-time frozen UNSW test metrics")
    print("OFFICIAL PRIVATE TEST ARCHIVE:",rel["html_url"],flush=True)
    print("OFFICIAL FINAL VERIFIED GITHUB COMMIT:",
          f"https://github.com/{REPO}/commit/{pub}",flush=True)
    print("OFFICIAL RAW TEST ARCHIVE SHA256:",sha(destination),flush=True)
