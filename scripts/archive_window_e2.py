"""WINDOW-E2-001 crash-safe private per-candidate draft archives + public scores.

No test data, validation prediction rows or fitted boosters are ever committed to
the public repo. Completed candidates can be restored after Kaggle restart.
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
from sklearn.metrics import confusion_matrix,f1_score
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
EXPERIMENT="WINDOW-E2-001-S42"
CANDIDATES=("window_unweighted","window_rare_sqrt")
PUBLIC_ROOT=("config_effective.json","data_lineage.json","feature_manifest.json",
             "control_comparison.json","source_code_commit.txt","training_environment.txt")
PUBLIC_PER_MODEL=("metrics.json","confusion.csv","per_class.csv","learning.csv")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4194304),b""):h.update(block)
    return h.hexdigest()

def j(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def validate(root,model):
    root=Path(root)
    if model not in CANDIDATES:raise ValueError("Unexpected E2 model")
    cfg=j(root/"config_effective.json")
    lineage=j(root/"data_lineage.json")
    if (cfg["experiment_id"]!=EXPERIMENT or cfg["official_test"]!="FORBIDDEN" or
        lineage["test_dataset_accessed"] is not False or
        lineage["train_target_rows"]!=140328 or
        lineage["validation_target_rows"]!=35005 or
        lineage["flat_features"]!=210):
        raise ValueError("Training-only, five-row protocol mismatch")
    folder=root/model
    score=j(folder/"metrics.json")
    if (score["candidate"]!=model or score["count"]!=35005 or
        score.get("official_test_used") is not False):
        raise ValueError("Invalid model result count or official-test protocol")
    if (score["checkpoint_sha256"]!=sha(folder/"model.txt") or
        score["private_predictions_sha256"]!=sha(folder/"predictions_PRIVATE.npz")):
        raise ValueError("Private model/prediction archive checksums failed")
    with np.load(folder/"predictions_PRIVATE.npz",allow_pickle=False) as dat:
        y=dat["y_true"].astype(np.int64)
        p=dat["probabilities"].astype(float)
        stored_prediction=dat["y_pred"].astype(int)
    if (p.shape!=(35005,10) or y.shape!=(35005,) or
        not np.isfinite(p).all() or np.any(y>9) or
        not np.array_equal(p.argmax(1),stored_prediction)):
        raise ValueError("Per-record model evidence malformed")
    cm=confusion_matrix(y,p.argmax(1),labels=np.arange(10))
    accuracy=float(np.trace(cm)/35005)
    macro=float(f1_score(y,p.argmax(1),labels=np.arange(10),
                         average="macro",zero_division=0))
    if (abs(accuracy-score["accuracy"])>1e-12 or
        abs(macro-score["macro_f1"])>1e-12):
        raise ValueError("Independent metric check on actual probability file failed")
    with (folder/"confusion.csv").open(newline="") as f:
        rows=list(csv.reader(f))
    classes=lineage["class_names"]
    if (len(rows)!=11 or rows[0][1:]!=classes or
        any(row[0]!=classes[i] for i,row in enumerate(rows[1:])) or
        not np.array_equal(np.array([[int(x) for x in row[1:]]
                                    for row in rows[1:]],dtype=np.int64),cm)):
        raise ValueError("Saved confusion matrix differs from actual predictions")
    public=[root/file for file in PUBLIC_ROOT]+[
        folder/file for file in PUBLIC_PER_MODEL]
    for pth in public:
        if not pth.is_file() or pth.is_symlink():
            raise FileNotFoundError("Missing public scientific evidence: "+str(pth))
    return public

def session(token):
    if not token:raise RuntimeError("Enable Kaggle pushRare for private archival")
    s=requests.Session()
    s.headers.update({"Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28"})
    return s

def api(s,method,path,**kwargs):
    r=s.request(method,f"https://api.github.com/repos/{REPO}/{path}",
                timeout=(45,180),**kwargs)
    r.raise_for_status()
    return r.json()

def release_for(s,tag):
    releases=api(s,"GET","releases",params={"per_page":100})
    return next((x for x in releases if x["tag_name"]==tag),None)

def tag_for(model):
    if model not in CANDIDATES:raise ValueError("Unknown candidate")
    return "window-e2-001-s42-"+model.replace("_","-")+"-full-artifacts"

def archive_and_sync(root,model,token):
    root=Path(root)
    public=validate(root,model)
    tag=tag_for(model)
    destination=root.parent/"archives"/f"{EXPERIMENT}_{model}_FULL_ARTIFACTS.tar"
    destination.parent.mkdir(parents=True,exist_ok=True)
    side=destination.with_suffix(".tar.sha256")
    if destination.exists():
        if not side.exists() or sha(destination)!=side.read_text().split()[0]:
            raise ValueError("Existing local full archive mismatched")
    else:
        wanted=[root/name for name in PUBLIC_ROOT]+[
            root/model/name for name in
            ("metrics.json","confusion.csv","per_class.csv","learning.csv",
             "model.txt","predictions_PRIVATE.npz")]
        catalog=[{"path":"run/"+str(p.relative_to(root)),
                  "sha256":sha(p),"bytes":p.stat().st_size} for p in wanted]
        manifest=(json.dumps({"experiment_id":EXPERIMENT,
                 "candidate":model,"private_artifact":True,
                 "official_test_used":False,"files":catalog},
                 indent=2,sort_keys=True)+"\n").encode()
        temp=destination.with_suffix(".partial")
        try:
            with tarfile.open(temp,"w") as tar:
                for item,p in tqdm(zip(catalog,wanted),total=len(wanted),
                                   desc="Freeze E2 "+model,unit="files"):
                    info=tar.gettarinfo(str(p),arcname=item["path"])
                    info.uid=info.gid=0;info.uname=info.gname=""
                    with p.open("rb") as handle:tar.addfile(info,handle)
                m=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
                m.size=len(manifest);m.mtime=0
                tar.addfile(m,io.BytesIO(manifest))
            os.replace(temp,destination)
        finally:
            temp.unlink(missing_ok=True)
        side.write_text(f"{sha(destination)}  {destination.name}\n")
    with session(token) as s:
        release=release_for(s,tag)
        if release is None:
            release=api(s,"POST","releases",json={"tag_name":tag,
                "target_commitish":"main","name":EXPERIMENT+" "+model+" frozen checkpoints",
                "draft":True,"prerelease":True,
                "body":"PRIVATE fitted booster and per-example probabilities; no official test. SHA256 "+sha(destination)})
        if not release["draft"]:
            raise ValueError("Refuse public release of raw predictions")
        known={item["name"]:item for item in release.get("assets",[])}
        for f,mime in ((destination,"application/x-tar"),(side,"text/plain")):
            remote=known.get(f.name)
            expected="sha256:"+sha(f)
            if remote is not None:
                if remote["size"]!=f.stat().st_size or remote.get("digest")!=expected:
                    raise ValueError("Existing draft differs: "+f.name)
                continue
            url=release["upload_url"].split("{")[0]
            class BarFile:
                def __init__(self,path):
                    self.f=path.open("rb")
                    self.p=tqdm(total=path.stat().st_size,desc="Upload "+model,
                                unit="B",unit_scale=True)
                def read(self,n=-1):
                    b=self.f.read(n)
                    self.p.update(len(b))
                    return b
                def close(self):
                    self.f.close();self.p.close()
            stream=BarFile(f)
            try:
                r=s.post(url,params={"name":f.name},
                    headers={"Content-Type":mime,
                             "Content-Length":str(f.stat().st_size)},
                    data=stream,timeout=(45,7200))
                r.raise_for_status()
                uploaded=r.json()
            finally:
                stream.close()
            if (uploaded["size"]!=f.stat().st_size or
                uploaded["state"]!="uploaded" or
                uploaded.get("digest") not in (None,expected)):
                raise ValueError("Private GitHub upload size/state/hash mismatch")
        commit=push_aggregates(s,root,public,
                 "results(e2): actual five-window "+model+" validation evidence")
    print("E2 PRIVATE DRAFT VERIFIED:",release["html_url"],flush=True)
    print("E2 PUBLIC AGGREGATES COMMIT:",f"https://github.com/{REPO}/commit/{commit}",flush=True)
    print("E2 FULL PRIVATE SHA256:",sha(destination),flush=True)

def push_aggregates(s,root,public,message):
    root=Path(root)
    head=api(s,"GET","git/ref/heads/main")["object"]["sha"]
    base=api(s,"GET","git/commits/"+head)["tree"]["sha"]
    tree=[{"path":f"results/{EXPERIMENT}/{p.relative_to(root)}",
           "mode":"100644","type":"blob","content":p.read_text(encoding="utf-8")}
          for p in public]
    created=api(s,"POST","git/trees",json={"base_tree":base,"tree":tree})["sha"]
    if created==base:return head
    commit=api(s,"POST","git/commits",json={
        "message":message,"tree":created,"parents":[head]})["sha"]
    api(s,"PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    return commit

def sync_final_summary(root,token):
    root=Path(root)
    summary=j(root/"summary.json")
    if (summary["experiment_id"]!=EXPERIMENT or
        summary["official_test_used"] is not False or
        set(summary["models"])!=set(CANDIDATES)|{"lgbm_unweighted","lgbm_rare_sqrt"}):
        raise ValueError("Final report incomplete")
    allpublic=[root/file for file in
               ("summary.json","comparison_all_models.csv","results_so_far.json")]
    for key in CANDIDATES:validate(root,key)
    with session(token) as s:
        commit=push_aggregates(s,root,allpublic,
            "results(e2): final exact-window vs frozen last-record LightGBM comparison")
    print("E2 FINAL GITHUB REPORT:",f"https://github.com/{REPO}/commit/{commit}",flush=True)

def restore_prior_candidate(root,model,token):
    root=Path(root)
    if (root/model/"model.txt").is_file() and (root/model/"metrics.json").is_file():
        validate(root,model)
        return True
    if not token:return False
    with session(token) as s:
        release=release_for(s,tag_for(model))
        if release is None:return False
        if not release["draft"]:raise ValueError("Private candidate archive cannot be public")
        item=next((x for x in release.get("assets",[]) if x["name"]==
                   f"{EXPERIMENT}_{model}_FULL_ARTIFACTS.tar"),None)
        if item is None:raise ValueError("Prior private release has no complete model archive")
        expected=item.get("digest","")
        if not expected.startswith("sha256:"):
            raise ValueError("Private saved archive missing expected digest")
        url=f"https://api.github.com/repos/{REPO}/releases/assets/{item['id']}"
        r=s.get(url,headers={"Accept":"application/octet-stream"},
                stream=True,timeout=(45,300))
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type",""):
            raise ValueError("Draft archive returned metadata, not bytes")
        target=root.parent/"archives"/item["name"]
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(".restore")
        with temporary.open("wb") as out,tqdm(total=item["size"],unit="B",
                       unit_scale=True,desc="Restore archived "+model) as bar:
            for b in r.iter_content(2097152):
                if b:out.write(b);bar.update(len(b))
        if temporary.stat().st_size!=item["size"] or sha(temporary)!=expected[7:]:
            temporary.unlink(missing_ok=True)
            raise ValueError("Restored GitHub private model checksum mismatch")
        temporary.replace(target)
    with tarfile.open(target,"r") as tar:
        members={m.name:m for m in tar.getmembers()}
        manifest=json.load(tar.extractfile("PRIVATE_ARTIFACT_CATALOGUE.json"))
        if (manifest["experiment_id"]!=EXPERIMENT or
            manifest["candidate"]!=model or
            manifest["official_test_used"] is not False):
            raise ValueError("Wrong immutable E2 private archive contents")
        for entry in manifest["files"]:
            rel=Path(entry["path"])
            if (rel.is_absolute() or rel.parts[0]!="run" or
                ".." in rel.parts or len(rel.parts)<2):
                raise ValueError("Unsafe private archive member path")
            member=members.get(str(rel))
            if member is None or not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Missing/unsafe E2 model archive path")
            p=root/Path(*rel.parts[1:])
            p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as f,p.open("wb") as dest:
                for buf in iter(lambda:f.read(4194304),b""):
                    dest.write(buf)
            if p.stat().st_size!=entry["bytes"] or sha(p)!=entry["sha256"]:
                p.unlink(missing_ok=True)
                raise ValueError("Restored E2 file hash mismatch: "+str(rel))
    validate(root,model)
    target.with_suffix(".tar.sha256").write_text(f"{sha(target)}  {target.name}\n")
    print("REUSED EXISTING GITHUB-VERIFIED E2 CANDIDATE:",model,flush=True)
    return True
