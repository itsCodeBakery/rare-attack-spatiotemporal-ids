"""E6 immutable private booster/prediction backups and aggregate GitHub sync.

Only scientific aggregate data may be committed publicly. Private archives
remain unpublished GitHub draft releases. Fully trained models can be restored
without repeating any fit when a Kaggle runtime restarts.
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
EXPERIMENT="E6-ORDER-ROBUSTNESS-001"
NAMES=("Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Normal",
       "Reconnaissance","Shellcode","Worms")
FAMILIES=("target_full","intact_full","shuffled_full",
          "target_nonoverlap","intact_nonoverlap")
PUBLIC_META=("config_effective.json","split_provenance.json",
             "model_feature_manifest.json","source_code_commit.txt","environment.txt")
PUBLIC_MODEL=("metrics.json","confusion.csv","per_class.csv",
              "subgroups.csv","top40_features.csv")
PRIVATE_EXTRA=("split_indices_PRIVATE.npz",)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()

def read_json(file):return json.loads(Path(file).read_text())

def validate(root,model):
    root=Path(root)
    if model not in FAMILIES:raise ValueError("Unknown preregistered E6 candidate")
    cfg=read_json(root/"config_effective.json")
    prov=read_json(root/"split_provenance.json")
    score=read_json(root/model/"metrics.json")
    if (cfg["official_test"]!="FORBIDDEN" or
        cfg["experiment_id"]!=EXPERIMENT or
        prov["official_test_opened"] is not False or
        score["official_test_accessed"] is not False or
        score["test_used_for_tuning"] is not False or
        score["validation_used_for_early_stopping"] is not False or
        score["candidate"]!=model or
        score["fixed_iterations"]!=58 or
        score["evaluation_full_windows"]!=prov["val_windows"] or
        score["evaluation_disjoint_windows"]!=prov["nonoverlap_validation_windows"] or
        score["model_feature_count"]!=(42 if model.startswith("target_") else 444)):
        raise ValueError("E6 result violates protocol")
    folder=root/model
    fitted=folder/"model.txt"
    preds=folder/"predictions_PRIVATE.npz"
    if sha(fitted)!=score["model_sha256"] or sha(preds)!=score["predictions_sha256"]:
        raise ValueError("E6 actual fitted booster/probability hashes inconsistent")
    with np.load(preds,allow_pickle=False) as f:
        y=f["y_true"].astype(np.int64)
        probabilities=f["probabilities"].astype(np.float64)
        saved=f["y_pred"].astype(np.int64)
        nonoverlap=f["nonoverlap_validation_indices"].astype(np.int64)
    if (len(y)!=prov["val_windows"] or
        probabilities.shape!=(len(y),10) or
        not np.isfinite(probabilities).all() or
        not np.array_equal(probabilities.argmax(1),saved) or
        not np.array_equal(nonoverlap,np.arange(0,len(y),5))):
        raise ValueError("Saved E6 real predictions/nonoverlap target indexing changed")
    cm=confusion_matrix(y,saved,labels=np.arange(10))
    mac=float(f1_score(y,saved,average="macro",labels=np.arange(10),zero_division=0))
    acc=float(np.trace(cm)/len(y))
    macsub=float(f1_score(y[nonoverlap],saved[nonoverlap],average="macro",
                           labels=np.arange(10),zero_division=0))
    accsub=float(np.mean(y[nonoverlap]==saved[nonoverlap]))
    if (abs(score["accuracy"]-acc)>1e-12 or
        abs(score["macro_f1"]-mac)>1e-12 or
        abs(score["nonoverlap"]["accuracy"]-accsub)>1e-12 or
        abs(score["nonoverlap"]["macro_f1"]-macsub)>1e-12):
        raise ValueError("Independent E6 metrics not reproduced from true predictions")
    with (folder/"confusion.csv").open(newline="") as f:
        rows=list(csv.reader(f))
    if (len(rows)!=11 or rows[0][1:]!=list(NAMES) or
        [row[0] for row in rows[1:]]!=list(NAMES) or
        not np.array_equal(cm,[[int(c) for c in row[1:]] for row in rows[1:]])):
        raise ValueError("E6 class confusion differs from original predictions")
    public=[root/name for name in PUBLIC_META]+[
        folder/name for name in PUBLIC_MODEL]
    required=public+[root/name for name in PRIVATE_EXTRA]+[fitted,preds]
    for file in required:
        if not file.is_file() or file.is_symlink():
            raise ValueError("E6 missing required original research artifact: "+str(file))
    return public,required

def session_for(token):
    if not token:raise ValueError("Kaggle pushRare secret required to archive E6 privately")
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

def tag(model):
    if model not in FAMILIES:raise ValueError("Unknown E6 model")
    return f"e6-order-robustness-001-{model.replace('_','-')}-frozen-artifacts"

def private_tar_name(model):
    return f"{EXPERIMENT}_{model}_FULL_PRIVATE_ARTIFACTS.tar"

def find_draft(s,model):
    allreleases=api(s,"GET","releases",params={"per_page":100})
    return next((r for r in allreleases if r["tag_name"]==tag(model)),None)

def make_tar(root,model,files):
    root=Path(root)
    target=root.parent/"e6_private_archives"/private_tar_name(model)
    target.parent.mkdir(parents=True,exist_ok=True)
    side=target.with_suffix(".tar.sha256")
    entries=[{"path":f.relative_to(root).as_posix(),
              "sha256":sha(f),"bytes":f.stat().st_size} for f in files]
    if target.exists():
        with tarfile.open(target,"r") as tar:
            existing=json.load(tar.extractfile("PRIVATE_E6_MANIFEST.json"))
        if existing["files"]!=entries or existing["candidate"]!=model:
            raise ValueError("Existing E6 private archive differs: no overwrite")
    else:
        temp=target.with_suffix(".partial")
        manifest=(json.dumps({"experiment_id":EXPERIMENT,
            "candidate":model,"official_test_accessed":False,
            "files":entries},sort_keys=True,indent=2)+"\n").encode()
        try:
            with tarfile.open(temp,"w") as tar:
                for file in tqdm(files,desc=f"Freezing E6 {model}",unit="files"):
                    info=tar.gettarinfo(str(file),arcname="run/"+file.relative_to(root).as_posix())
                    info.uid=info.gid=0;info.uname=info.gname=""
                    with file.open("rb") as h:tar.addfile(info,h)
                info=tarfile.TarInfo("PRIVATE_E6_MANIFEST.json")
                info.size=len(manifest);info.mtime=0
                tar.addfile(info,io.BytesIO(manifest))
            os.replace(temp,target)
        finally:
            temp.unlink(missing_ok=True)
    checksum=sha(target)
    if side.exists() and side.read_text().split()[0]!=checksum:
        raise ValueError("Existing E6 sidecar contradicts private archive")
    side.write_text(f"{checksum}  {target.name}\n")
    return target,side

def upload(s,file,url,model):
    class Progress:
        def __init__(self,p):
            self.f=p.open("rb")
            self.bar=tqdm(total=p.stat().st_size,unit="B",unit_scale=True,
                          desc=f"E6 private upload {model}")
        def read(self,n=-1):
            b=self.f.read(n);self.bar.update(len(b));return b
        def close(self):self.f.close();self.bar.close()
    data=Progress(file)
    try:
        r=s.post(url,params={"name":file.name},
                 headers={"Content-Type":"application/x-tar" if file.suffix==".tar" else "text/plain",
                          "Content-Length":str(file.stat().st_size)},
                 data=data,timeout=(45,7200))
        r.raise_for_status()
        result=r.json()
    finally:
        data.close()
    if (result["size"]!=file.stat().st_size or
        result.get("state")!="uploaded" or
        result.get("digest") not in (None,"sha256:"+sha(file))):
        raise ValueError("Remote E6 fitted candidate size/digest does not match")
    return result

def publish(s,root,files,commit_message):
    root=Path(root)
    trees=[]
    for file in files:
        file=Path(file)
        if (file.name.endswith("_PRIVATE.npz") or
            file.name=="model.txt" or "_PRIVATE" in file.name):
            raise ValueError("Attempt to publish PRIVATE fitted E6 evidence")
        if not file.is_file() or file.is_symlink():
            raise ValueError("Missing public aggregate evidence")
        trees.append({"path":f"results/{EXPERIMENT}/{file.relative_to(root).as_posix()}",
            "mode":"100644","type":"blob","content":file.read_text(encoding="utf-8")})
    head=api(s,"GET","git/ref/heads/main")["object"]["sha"]
    base=api(s,"GET","git/commits/"+head)["tree"]["sha"]
    after=api(s,"POST","git/trees",json={"base_tree":base,"tree":trees})["sha"]
    if after==base:return head
    commit=api(s,"POST","git/commits",json={
        "message":commit_message,"tree":after,"parents":[head]})["sha"]
    api(s,"PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    return commit

def archive_candidate(root,model,token):
    root=Path(root)
    public,allfiles=validate(root,model)
    tarp,side=make_tar(root,model,allfiles)
    with session_for(token) as s:
        rel=find_draft(s,model)
        if rel is None:
            rel=api(s,"POST","releases",json={
                "tag_name":tag(model),"target_commitish":"main",
                "draft":True,"prerelease":True,
                "name":f"E6 original read-only checkpoint {model}",
                "body":"Fitted model and original validation predictions, TRAIN only; sha256 "+sha(tarp)})
        if not rel["draft"]:
            raise ValueError("Refuse public E6 fitted model Release")
        exists={asset["name"]:asset for asset in rel.get("assets",[])}
        for f in (tarp,side):
            e=exists.get(f.name)
            if e:
                if e["size"]!=f.stat().st_size or e.get("digest")!="sha256:"+sha(f):
                    raise ValueError("Private E6 existing remote evidence mismatches")
            else:
                upload(s,f,rel["upload_url"].split("{")[0],model)
        commit=publish(s,root,public,
            f"results(e6): original {model} grouped split + window order controls")
    print("PRIVATE E6 MODEL BACKED UP:",rel["html_url"],flush=True)
    print("PUBLIC E6 AGGREGATE COMMIT:",f"https://github.com/{REPO}/commit/{commit}",flush=True)

def restore_private_candidate(root,model,token):
    root=Path(root)
    model_dir=root/model
    if (model_dir/"metrics.json").exists():
        validate(root,model)
        # If a Kaggle restart occurred between model completion and backup,
        # archive this already-completed candidate rather than retraining.
        with session_for(token) as s:
            remote=find_draft(s,model)
        if remote is None or not remote.get("assets"):
            archive_candidate(root,model,token)
        print("E6 resumed locally complete model, no retraining:",model,flush=True)
        return True
    with session_for(token) as s:
        rel=find_draft(s,model)
        if rel is None:return False
        if not rel["draft"]:
            raise ValueError("E6 stored model Release must remain draft")
        asset=next((a for a in rel.get("assets",[])
                    if a["name"]==private_tar_name(model)),None)
        if asset is None:
            raise ValueError("Existing private E6 draft incomplete; inspect before retry")
        if not asset.get("digest","").startswith("sha256:"):
            raise ValueError("Existing private E6 release has no verified archive hash")
        path=root.parent/"e6_private_archives"/asset["name"]
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or sha(path)!=asset["digest"][7:]:
            r=s.get(f"https://api.github.com/repos/{REPO}/releases/assets/{asset['id']}",
                headers={"Accept":"application/octet-stream"},
                stream=True,timeout=(45,300))
            r.raise_for_status()
            if "json" in r.headers.get("Content-Type",""):
                raise RuntimeError("GitHub returned release metadata not tar archive")
            partial=path.with_suffix(".partial")
            with partial.open("wb") as h,tqdm(
                total=asset["size"],desc=f"Restore E6 {model}",
                unit="B",unit_scale=True) as progress:
                for chunk in r.iter_content(2097152):
                    if chunk:h.write(chunk);progress.update(len(chunk))
            if partial.stat().st_size!=asset["size"] or sha(partial)!=asset["digest"][7:]:
                partial.unlink(missing_ok=True)
                raise ValueError("Restored E6 private fitted checkpoint digest mismatch")
            os.replace(partial,path)
    with tarfile.open(path,"r") as tar:
        manifest=json.load(tar.extractfile("PRIVATE_E6_MANIFEST.json"))
        if (manifest["experiment_id"]!=EXPERIMENT or
            manifest["candidate"]!=model or
            manifest["official_test_accessed"] is not False):
            raise ValueError("Wrong E6 model in previous private Release")
        for entry in manifest["files"]:
            relpath=Path(entry["path"])
            if relpath.is_absolute() or ".." in relpath.parts:
                raise ValueError("E6 archive has path traversal")
            member=tar.getmember("run/"+relpath.as_posix())
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("E6 archive includes unsafe member")
            destination=root/relpath
            if relpath.parts[0]!=model:
                if not destination.is_file() or sha(destination)!=entry["sha256"]:
                    raise ValueError("E6 destination protocol differs from private checkpoint")
                continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,destination.open("wb") as dst:
                for b in iter(lambda:src.read(4194304),b""):dst.write(b)
            if destination.stat().st_size!=entry["bytes"] or sha(destination)!=entry["sha256"]:
                raise ValueError("Restored E6 model content SHA mismatch")
    validate(root,model)
    print("RESTORED ORIGINAL E6 COMPLETED MODEL:",model,flush=True)
    return True

def publish_final_e6(root,token):
    root=Path(root)
    summary=read_json(root/"summary.json")
    if (summary["experiment_id"]!=EXPERIMENT or
        summary["official_test_opened"] is not False or
        summary["fitted_new_models"]!=5):
        raise ValueError("E6 incomplete official-test-locked final report")
    for name in FAMILIES:validate(root,name)
    public=[root/p for p in (
        "comparison_all_models.csv","paired_comparisons.csv",
        "summary.json","progress_public.json")]
    with session_for(token) as s:
        commit=publish(s,root,public,
            "results(e6): all five original grouped development models, nonoverlap control and paired findings")
    print("E6 FINAL AGGREGATES COMMITTED:",
        f"https://github.com/{REPO}/commit/{commit}",flush=True)
