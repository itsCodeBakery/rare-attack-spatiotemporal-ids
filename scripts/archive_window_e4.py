"""WINDOW-E4-REPL-001: independently audit each trained seed, private draft backup.

Every completed model is safeguarded before next fit. Only aggregates are
committed publicly; row-level probabilities and boosters stay in GitHub drafts.
"""
import csv
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score,confusion_matrix
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
EXPERIMENT="WINDOW-E4-REPL-001"
SEEDS=(42,43,44,45,46)
FAMILIES=("stats_rare_sqrt","block_shuffled_rare_sqrt",
          "independent_shuffled_rare_sqrt")

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def validate(root,seed,family):
    root=Path(root)
    if seed not in SEEDS or family not in FAMILIES:
        raise ValueError("Unplanned seed/method in E4 archive")
    if seed==42 and family=="stats_rare_sqrt":
        raise ValueError("Original E3 seed42 is reused, not trained/archived anew")
    cfg=load(root/"config_effective.json")
    lineage=load(root/"data_lineage.json")
    if (cfg["experiment_id"]!=EXPERIMENT or
        cfg["official_test"]!="FORBIDDEN" or
        lineage["test_dataset_accessed"] is not False or
        lineage["validation_target_rows"]!=35005 or
        lineage["total_expanded_features"]!=444):
        raise ValueError("Frozen source/test lock broken")
    prefix=root/f"S{seed}"
    score=load(prefix/family/"metrics.json")
    if (score["candidate"]!=family or score["count"]!=35005 or
        score["feature_count"]!=444 or
        score["official_test_used"] is not False):
        raise ValueError("Unrecognized E4 scoring configuration")
    model=prefix/family/"model.txt"
    preds=prefix/family/"predictions_PRIVATE.npz"
    if digest(model)!=score["checkpoint_sha256"] or digest(preds)!=score["private_predictions_sha256"]:
        raise ValueError("Private model checkpoint / raw probability SHA invalid")
    with np.load(preds,allow_pickle=False) as data:
        y=data["y_true"].astype(np.int64)
        p=data["probabilities"].astype(np.float64)
        yp=data["y_pred"].astype(np.int64)
    if (len(y)!=35005 or p.shape!=(35005,10) or
        not np.isfinite(p).all() or not np.array_equal(p.argmax(1),yp)):
        raise ValueError("Per-example frozen E4 probabilities inconsistent")
    cm=confusion_matrix(y,yp,labels=np.arange(10))
    if (abs(score["accuracy"]-float(np.trace(cm)/35005))>1e-12 or
        abs(score["macro_f1"]-float(f1_score(
            y,yp,labels=np.arange(10),average="macro",zero_division=0)))>1e-12):
        raise ValueError("Independent recomputation of E4 reported metrics failed")
    with (prefix/family/"confusion.csv").open(newline="") as f:
        rows=list(csv.reader(f))
    classes=lineage["class_names"]
    if (len(rows)!=11 or rows[0][1:]!=classes or
        not np.array_equal([[int(z) for z in row[1:]] for row in rows[1:]],cm)):
        raise ValueError("Recorded confusion differs from recomputed predictions")
    public=[root/x for x in ("config_effective.json","data_lineage.json",
                              "feature_manifest.json","source_code_commit.txt",
                              "training_environment.txt")]
    public.append(prefix/"protocol.json")
    public.extend(prefix/family/f for f in (
        "metrics.json","confusion.csv","per_class.csv","learning.csv"))
    for pth in public:
        if not pth.is_file() or pth.is_symlink():
            raise ValueError("Invalid public aggregate file: "+str(pth))
    return public

def gh(token):
    if not token:raise ValueError("Authorize Kaggle pushRare for GitHub private backups")
    s=requests.Session()
    s.headers.update({"Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28"})
    return s

def req(s,method,path,**kw):
    r=s.request(method,f"https://api.github.com/repos/{REPO}/{path}",
                timeout=(45,180),**kw)
    r.raise_for_status();return r.json()

def tag(seed,family):
    return f"window-e4-repl-001-s{seed}-{family.replace('_','-')}-full-artifacts"

def find_release(s,seed,family):
    releases=req(s,"GET","releases",params={"per_page":100})
    return next((r for r in releases if r["tag_name"]==tag(seed,family)),None)

def private_filename(seed,family):
    return f"{EXPERIMENT}_S{seed}_{family}_FULL_ARTIFACTS.tar"

def public_commit(s,root,files,message):
    root=Path(root)
    head=req(s,"GET","git/ref/heads/main")["object"]["sha"]
    base=req(s,"GET","git/commits/"+head)["tree"]["sha"]
    paths=[]
    for p in files:
        p=Path(p)
        if not p.is_file() or p.is_symlink():
            raise ValueError("Missing public evidence: "+str(p))
        if p.name in ("predictions_PRIVATE.npz","model.txt"):
            raise ValueError("Fitted model/private predictions cannot be pushed publicly")
        paths.append({"path":f"results/{EXPERIMENT}/{p.relative_to(root).as_posix()}",
                "mode":"100644","type":"blob","content":p.read_text(encoding="utf-8")})
    tree=req(s,"POST","git/trees",json={"base_tree":base,"tree":paths})["sha"]
    if tree==base:return head
    commit=req(s,"POST","git/commits",json={
        "message":message,"tree":tree,"parents":[head]})["sha"]
    req(s,"PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    return commit

def archive_and_sync(root,seed,family,token):
    root=Path(root)
    public=validate(root,seed,family)
    folder=root/f"S{seed}"/family
    source=root/"source_code_commit.txt"
    folder.joinpath("training_source_commit.txt").write_text(source.read_text())
    wanted=public+[folder/"model.txt",folder/"predictions_PRIVATE.npz",
                   folder/"training_source_commit.txt"]
    manifest=[{"relative_path":p.relative_to(root).as_posix(),
               "sha256":digest(p),"bytes":p.stat().st_size} for p in wanted]
    name=private_filename(seed,family)
    path=root.parent/"e4_archives"/name
    path.parent.mkdir(parents=True,exist_ok=True)
    side=path.with_suffix(".tar.sha256")
    if path.is_file():
        if not side.is_file() or side.read_text().split()[0]!=digest(path):
            raise ValueError("Existing private archive immutable hash mismatch")
    else:
        record=(json.dumps({"experiment_id":EXPERIMENT,"seed":seed,
              "family":family,"official_test_used":False,
              "files":manifest},sort_keys=True,indent=2)+"\n").encode()
        partial=path.with_suffix(".partial")
        try:
            with tarfile.open(partial,"w") as tar:
                for file in tqdm(wanted,desc=f"Freeze E4 seed{seed} {family}",
                                  unit="files"):
                    m=tar.gettarinfo(str(file),
                        arcname="run/"+file.relative_to(root).as_posix())
                    m.uid=m.gid=0;m.uname=m.gname=""
                    with file.open("rb") as src:tar.addfile(m,src)
                m=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
                m.size=len(record)
                tar.addfile(m,io.BytesIO(record))
            os.replace(partial,path)
        finally:
            partial.unlink(missing_ok=True)
        side.write_text(f"{digest(path)}  {path.name}\n")
    with gh(token) as session:
        release=find_release(session,seed,family)
        if release is None:
            release=req(session,"POST","releases",json={
                "tag_name":tag(seed,family),
                "name":f"Frozen E4 seed {seed} {family}",
                "target_commitish":"main",
                "draft":True,"prerelease":True,
                "body":"PRIVATE model and flow predictions, official test untouched, sha256 "+digest(path)})
        if not release["draft"]:
            raise ValueError("Refusing to publish raw per-example data")
        present={asset["name"]:asset for asset in release.get("assets",[])}
        for item,mime in ((path,"application/x-tar"),(side,"text/plain")):
            expected="sha256:"+digest(item)
            if item.name in present:
                remote=present[item.name]
                if remote["size"]!=item.stat().st_size or remote.get("digest")!=expected:
                    raise ValueError("Existing remote draft archive has changed")
                continue
            class Progress:
                def __init__(self,p):
                    self.f=p.open("rb")
                    self.bar=tqdm(desc="Upload E4 "+family,total=p.stat().st_size,
                                   unit="B",unit_scale=True)
                def read(self,n=-1):
                    b=self.f.read(n);self.bar.update(len(b));return b
                def close(self):
                    self.f.close();self.bar.close()
            stream=Progress(item)
            try:
                r=session.post(release["upload_url"].split("{")[0],
                    params={"name":item.name},
                    headers={"Content-Length":str(item.stat().st_size),
                             "Content-Type":mime},
                    data=stream,timeout=(45,7200))
                r.raise_for_status()
                meta=r.json()
            finally:stream.close()
            if meta["size"]!=item.stat().st_size or meta.get("digest") not in (None,expected):
                raise ValueError("GitHub archive size/hash mismatch")
        link=public_commit(session,root,public,
                f"results(e4): true S{seed} {family} experiment metrics")
    print("E4 PRIVATE DRAFT:",release["html_url"],flush=True)
    print("E4 PUBLIC COMMIT:",f"https://github.com/{REPO}/commit/{link}",flush=True)
    print("E4 ARCHIVE SHA256:",digest(path),flush=True)

def restore_prior_candidate(root,seed,family,token):
    root=Path(root)
    if seed==42 and family=="stats_rare_sqrt":
        return False
    f=root/f"S{seed}"/family/"model.txt"
    if f.is_file():
        validate(root,seed,family)
        print("REUSING EXISTING VERIFIED E4 MODEL:",seed,family,flush=True)
        return True
    with gh(token) as session:
        release=find_release(session,seed,family)
        if release is None:return False
        if not release["draft"]:
            raise ValueError("E4 model release must remain unpublished")
        asset=next((x for x in release.get("assets",[])
                    if x["name"]==private_filename(seed,family)),None)
        if asset is None:raise FileNotFoundError("E4 draft lacks expected private model")
        expected=asset.get("digest","")
        if not expected.startswith("sha256:"):
            raise ValueError("Remote E4 draft missing SHA256")
        destination=root.parent/"e4_archives"/asset["name"]
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.is_file() or digest(destination)!=expected[7:]:
            response=session.get(
                f"https://api.github.com/repos/{REPO}/releases/assets/{asset['id']}",
                headers={"Accept":"application/octet-stream"},
                stream=True,timeout=(45,300))
            response.raise_for_status()
            if "json" in response.headers.get("Content-Type",""):
                raise RuntimeError("GitHub returned release metadata rather than archive bytes")
            temp=destination.with_suffix(".download")
            with temp.open("wb") as output,tqdm(total=asset["size"],desc="Restore E4 "+family,
                                                unit="B",unit_scale=True) as bar:
                for b in response.iter_content(2097152):
                    if b:output.write(b);bar.update(len(b))
            if temp.stat().st_size!=asset["size"] or digest(temp)!=expected[7:]:
                temp.unlink(missing_ok=True)
                raise ValueError("E4 restored archive SHA256 mismatch")
            temp.replace(destination)
    with tarfile.open(destination,"r") as tar:
        manifest=json.load(tar.extractfile("PRIVATE_ARTIFACT_CATALOGUE.json"))
        if (manifest["experiment_id"]!=EXPERIMENT or
            manifest["family"]!=family or manifest["seed"]!=seed or
            manifest["official_test_used"] is not False):
            raise ValueError("E4 resumed wrong fitted model")
        for file in manifest["files"]:
            p=Path(file["relative_path"])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Private E4 archive path traversal")
            target=root/p
            member=tar.getmember("run/"+p.as_posix())
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Invalid tar member type")
            # Do not overwrite top-level provenance for other model runs.
            if p.parts[0]!="S"+str(seed):
                if not target.is_file():raise ValueError("Missing prerequisite root file")
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open("wb") as dst:
                for b in iter(lambda:src.read(4194304),b""):dst.write(b)
            if target.stat().st_size!=file["bytes"] or digest(target)!=file["sha256"]:
                raise ValueError("Restored fitted E4 file checksum mismatch")
    validate(root,seed,family)
    print("E4 RESTORED SHA256-VERIFIED COMPLETED RUN:",seed,family,flush=True)
    return True

def sync_progress(root,seed,records,token):
    root=Path(root)
    results=pd.DataFrame(records).sort_values(["family","seed"])
    results.to_csv(root/"records_so_far.csv",index=False)
    with gh(token) as session:
        commit=public_commit(session,root,[
             root/"records_so_far.csv",root/f"S{seed}"/"seed_scores.csv"],
             f"results(e4): finished 3 matched methods for seed {seed}")
    print("E4 PER-SEED BACKUP:",f"https://github.com/{REPO}/commit/{commit}",flush=True)

def sync_final(root,token):
    root=Path(root)
    summary=load(root/"summary.json")
    if (summary["experiment_id"]!=EXPERIMENT or
        summary["official_test_used"] is not False or
        summary["n_seeds"]!=5 or summary["n_fitted_models_in_E4"]!=14):
        raise ValueError("Incomplete E4 final research summary")
    for seed in SEEDS:
        for fam in FAMILIES:
            if seed==42 and fam=="stats_rare_sqrt":continue
            validate(root,seed,fam)
    public=[root/p for p in (
        "summary.json","all_seed_scores.csv","five_seed_summary.csv",
        "paired_differences.csv","paired_difference_details.json",
        "records_so_far.csv")]
    with gh(token) as session:
        commit=public_commit(session,root,public,
                             "results(e4): verified five-seed replication and paired negative controls")
    print("E4 FINAL GITHUB RESULTS:",f"https://github.com/{REPO}/commit/{commit}",flush=True)
