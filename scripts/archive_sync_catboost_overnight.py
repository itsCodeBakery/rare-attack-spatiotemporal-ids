"""Overnight CTRL-CAT repeated-seed evidence: validation, private archive and public sync.

Called after each fully completed CatBoost seed. Does not fit any models, change
metrics, access official test data or upload per-record private predictions publicly.
"""
import argparse
import base64
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX="results/CTRL-CAT-OVERNIGHT"
SEEDS=(42,43,44,45,46)

def digest(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def json_obj(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))

def verify(run,seed):
    run=Path(run)
    summary=json_obj(run/"summary.json")
    cfg=json_obj(run/"config_effective.json")
    lineage=json_obj(run/"data_lineage.json")
    name=f"CTRL-CAT-001-S{seed}"
    if (summary["experiment_id"]!=name or
        summary["reference_validation_count"]!=35005 or
        summary.get("official_test_used") is not False or
        summary.get("NO_OFFICIAL_SUPERIORITY_CLAIM") is not True or
        cfg["official_test"]!="LOCKED" or
        cfg["catboost"]["seed"]!=seed or
        lineage.get("test_dataset_accessed") is not False or
        lineage.get("validation_target_rows")!=35005 or
        lineage.get("validation_target_sequence_matches_original") is not True):
        raise ValueError("Overnight official-test/split/seed provenance failed")
    classes=lineage["class_names"]
    declared={m["id"] for m in cfg["models"]}|{m["id"] for m in cfg["fixed_ensembles"]}
    if len(declared)!=10 or declared!=set(summary["models"]):
        raise ValueError("Expected exactly two CatBoost fits and eight fixed ablations")
    public=[run/p for p in (
        "summary.json","summary_scores.csv","figure_manifest.json",
        "config_effective.json","data_lineage.json","source_code_commit.txt")]
    for key in sorted(declared):
        folder=run/key
        score=json_obj(folder/"metrics.json")
        if score!=summary["models"][key] or score["count"]!=35005:
            raise ValueError("Summary and scored model disagree: "+key)
        if digest(folder/"predictions_PRIVATE.npz")!=score["private_probabilities_sha256"]:
            raise ValueError("Original predictions checksum mismatch: "+key)
        if key in {x["id"] for x in cfg["models"]}:
            if digest(folder/"trained_model.cbm")!=score["model_sha256"]:
                raise ValueError("CatBoost fitted model checksum mismatch: "+key)
        with (folder/"confusion.csv").open(newline="") as f:
            rows=list(csv.reader(f))
        if (len(rows)!=11 or rows[0][1:]!=classes or
            any(x[0]!=classes[j] or len(x)!=11 for j,x in enumerate(rows[1:]))):
            raise ValueError("Ten-class confusion schema mismatch: "+key)
        matrix=[[int(z) for z in row[1:]] for row in rows[1:]]
        support=[sum(row) for row in matrix]
        column=[sum(matrix[i][j] for i in range(10)) for j in range(10)]
        n=sum(support);correct=sum(matrix[j][j] for j in range(10))
        f1=sum(2*matrix[j][j]/(support[j]+column[j])
               if support[j]+column[j] else 0 for j in range(10))/10
        if (n!=35005 or correct!=score["correct"] or
            abs(correct/n-score["accuracy"])>1e-12 or
            abs(f1-score["macro_f1"])>1e-12):
            raise ValueError("Recomputed confusion metrics mismatch: "+key)
        public.extend(folder/p for p in ("metrics.json","confusion.csv","per_class.csv"))
    fig=json_obj(run/"figure_manifest.json")
    if fig["experiment_id"]!=name or fig["count"]!=15 or len(fig["exports"])!=15:
        raise ValueError("Five figures × three formats required")
    for item in fig["exports"]:
        src=run/item["path"]
        if not src.is_file() or digest(src)!=item["sha256"] or src.stat().st_size!=item["bytes"]:
            raise ValueError("Plot checksum failed: "+str(src))
        if item["source"]!="summary_scores.csv":
            raise ValueError("Unknown plot source")
        public.append(src)
    if len(set(map(str,public)))!=len(public):
        raise ValueError("Duplicate public results file")
    if any(not p.is_file() or p.is_symlink() for p in public):
        raise FileNotFoundError("Public metric/figure is missing or symlinked")
    if any(p.name in ("predictions_PRIVATE.npz","trained_model.cbm") for p in public):
        raise ValueError("Private model data incorrectly included in GitHub public files")
    print(f"VALIDATED SEED {seed}: 10 actual confusion matrices and 15 figures")
    return public

def archive_private(run,seed,token,root):
    run=Path(run);root=Path(root)
    verify(run,seed)
    folder=root/"archives"
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/f"CTRL-CAT-001-S{seed}_FULL_ARTIFACTS.tar"
    side=target.with_suffix(".tar.sha256")
    if target.is_file():
        if not side.is_file() or side.read_text().split()[0]!=digest(target):
            raise ValueError("Existing local archive SHA mismatch; no overwrite")
        sha=digest(target)
    else:
        files=sorted(p for p in run.rglob("*") if p.is_file())
        if any(p.is_symlink() for p in files):
            raise ValueError("Symlinks forbidden in scientific artifacts")
        records=[{"path":"run/"+str(p.relative_to(run)),
                 "bytes":p.stat().st_size,"sha256":digest(p)} for p in
                 tqdm(files,desc=f"Seed {seed} verify full private files",unit="files")]
        catalog=(json.dumps({"experiment_id":f"CTRL-CAT-001-S{seed}",
          "private_artifact":True,"official_test_used":False,
          "created_utc":datetime.now(timezone.utc).isoformat(),
          "files":records},indent=2,sort_keys=True)+"\n").encode()
        partial=target.with_suffix(".partial")
        try:
            with tarfile.open(partial,"w") as tar:
                for p,record in tqdm(list(zip(files,records)),
                     desc=f"Seed {seed} archive actual checkpoints",unit="files"):
                    ti=tar.gettarinfo(str(p),arcname=record["path"])
                    ti.uid=0;ti.gid=0;ti.uname="";ti.gname=""
                    with p.open("rb") as handle:tar.addfile(ti,handle)
                info=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
                info.size=len(catalog);info.mtime=0
                tar.addfile(info,io.BytesIO(catalog))
            os.replace(partial,target)
        finally:
            partial.unlink(missing_ok=True)
        sha=digest(target)
        side.write_text(f"{sha}  {target.name}\n")
    if not token:
        raise RuntimeError("No pushRare secret for secure draft GitHub backup")
    tag=f"ctrl-cat-001-s{seed}-overnight-full-artifacts"
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
             "Accept":"application/vnd.github+json",
             "X-GitHub-Api-Version":"2022-11-28"})
        r=session.get(api+"/releases",params={"per_page":100},timeout=60)
        r.raise_for_status()
        release=next((x for x in r.json() if x["tag_name"]==tag),None)
        if release is None:
            r=session.post(api+"/releases",json={
                "tag_name":tag,"target_commitish":"main","draft":True,
                "prerelease":True,
                "name":f"CTRL-CAT-001-S{seed}: Frozen Overnight Full Experiment",
                "body":"Private model checkpoints and raw per-example predictions. No official test. SHA-256: "+sha
            },timeout=90)
            r.raise_for_status();release=r.json()
        if not release["draft"]:
            raise ValueError("Private archive can only be stored in UNPUBLISHED draft release")
        url=release["upload_url"].split("{")[0]
        existing={a["name"]:a for a in release.get("assets",[])}
        for file,kind in ((target,"application/x-tar"),(side,"text/plain")):
            old=existing.get(file.name)
            if old:
                if old.get("digest")!="sha256:"+digest(file) or old["size"]!=file.stat().st_size:
                    raise ValueError("An existing private asset differs; refuse overwrite")
                continue
            from tqdm.auto import tqdm as progress_tqdm
            class Reader:
                def __init__(self,p):
                    self.fp=p.open("rb")
                    self.bar=progress_tqdm(total=p.stat().st_size,unit="B",
                               unit_scale=True,desc="Upload "+p.name)
                def read(self,size=-1):
                    b=self.fp.read(size);self.bar.update(len(b));return b
                def close(self):
                    self.fp.close();self.bar.close()
            stream=Reader(file)
            try:
                r=session.post(url,params={"name":file.name},
                    headers={"Content-Type":kind,
                             "Content-Length":str(file.stat().st_size)},
                    data=stream,timeout=(45,7200))
                r.raise_for_status()
                asset=r.json()
            finally:
                stream.close()
            if (asset["state"]!="uploaded" or asset["size"]!=file.stat().st_size
                 or asset.get("digest") not in (None,"sha256:"+digest(file))):
                raise ValueError("Remote draft asset integrity failed")
    print(f"SEED {seed} PRIVATE DRAFT RELEASE VERIFIED:",release["html_url"])
    print("FULL MODEL ARCHIVE SHA-256:",sha)
    return {"release":release["html_url"],"sha256":sha,"bytes":target.stat().st_size}

def sync_public(run,seed,token):
    public=verify(run,seed)
    if not token:raise RuntimeError("No pushRare secret for public aggregate sync")
    run=Path(run)
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"})
        def req(method,route,**kw):
            r=session.request(method,api+"/"+route,timeout=180,**kw)
            r.raise_for_status()
            return r.json()
        head=req("GET","git/ref/heads/main")["object"]["sha"]
        base=req("GET","git/commits/"+head)["tree"]["sha"]
        entries=[]
        for p in tqdm(public,desc=f"Seed {seed} publish aggregate-only evidence",unit="files"):
            rel=PREFIX+f"/S{seed}/"+str(p.relative_to(run)).replace(os.sep,"/")
            buf=p.read_bytes()
            if p.suffix in (".png",".pdf"):
                obj=req("POST","git/blobs",json={
                    "content":base64.b64encode(buf).decode("ascii"),
                    "encoding":"base64"})["sha"]
                entries.append({"path":rel,"mode":"100644","type":"blob","sha":obj})
            else:
                entries.append({"path":rel,"mode":"100644","type":"blob",
                                "content":buf.decode("utf-8")})
        t=req("POST","git/trees",json={"base_tree":base,"tree":entries})["sha"]
        if t==base:
            commit=head
        else:
            commit=req("POST","git/commits",json={
                "message":f"results(catboost): real S{seed} Pareto multiclass results (10 variants, no official test)",
                "tree":t,"parents":[head]})["sha"]
            req("PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    print(f"SEED {seed} AGGREGATE GITHUB VERIFIED: https://github.com/{REPO}/commit/{commit}")
    return commit

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--seed",required=True,type=int)
    parser.add_argument("--destination",required=True)
    args=parser.parse_args()
    token=os.environ.get("GITHUB_TOKEN")
    archive_private(args.run,args.seed,token,args.destination)
    sync_public(args.run,args.seed,token)
