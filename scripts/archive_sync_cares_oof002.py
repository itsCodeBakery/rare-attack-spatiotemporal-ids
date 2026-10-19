"""Private CARES-OOF-002 full-model archive and safe GitHub aggregate evidence sync.

No inference/training; hash frozen 3-fold × 2 OOF checkpoints, all validations,
real confusion tables, and publish publicly only aggregate scores.
"""
import argparse,base64,csv,hashlib,io,json,os,tarfile
from datetime import datetime,timezone
from pathlib import Path
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
EXPERIMENT="CARES-OOF-002-S42"
TAG="cares-oof-002-s42-full-artifacts"
VARIANTS=("frozen_base","simple_average","meta_full","risk_gated")

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()

def inspect(root):
    root=Path(root)
    summary=json.loads((root/"summary.json").read_text())
    lineage=json.loads((root/"data_lineage.json").read_text())
    gate=json.loads((root/"gate_selection.json").read_text())
    if (summary["experiment_id"]!=EXPERIMENT or
        summary.get("official_test_used") is not False or
        lineage.get("test_dataset_accessed") is not False or
        lineage.get("validation_target_rows")!=35005 or
        summary.get("source_validation_untouched_for_meta_training") is not True or
        gate.get("selected_without_validation_label_access") is not True or
        set(summary["variants"])!=set(VARIANTS)):
        raise ValueError("Official test or out-of-fold development protocol mismatch")
    classes=lineage["class_names"]
    if len(classes)!=10:raise ValueError("Expected ten attack classes")
    public=[root/p for p in ("summary.json","comparison_scores.csv",
        "config_effective.json","data_lineage.json","source_commit.txt",
        "gate_selection.json","gate_search_training_only.csv")]
    for name in VARIANTS:
        directory=root/name
        metrics=json.loads((directory/"metrics.json").read_text())
        if metrics!=summary["variants"][name] or metrics["n"]!=35005:
            raise ValueError("Variant summary mismatch: "+name)
        if metrics["predictions_sha256"]!=sha(directory/"predictions_PRIVATE.npz"):
            raise ValueError("Per-example prediction hash mismatch: "+name)
        with (directory/"confusion.csv").open(newline="") as f:
            rows=list(csv.reader(f))
        if len(rows)!=11 or rows[0][1:]!=classes:
            raise ValueError("Confusion matrix headings mismatch: "+name)
        cm=[[int(v) for v in row[1:]] for row in rows[1:]]
        n=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(10))
        actual=[sum(x) for x in cm]
        predicted=[sum(cm[i][j] for i in range(10)) for j in range(10)]
        macro=sum(2*cm[i][i]/(actual[i]+predicted[i])
                  if actual[i]+predicted[i]>0 else 0 for i in range(10))/10
        if (n!=35005 or correct!=metrics["correct"] or
            abs(correct/n-metrics["accuracy"])>1e-12 or
            abs(macro-metrics["macro_f1"])>1e-12):
            raise ValueError("Independent confusion metric audit failed: "+name)
        public.extend(directory/x for x in ("metrics.json","confusion.csv","per_class.csv"))
    if sha(root/"train_OOF_probabilities_PRIVATE.npz")!=summary["out_of_fold_prediction_hash"]:
        raise ValueError("OOF training records checksum drift")
    for fold in (1,2,3):
        for name in ("standard","rare"):
            for suffix in (".txt","_OOF_PRIVATE.npz"):
                f=root/f"fold_{fold}"/(name+suffix)
                if not f.is_file():raise FileNotFoundError(f)
    if not (root/"meta_model_PRIVATE.joblib").is_file():
        raise FileNotFoundError("Meta model binary missing")
    if any(not p.is_file() or p.is_symlink() for p in public):
        raise FileNotFoundError("Public aggregate file missing/symlinked")
    if any(p.name.endswith("PRIVATE.npz") or p.suffix in (".txt",".joblib") for p in public):
        raise ValueError("Private model/flow information in public GitHub selection")
    return public

def archive(root,archive_file,token):
    root=Path(root);archive_file=Path(archive_file)
    inspect(root)
    side=archive_file.with_suffix(".tar.sha256")
    if archive_file.exists():
        if not side.exists() or sha(archive_file)!=side.read_text().split()[0]:
            raise ValueError("Pre-existing archive does not match immutable sidecar")
        digest=sha(archive_file)
    else:
        files=sorted(p for p in root.rglob("*") if p.is_file())
        if any(p.is_symlink() for p in files):raise ValueError("No symlinks in private artifact")
        entries=[{"path":"run/"+str(p.relative_to(root)),"bytes":p.stat().st_size,
                  "sha256":sha(p)} for p in files]
        manifest=(json.dumps({"experiment_id":EXPERIMENT,"private_artifact":True,
            "official_test_used":False,"created_utc":datetime.now(timezone.utc).isoformat(),
            "files":entries},indent=2,sort_keys=True)+"\n").encode()
        archive_file.parent.mkdir(parents=True,exist_ok=True)
        partial=archive_file.with_suffix(".partial")
        try:
            with tarfile.open(partial,"w") as tar:
                for p,item in tqdm(zip(files,entries),total=len(entries),
                                  desc="Freeze CARES-OOF checkpoints",unit="files"):
                    m=tar.gettarinfo(str(p),arcname=item["path"])
                    m.uid=0;m.gid=0;m.uname="";m.gname=""
                    with p.open("rb") as handle:tar.addfile(m,handle)
                m=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
                m.size=len(manifest);m.mtime=0
                tar.addfile(m,io.BytesIO(manifest))
            os.replace(partial,archive_file)
        finally:
            partial.unlink(missing_ok=True)
        digest=sha(archive_file)
        side.write_text(f"{digest}  {archive_file.name}\n")
    if not token:raise RuntimeError("Authorize Kaggle pushRare for private draft backup")
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as s:
        s.headers.update({"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"})
        r=s.get(api+"/releases",params={"per_page":100},timeout=90)
        r.raise_for_status()
        rel=next((r for r in r.json() if r["tag_name"]==TAG),None)
        if rel is None:
            r=s.post(api+"/releases",json={
                "tag_name":TAG,"target_commitish":"main","draft":True,
                "prerelease":True,"name":"Frozen CARES-OOF-002 real cross-fitted experiment",
                "body":"Private fitted fold+meta checkpoints and per-row predictions. Official test untouched. SHA256 "+digest
                },timeout=90)
            r.raise_for_status();rel=r.json()
        if not rel["draft"]:raise ValueError("Cannot upload private model data to published release")
        existing={x["name"]:x for x in rel.get("assets",[])}
        url=rel["upload_url"].split("{")[0]
        for p,mime in ((archive_file,"application/x-tar"),(side,"text/plain")):
            prior=existing.get(p.name)
            if prior:
                if prior["size"]!=p.stat().st_size or prior.get("digest")!="sha256:"+sha(p):
                    raise ValueError("Existing private release asset differs: "+p.name)
                continue
            class Progress:
                def __init__(self,file):
                    self.f=file.open("rb")
                    self.bar=tqdm(total=file.stat().st_size,unit="B",unit_scale=True,
                        desc="Upload "+file.name)
                def read(self,size=-1):
                    b=self.f.read(size);self.bar.update(len(b));return b
                def close(self):
                    self.f.close();self.bar.close()
            stream=Progress(p)
            try:
                r=s.post(url,params={"name":p.name},data=stream,
                     headers={"Content-Type":mime,
                              "Content-Length":str(p.stat().st_size)},
                     timeout=(45,7200))
                r.raise_for_status()
                meta=r.json()
            finally:
                stream.close()
            if (meta["state"]!="uploaded" or meta["size"]!=p.stat().st_size or
                meta.get("digest") not in (None,"sha256:"+sha(p))):
                raise ValueError("Draft upload integrity error")
    print("CARES-OOF PRIVATE DRAFT RELEASE VERIFIED:",rel["html_url"])
    print("FULL ARCHIVE SHA-256:",digest)
    return rel["html_url"]

def sync(root,token):
    root=Path(root)
    public=inspect(root)
    if not token:raise RuntimeError("Kaggle pushRare required")
    api="https://api.github.com/repos/"+REPO
    with requests.Session() as s:
        s.headers.update({"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28"})
        def req(method,path,**kw):
            r=s.request(method,api+"/"+path,timeout=180,**kw)
            r.raise_for_status()
            return r.json()
        head=req("GET","git/ref/heads/main")["object"]["sha"]
        base=req("GET","git/commits/"+head)["tree"]["sha"]
        tree=[{"path":"results/"+EXPERIMENT+"/"+str(p.relative_to(root)),
               "mode":"100644","type":"blob","content":p.read_text(encoding="utf-8")}
              for p in public]
        changed=req("POST","git/trees",json={"base_tree":base,"tree":tree})["sha"]
        if changed==base:commit=head
        else:
            commit=req("POST","git/commits",json={
                "message":"results(cares002): actual train-only OOF risk-gated multiclass ablations",
                "tree":changed,"parents":[head]})["sha"]
            req("PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
    print("CARES-OOF VERIFIED RESULTS COMMIT: https://github.com/"+REPO+"/commit/"+commit)
    return commit

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True);ap.add_argument("--archive",required=True)
    args=ap.parse_args()
    token=os.environ.get("GITHUB_TOKEN")
    archive(args.run,args.archive,token)
    sync(args.run,token)
    print("CARES-OOF-002 REAL PROPOSED-METHOD RUN PERMANENTLY PRESERVED")
