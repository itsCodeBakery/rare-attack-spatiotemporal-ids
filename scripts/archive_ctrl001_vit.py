"""Complete private CTRL-VIT-001 artifact archive + GitHub draft Release.

No raw dataset or secrets are archived. Integrity checked after upload.
Never silently replace an existing archive with a different checksum.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime,timezone
import requests
from tqdm.auto import tqdm

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
TAG="ctrl-vit-001-s42-full-artifacts"
ROOT="CTRL-VIT-001-S42"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for blob in iter(lambda:f.read(4194304),b""):h.update(blob)
    return h.hexdigest()

def create_archive(run,weights,path):
    run,weights,path=Path(run),Path(weights),Path(path)
    mandatory=[run/"status.json",run/"best_validation_predictions.npz",
               run/"artifact_manifest.json",weights/"best_val_macro_f1.pt",
               weights/"last_resume.pt"]
    if any(not p.is_file() for p in mandatory):
        raise FileNotFoundError("One or more frozen model artifacts missing; do not retrain.")
    paths=sorted([p for p in run.rglob("*") if p.is_file()]+
                 [p for p in weights.rglob("*") if p.is_file()])
    if any(p.is_symlink() for p in paths):raise ValueError("Archive symlinks forbidden")
    files=[]
    for p in tqdm(paths,desc="Hash private model artifacts",unit="files"):
        arc=("run/"+str(p.relative_to(run)) if p.is_relative_to(run)
             else "checkpoints/"+str(p.relative_to(weights)))
        files.append({"path":arc,"bytes":p.stat().st_size,"sha256":sha(p)})
    catalogue={"experiment_id":ROOT,"private_artifact":True,
       "test_partition_used":False,
       "created_utc":datetime.now(timezone.utc).isoformat(),
       "files":files}
    manifest=(json.dumps(catalogue,indent=2,sort_keys=True)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".partial")
    try:
        with tarfile.open(temp,"w") as tar:
            for p,item in tqdm(list(zip(paths,files)),desc="Package all checkpoints",unit="files"):
                info=tar.gettarinfo(str(p),arcname=item["path"])
                info.uid=0;info.gid=0;info.uname="";info.gname=""
                with p.open("rb") as fh:tar.addfile(info,fh)
            m=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
            m.size=len(manifest);m.mtime=0
            tar.addfile(m,io.BytesIO(manifest))
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
    digest=sha(path)
    sidecar=path.with_suffix(".tar.sha256")
    sidecar.write_text(f"{digest}  {path.name}\n",encoding="utf-8")
    print("CONTROL VIT FULL ARCHIVE CREATED:",path,
          "SHA-256:",digest,"MiB:",round(path.stat().st_size/1048576,2))
    return path,sidecar,digest

class Upload:
    def __init__(self,path):
        self.path=Path(path);self.f=self.path.open("rb")
        self.bar=tqdm(total=self.path.stat().st_size,
                      desc="Upload "+self.path.name,unit="B",unit_scale=True)
    def read(self,n=-1):
        b=self.f.read(n);self.bar.update(len(b));return b
    def close(self):
        self.f.close();self.bar.close()

def publish_draft(archive,sidecar,digest,token):
    if not token:
        print("Archive exists locally but not uploaded; Kaggle pushRare unavailable")
        return None
    api=f"https://api.github.com/repos/{REPO}"
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
                                "Accept":"application/vnd.github+json",
                                "X-GitHub-Api-Version":"2022-11-28"})
        r=session.get(api+"/releases",params={"per_page":100},timeout=60)
        r.raise_for_status()
        release=next((x for x in r.json() if x["tag_name"]==TAG),None)
        if release is None:
            r=session.post(api+"/releases",json={
                "tag_name":TAG,"target_commitish":"main",
                "name":"CTRL-VIT-001-S42 Full Reproducibility Archive",
                "draft":True,"prerelease":True,
                "body":"Frozen standalone ViT control evidence on the original SCV train-derived validation split. No official-test inference. SHA-256: "+digest
            },timeout=60)
            r.raise_for_status();release=r.json()
        if not release["draft"]:
            raise RuntimeError("Refuse adding private archive to published Release")
        r=session.get(api+f"/releases/{release['id']}",timeout=60)
        r.raise_for_status();release=r.json()
        assets={a["name"]:a for a in release.get("assets",[])}
        url=release["upload_url"].split("{")[0]
        uploaded=[]
        for item,kind in ((archive,"application/x-tar"),(sidecar,"text/plain")):
            old=assets.get(item.name)
            if old:
                if old.get("digest")==f"sha256:{sha(item)}" and old["size"]==item.stat().st_size:
                    uploaded.append(old)
                    continue
                raise ValueError("Different existing release asset; will not overwrite "+item.name)
            stream=Upload(item)
            try:
                r=session.post(url,params={"name":item.name},
                        headers={"Content-Type":kind,
                                 "Content-Length":str(item.stat().st_size)},
                        data=stream,timeout=(60,7200))
                r.raise_for_status()
                asset=r.json()
            finally:
                stream.close()
            if asset["state"]!="uploaded" or asset["size"]!=item.stat().st_size:
                raise ValueError("Release file incomplete: "+item.name)
            if asset.get("digest") and asset["digest"]!="sha256:"+sha(item):
                raise ValueError("GitHub uploaded asset SHA-256 mismatch: "+item.name)
            uploaded.append(asset)
    print("CTRL-VIT PRIVATE RELEASE VERIFIED:",release["html_url"])
    print("ARCHIVE SHA-256:",digest)
    print("DRAFT: TRUE; REVIEW CONTENTS BEFORE ANY PUBLICATION")
    return {"url":release["html_url"],"archive_sha256":digest,
            "assets":[{"name":a["name"],"size":a["size"],"digest":a.get("digest")}
                      for a in uploaded]}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--checkpoints",required=True)
    p.add_argument("--archive",required=True)
    a=p.parse_args()
    archive,sidecar,checksum=create_archive(a.run,a.checkpoints,a.archive)
    published=publish_draft(archive,sidecar,checksum,os.environ.get("GITHUB_TOKEN"))
    if published:
        print("CONTROL PRIVATE ARCHIVE UPLOAD COMPLETE")
