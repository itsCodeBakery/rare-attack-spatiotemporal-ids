"""Integrity-preserving CARES-PARETO-003 archive and GitHub aggregate-only sync."""
import argparse,hashlib,json,os,tarfile,io,base64,csv
from pathlib import Path
import requests

REPO="itsCodeBakery/rare-attack-spatiotemporal-ids"
ID="CARES-PARETO-003-S42"
TAG="cares-pareto-003-s42-full-artifacts"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()
def verify(run):
    run=Path(run)
    s=json.loads((run/"summary.json").read_text())
    if s["experiment_id"]!=ID or s["official_test_used"] is not False:
        raise ValueError("Unexpected test exposure/experiment ID")
    variants=("frozen_rare","frozen_standard","unrestricted_switch","risk_pair_gated")
    if set(s["variants"])!=set(variants):raise ValueError("Wrong variants")
    public=[run/"summary.json",run/"summary_scores.csv",run/"protocol.json"]
    for variant in variants:
        p=run/variant
        m=json.loads((p/"metrics.json").read_text())
        if m!=s["variants"][variant] or m["n"]!=35005:
            raise ValueError("Wrong scored validation sample count")
        if sha(p/"predictions_PRIVATE.npz")!=m["private_predictions_sha256"]:
            raise ValueError("Per-record probability source checksum drift")
        with (p/"confusion.csv").open(newline="") as f:
            rows=list(csv.reader(f))
        mat=[[int(k) for k in x[1:]] for x in rows[1:]]
        if len(mat)!=10 or any(len(x)!=10 for x in mat) or sum(map(sum,mat))!=35005:
            raise ValueError("Invalid 10-class confusion matrix")
        acc=sum(mat[i][i] for i in range(10))/35005
        tp=sum(mat[i][i] for i in range(10))
        marg_r=[sum(x) for x in mat]
        marg_c=[sum(mat[i][j] for i in range(10)) for j in range(10)]
        f1=sum((2*mat[i][i]/(marg_r[i]+marg_c[i])) if marg_r[i]+marg_c[i] else 0
               for i in range(10))/10
        if abs(acc-m["accuracy"])>1e-12 or abs(f1-m["macro_f1"])>1e-12 or tp!=m["correct"]:
            raise ValueError("Independent confusion matrix arithmetic check failed")
        public.extend(p/file for file in ("metrics.json","per_class.csv","confusion.csv"))
    if any(not p.is_file() or p.is_symlink() for p in public):
        raise FileNotFoundError("Public aggregates missing/symlinked")
    return public

def preserve(run,destination,token):
    run,destination=Path(run),Path(destination)
    pub=verify(run)
    files=sorted(p for p in run.rglob("*") if p.is_file())
    manifest=[{"path":str(p.relative_to(run)),"sha256":sha(p),"bytes":p.stat().st_size}
              for p in files]
    manifest_content=(json.dumps({"experiment_id":ID,"private_artifact":True,
       "official_test_used":False,"files":manifest},indent=2)+"\n").encode()
    if not destination.exists():
        destination.parent.mkdir(exist_ok=True,parents=True)
        with tarfile.open(destination,"w") as tar:
            for p in files:
                t=tar.gettarinfo(str(p),arcname="run/"+str(p.relative_to(run)))
                t.uid=t.gid=0;t.uname=t.gname=""
                with p.open("rb") as f:tar.addfile(t,f)
            info=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
            info.size=len(manifest_content)
            tar.addfile(info,io.BytesIO(manifest_content))
    sd=sha(destination)
    side=destination.with_suffix(".tar.sha256")
    if side.exists() and side.read_text().split()[0]!=sd:
        raise ValueError("Existing archive SHA differs; refusing overwrite")
    side.write_text(f"{sd}  {destination.name}\n")
    if not token:raise RuntimeError("Authorize Kaggle pushRare before backup")
    api=f"https://api.github.com/repos/{REPO}"
    with requests.Session() as session:
        session.headers.update({"Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
        def req(method,path,**kwargs):
            r=session.request(method,api+"/"+path,timeout=180,**kwargs)
            r.raise_for_status();return r.json()
        releases=req("GET","releases?per_page=100")
        release=next((x for x in releases if x["tag_name"]==TAG),None)
        if release is None:
            release=req("POST","releases",json={"tag_name":TAG,
                "name":"CARES-PARETO-003: Frozen class-pair routing results",
                "draft":True,"prerelease":True,"target_commitish":"main",
                "body":"Research-only frozen development validation, private per-flow labels. No official test. Archive SHA256 "+sd})
        if not release["draft"]:raise ValueError("Refuse public release of per-flow predictions")
        known={x["name"]:x for x in release.get("assets",[])}
        for file,mime in ((destination,"application/x-tar"),(side,"text/plain")):
            expected="sha256:"+sha(file)
            if file.name in known:
                a=known[file.name]
                if a.get("digest")!=expected or a["size"]!=file.stat().st_size:
                    raise ValueError("Existing remote asset differs: "+file.name)
                continue
            url=release["upload_url"].split("{")[0]
            with file.open("rb") as f:
                r=session.post(url,params={"name":file.name},
                   headers={"Content-Type":mime,
                            "Content-Length":str(file.stat().st_size)},
                   data=f,timeout=(60,7200))
            r.raise_for_status();asset=r.json()
            if asset["size"]!=file.stat().st_size or asset.get("digest") not in (None,expected):
                raise ValueError("Remote private asset checksum mismatch")
        head=req("GET","git/ref/heads/main")["object"]["sha"]
        base=req("GET","git/commits/"+head)["tree"]["sha"]
        tree=[{"path":"results/"+ID+"/"+str(p.relative_to(run)),
               "mode":"100644","type":"blob","content":p.read_text()}
              for p in pub]
        root=req("POST","git/trees",json={"base_tree":base,"tree":tree})["sha"]
        if root!=base:
            commit=req("POST","git/commits",json={
              "message":"results(cares003): observed class-pair Pareto router and frozen audit",
              "tree":root,"parents":[head]})["sha"]
            req("PATCH","git/refs/heads/main",json={"sha":commit,"force":False})
        else:commit=head
    print("CARES-PARETO-003 PRIVATE DRAFT VERIFIED:",release["html_url"],flush=True)
    print("CARES-PARETO-003 PUBLIC AGGREGATES:",f"https://github.com/{REPO}/commit/{commit}",flush=True)
    print("ARCHIVE SHA256:",sd,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--archive",required=True)
    args=p.parse_args()
    preserve(args.run,args.archive,os.environ.get("GITHUB_TOKEN"))
