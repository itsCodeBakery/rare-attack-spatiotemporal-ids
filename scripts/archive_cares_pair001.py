"""Freeze first CARES-PAIR-001 measured six-ablation experiment.

No training. Hash-check every source model, per-example prediction and ten-class
confusion matrix, then package all local outputs into a complete TAR and upload
to an unpublished GitHub draft Release. Never use a designated test split.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime,timezone
import archive_ctrl003_tabular as upload
from tqdm.auto import tqdm

ID="CARES-PAIR-001-S42"
CANDIDATES=("baseline","dos_only","backdoor_only","analysis_only",
            "all_unrestricted","all_selective")
SPECIALISTS=("dos","backdoor","analysis")
upload.TAG="cares-pair-001-s42-full-artifacts"
upload.ROOT=ID

def sha(path):
    return upload.sha(path)

def validated_outputs(root):
    root=Path(root)
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    lineage=json.loads((root/"data_lineage.json").read_text(encoding="utf-8"))
    cfg=json.loads((root/"config_effective.json").read_text(encoding="utf-8"))
    if (summary["experiment_id"]!=ID or summary.get("official_test_used") is not False or
        lineage.get("test_dataset_accessed") is not False or
        lineage.get("validation_target_sequence_matches_original") is not True or
        lineage.get("validation_target_rows")!=35005 or
        cfg.get("official_test")!="FORBIDDEN"):
        raise ValueError("Original training-derived validation or forbidden-test protocol drift")
    if set(summary["variants"])!=set(CANDIDATES):
        raise ValueError("All six experimental ablations must be present")
    if set(summary["specialists"])!={"DoS","Backdoor","Analysis"}:
        raise ValueError("Exactly three originally declared specialists required")
    if abs(summary["parent_unchanged_macro_f1"]-0.6366908581786823)>1e-10:
        raise ValueError("Frozen original baseline score changed")
    classes=lineage["class_names"]
    if len(classes)!=10 or len(set(classes))!=10:
        raise ValueError("Ten class names not preserved")
    import csv
    for name in CANDIDATES:
        directory=root/name
        info=json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        if info!=summary["variants"][name]:
            raise ValueError("Experiment summary and frozen variant disagree: "+name)
        if info["n"]!=35005 or info["predictions_sha256"]!=sha(directory/"predictions_PRIVATE.npz"):
            raise ValueError("Private inference outputs mismatch: "+name)
        with (directory/"confusion.csv").open(newline="") as f:
            rows=list(csv.reader(f))
        if len(rows)!=11 or rows[0][1:]!=classes or any(row[0]!=classes[i] for i,row in enumerate(rows[1:])):
            raise ValueError("Wrong ten-class confusion structure: "+name)
        cm=[[int(v) for v in row[1:]] for row in rows[1:]]
        support=[sum(row) for row in cm]
        preds=[sum(cm[i][j] for i in range(10)) for j in range(10)]
        n=sum(support); correct=sum(cm[i][i] for i in range(10))
        macro=sum(2*cm[i][i]/(support[i]+preds[i])
                  if (support[i]+preds[i]) else 0 for i in range(10))/10
        if n!=35005 or correct!=info["correct"] or abs(macro-info["macro_f1"])>1e-12 or abs(correct/n-info["accuracy"])>1e-12:
            raise ValueError("Confusion matrix does not support claimed macro-F1: "+name)
    for pair in SPECIALISTS:
        folder=root/("specialist_"+pair)
        info=json.loads((folder/"specialist_info.json").read_text(encoding="utf-8"))
        if info["checkpoint_sha256"]!=sha(folder/"specialist.txt") or info["private_conditional_sha256"]!=sha(folder/"validation_specialist_PRIVATE.npz"):
            raise ValueError("Trained specialist SHA-256 mismatch: "+pair)
    return summary

def create_archive(root,destination):
    root=Path(root);dest=Path(destination)
    validated_outputs(root)
    if dest.is_relative_to(root):raise ValueError("Archive must live outside experiment")
    files=sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):raise ValueError("Archive symlink not permitted")
    assets=[{"path":"run/"+str(path.relative_to(root)),
             "bytes":path.stat().st_size,"sha256":sha(path)}
            for path in tqdm(files,desc="Hash frozen CARES experiment",unit="files")]
    catalog=json.dumps({"experiment_id":ID,"private_artifact":True,
         "official_test_used":False,
         "created_utc":datetime.now(timezone.utc).isoformat(),
         "files":assets},indent=2,sort_keys=True).encode()+b"\n"
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_suffix(".partial")
    with tarfile.open(partial,"w") as tar:
        for source,item in tqdm(list(zip(files,assets)),desc="Archive all CARES specialists",unit="files"):
            ti=tar.gettarinfo(str(source),arcname=item["path"])
            ti.uid=0;ti.gid=0;ti.uname="";ti.gname=""
            with source.open("rb") as fh:tar.addfile(ti,fh)
        ti=tarfile.TarInfo("PRIVATE_ARTIFACT_CATALOGUE.json")
        ti.size=len(catalog);ti.mtime=0
        tar.addfile(ti,io.BytesIO(catalog))
    os.replace(partial,dest)
    digest=sha(dest)
    sidecar=dest.with_suffix(".tar.sha256")
    sidecar.write_text(digest+"  "+dest.name+"\n",encoding="utf-8")
    print("CARES PRIVATE ARCHIVE CREATED:",dest,
          "SHA-256:",digest,"MiB:",round(dest.stat().st_size/1048576,2))
    return dest,sidecar,digest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True)
    p.add_argument("--archive",required=True)
    args=p.parse_args()
    dest=Path(args.archive)
    if dest.is_file():
        side=dest.with_suffix(".tar.sha256")
        if not side.is_file() or side.read_text().split()[0]!=sha(dest):
            raise ValueError("Existing full archive hash/sidecar mismatch; refuse overwrite")
        validated_outputs(args.run)
        arc,digest=dest,sha(dest)
        print("REUSING VERIFIED CARES ARCHIVE:",dest)
    else:
        arc,side,digest=create_archive(args.run,dest)
    published=upload.publish_draft(arc,side,digest,os.environ.get("GITHUB_TOKEN"))
    if published is None:
        raise RuntimeError("Full frozen archive is local but not uploaded: check pushRare")
    print("CARES PRIVATE RELEASE VERIFIED:",published["url"])
    print("CARES PRIVATE ARCHIVE UPLOAD COMPLETE")
