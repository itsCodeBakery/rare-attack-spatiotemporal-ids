"""E7 safe, immutable GitHub aggregate auditor publisher, version 2.

Fixes the real 2026-10-19 case: GitHub PUT 201 succeeded while immediate GET
returned HTTP 404. Do NOT retry PUT blindly. Save its commit/blob receipt, then
verify by immutable blob SHA/commit-ref, with bounded exponential retries.

This module cannot publish arbitrary payloads: G0 reports are checked by original
strict validator; extended reports have their own typed, aggregate-only schema.
"""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "itsCodeBakery/rare-attack-spatiotemporal-ids"
PREFIX = "research/E7_CICIDS2017/results"
SRC = Path(__file__).with_name("e7_g0_publish.py")
_spec = importlib.util.spec_from_file_location("e7_publish_original", SRC)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
RETRY_SECS = (0, 1, 2, 3, 5, 8, 13)

def api(method, url, token, payload=None):
    if not token: raise ValueError("GitHub token missing")
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    if data is not None: headers["Content-Type"] = "application/json"
    with urlopen(Request(url, data=data, headers=headers, method=method), timeout=65) as response:
        return json.load(response)

def encode_safe(report, stage):
    if stage == "G0":
        _original.verify_report(report)
    elif stage == "G0_EXTENDED":
        expected = {"schema","status","source_file","source_sha256","rows",
                    "labels","timestamp_quality","duration_quality","feature_quality",
                    "duplicates","notes"}
        if (set(report) != expected or report["schema"] != "E7_G0_EXTENDED_AGGREGATE_001"
                or report["status"] != "REVIEW_REQUIRED"
                or not report["source_file"].endswith(".csv")
                or len(report["source_sha256"]) != 64
                or not isinstance(report["rows"],int)
                or report["rows"] < 1):
            raise ValueError("Unsupported E7 extended public report")
        safe_names = {"label_counts","raw_hour_histogram","parsed_weekdays","parse_failures",
                      "row_order_reversals","timestamp_min","timestamp_max","clock_ambiguous",
                      "duration_unit","duration_zero","duration_negative","duration_missing",
                      "duration_nonfinite","duration_max_raw","duration_min_raw",
                      "numeric_feature_count","columns_with_missing","columns_with_nonfinite",
                      "columns_with_negative","columns_with_zero","constant_numeric_columns",
                      "missing_endpoint_rows","candidate_duplicate_rows",
                      "candidate_label_conflict_rows","hash_scheme","disclaimer"}
        nested_expected = {
            "timestamp_quality":{"raw_hour_histogram","parsed_weekdays","parse_failures",
                                 "row_order_reversals","timestamp_min","timestamp_max",
                                 "clock_ambiguous"},
            "duration_quality":{"duration_unit","duration_zero","duration_negative",
                                "duration_missing","duration_nonfinite",
                                "duration_max_raw","duration_min_raw"},
            "feature_quality":{"numeric_feature_count","columns_with_missing",
                               "columns_with_nonfinite","columns_with_negative",
                               "columns_with_zero","constant_numeric_columns",
                               "missing_endpoint_rows"},
            "duplicates":{"candidate_duplicate_rows","candidate_label_conflict_rows",
                          "hash_scheme","disclaimer"}}
        if (set(report["timestamp_quality"]) != nested_expected["timestamp_quality"] or
            set(report["duration_quality"]) != nested_expected["duration_quality"] or
            set(report["feature_quality"]) != nested_expected["feature_quality"] or
            set(report["duplicates"]) != nested_expected["duplicates"]):
            raise ValueError("Unexpected source detail; review before upload")
        if not isinstance(report["notes"],list) or not all(isinstance(x,str) for x in report["notes"]):
            raise ValueError("Notes must be strings")
        if not isinstance(report["labels"],dict) or not all(isinstance(n,int) for n in report["labels"].values()):
            raise ValueError("Labels must be aggregated counts")
        if not isinstance(report["feature_quality"]["columns_with_missing"],dict):
            raise ValueError("Invalid feature quality counts")
        if not isinstance(report["feature_quality"]["columns_with_nonfinite"],dict):
            raise ValueError("Invalid feature quality counts")
        if not isinstance(report["feature_quality"]["columns_with_negative"],dict):
            raise ValueError("Invalid feature quality counts")
        if not isinstance(report["feature_quality"]["columns_with_zero"],dict):
            raise ValueError("Invalid feature quality counts")
        if not isinstance(report["timestamp_quality"]["raw_hour_histogram"],dict):
            raise ValueError("Invalid coarse source hour summary")
        # No raw identifying values or flow hashes can be stored in this schema.
        for bad in ("ip_address","source_ip","destination_ip","flow_id",
                    "per_row","ip_list","raw_flows","feature_rows","token"):
            if bad in json.dumps(report).lower():
                raise ValueError("Potentially sensitive flow-level content")
    else: raise ValueError("Only G0/G0_EXTENDED are authorized")
    return (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()

def atomic_json(path, object_):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".writing")
    tmp.write_text(json.dumps(object_, indent=2, sort_keys=True) + "\n",encoding="utf-8")
    tmp.replace(dest)

def from_content_api(response):
    return base64.b64decode(response["content"])

def check_remote(url, token, content, *, blob=None, commit=None):
    # Git's immutable SHA endpoint is often visible before the default-branch
    # contents index becomes consistent after a successful PUT.
    candidates=[]
    if blob:
        candidates.append(("blob",f"https://api.github.com/repos/{REPO}/git/blobs/{blob}"))
    if commit:
        candidates.append(("commit",url+"?ref="+quote(commit,safe="")))
    candidates.append(("current",url))
    for kind, source in candidates:
        try:
            result=api("GET",source,token)
        except HTTPError as exc:
            if exc.code in (404,409,429,500,502,503,504):
                continue
            raise
        if kind=="blob":
            encoded=result.get("content","")
            remote=base64.b64decode(encoded) if result.get("encoding")=="base64" else encoded.encode()
        else: remote=from_content_api(result)
        if remote != content:
            raise RuntimeError("Remote evidence differs from original local source: refusing overwrite")
        return {"via":kind, "blob_sha":result.get("sha",blob)}
    return None

def publish(local_file, *, token, stage, receipts_dir, pause=time.sleep):
    path=Path(local_file)
    record=json.loads(path.read_text(encoding="utf-8"))
    data=encode_safe(record,stage)
    basename=record["external_source_basename"] if stage=="G0" else record["source_file"]
    if ("/" in basename or "\\" in basename or ".." in basename or
            not basename.endswith(".csv")):
        raise ValueError("Unsafe research artifact basename")
    tail=basename+".audit.json" if stage=="G0" else basename+".quality.json"
    remote=f"{PREFIX}/{stage}/{tail}"
    url=f"https://api.github.com/repos/{REPO}/contents/"+quote(remote,safe="/")
    digest=hashlib.sha256(data).hexdigest()
    receipt=Path(receipts_dir)/(stage+"__"+tail+".receipt.json")

    if receipt.exists():
        state=json.loads(receipt.read_text(encoding="utf-8"))
        if state.get("source_sha256") != digest or state.get("remote") != remote:
            raise RuntimeError("Existing upload receipt contradicts local source; no overwrite")
        blob,commit=state.get("blob_sha"),state.get("commit_sha")
    else:
        blob=commit=None

    for sec in RETRY_SECS:
        if sec: pause(sec)
        verified=check_remote(url,token,data,blob=blob,commit=commit)
        if verified:
            atomic_json(receipt,{"remote":remote,"source_sha256":digest,
                      "commit_sha":commit,"blob_sha":blob or verified["blob_sha"],
                      "verified_via":verified["via"],"status":"VERIFIED"})
            return {"status":"VERIFIED","remote":remote,"sha256":digest,
                    "verified_via":verified["via"]}
        if receipt.exists():
            continue
        break

    if receipt.exists():
        raise RuntimeError("Original GitHub PUT may have succeeded: receipt exists but remote unreadable. Stop; never resend automatically.")

    # Exactly one create-only PUT; NEVER put with old SHA (no overwrite).
    try:
        uploaded=api("PUT",url,token,{
            "message":f"research(e7-{stage.lower()}): immutable original audit {basename}",
            "branch":"main","content":base64.b64encode(data).decode("ascii")})
    except HTTPError as exc:
        if exc.code in (409,422):
            # GitHub may have accepted another process's write. Only read/verify.
            for sec in RETRY_SECS:
                if sec: pause(sec)
                found=check_remote(url,token,data)
                if found:
                    atomic_json(receipt,{"remote":remote,"source_sha256":digest,
                                "blob_sha":found["blob_sha"],"commit_sha":None,
                                "verified_via":found["via"],"status":"VERIFIED"})
                    return {"status":"VERIFIED_CONCURRENT","remote":remote,"sha256":digest}
            raise RuntimeError("GitHub reports existing/conflicted content; never overwrite") from exc
        raise

    blob=uploaded.get("content",{}).get("sha")
    commit=uploaded.get("commit",{}).get("sha")
    if not blob or not commit:
        raise RuntimeError("GitHub responded to PUT without a content SHA/commit; manual recovery required")
    # CRITICAL: durable receipt immediately AFTER successful PUT, BEFORE slow verification.
    atomic_json(receipt,{"remote":remote,"source_sha256":digest,
                "blob_sha":blob,"commit_sha":commit,"status":"PUT_ACCEPTED_VERIFY_PENDING"})
    for sec in RETRY_SECS:
        if sec: pause(sec)
        found=check_remote(url,token,data,blob=blob,commit=commit)
        if found:
            atomic_json(receipt,{"remote":remote,"source_sha256":digest,
                    "blob_sha":blob,"commit_sha":commit,"verified_via":found["via"],
                    "status":"VERIFIED"})
            return {"status":"VERIFIED","remote":remote,"sha256":digest,
                    "commit_sha":commit,"verified_via":found["via"]}
    raise RuntimeError("PUT accepted; local receipt kept but remote verification pending. Stop rather than upload twice.")
