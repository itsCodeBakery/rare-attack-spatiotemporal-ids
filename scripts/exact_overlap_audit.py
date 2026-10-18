"""Exact-value overlap audit for the verified UNSW-NB15 official split.

Read-only: it neither removes data nor changes the designated test set.
SHA-256 is applied to a canonical JSON representation of all 42 parsed features.
No individual traffic records or record digests are exported.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import pandas as pd

NONFEATURES = {"id", "label", "attack_cat", "attack_category"}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(row):
    """Exact equality of parsed CSV values and types, not raw CSV byte equality."""
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).digest()


def signatures(frame, features):
    # Reject missing or non-finite data; no silent normalization or imputation.
    if frame[features].isna().any().any():
        raise ValueError("Missing feature values require a separate documented policy")
    for name in features:
        if pd.api.types.is_numeric_dtype(frame[name]):
            import numpy as np
            if not bool(np.isfinite(frame[name].to_numpy(dtype="float64")).all()):
                raise ValueError("Non-finite feature values require a separate policy")
    return [canonical_digest(row) for row in
            frame[features].itertuples(index=False, name=None)]


def normalize_category(frame):
    return frame["attack_cat"].astype("string").str.strip().tolist()


def audit(manifest_path, outdir):
    manifest_path, outdir = Path(manifest_path), Path(outdir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_info, test_info = manifest["inputs"]["train"], manifest["inputs"]["test"]
    if (train_info["rows"], test_info["rows"]) != (175341, 82332):
        raise ValueError("Wrong designated UNSW-NB15 split sizes; refusing audit")
    trpath, tepath = Path(train_info["path"]), Path(test_info["path"])
    for name, path, expected in (("train", trpath, train_info),
                                 ("test", tepath, test_info)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} file missing: {path}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"{name} dataset checksum changed since Step 3 audit")
    train = pd.read_csv(trpath, low_memory=False)
    test = pd.read_csv(tepath, low_memory=False)
    if len(train) != 175341 or len(test) != 82332:
        raise ValueError("Input counts no longer match designated official split")
    if list(train.columns) != list(test.columns):
        raise ValueError("Split schema mismatch")
    feats = [x for x in train if str(x).lower() not in NONFEATURES]
    if len(feats) != 42 or feats != manifest["schema"]["common_feature_names"]:
        raise ValueError("Feature schema changed since Step 3 audit")
    for frame in (train, test):
        if "attack_cat" not in frame or "label" not in frame:
            raise ValueError("Attack class and binary labels are required")
    dtype_diffs = [c for c in feats if train[c].dtype != test[c].dtype]
    if dtype_diffs:
        raise ValueError("Feature dtypes differ across CSV inputs: " + str(dtype_diffs))
    a_sig = signatures(train, feats)
    b_sig = signatures(test, feats)
    a_cat, b_cat = normalize_category(train), normalize_category(test)
    a_bin = train["label"].astype(str).tolist()
    b_bin = test["label"].astype(str).tolist()
    a_count, b_count = Counter(a_sig), Counter(b_sig)
    common = set(a_count).intersection(b_count)
    a_labels, b_labels = defaultdict(set), defaultdict(set)
    for key, attack, binary in zip(a_sig, a_cat, a_bin):
        a_labels[key].add((attack, binary))
    for key, attack, binary in zip(b_sig, b_cat, b_bin):
        b_labels[key].add((attack, binary))

    def conflicts(labels):
        return sum(len(v) > 1 for v in labels.values())

    overlap_counts = {
        "exact_shared_feature_signatures": len(common),
        "train_records_sharing_test_feature_values": sum(a_count[x] for x in common),
        "test_records_sharing_train_feature_values": sum(b_count[x] for x in common),
        "duplicate_train_feature_records_beyond_first": len(a_sig) - len(a_count),
        "duplicate_test_feature_records_beyond_first": len(b_sig) - len(b_count),
        "within_train_feature_groups_with_label_conflict": conflicts(a_labels),
        "within_test_feature_groups_with_label_conflict": conflicts(b_labels),
        "shared_feature_groups_with_label_conflict": sum(
            len(a_labels[k] | b_labels[k]) > 1 for k in common),
        "shared_feature_groups_with_disjoint_train_test_labels": sum(
            a_labels[k].isdisjoint(b_labels[k]) for k in common),
        "test_records_with_exact_feature_and_label_match_in_train": sum(
            (key in a_labels and (lab, binary) in a_labels[key])
            for key, lab, binary in zip(b_sig, b_cat, b_bin)),
        "test_records_with_feature_match_but_no_matching_train_label": sum(
            (key in a_labels and (lab, binary) not in a_labels[key])
            for key, lab, binary in zip(b_sig, b_cat, b_bin))
    }

    per_class = []
    for split_name, cats, sig, own_labels, other_labels in (
        ("train", a_cat, a_sig, a_labels, b_labels),
        ("test", b_cat, b_sig, b_labels, a_labels)
    ):
        for cat in sorted(set(cats)):
            indices = [i for i, v in enumerate(cats) if v == cat]
            overlapping = sum(sig[i] in other_labels for i in indices)
            mismatched = sum(sig[i] in other_labels and
                             not any(pair[0] == cat for pair in other_labels[sig[i]])
                             for i in indices)
            per_class.append({
                "split": split_name, "class": cat, "n": len(indices),
                "records_with_exact_feature_match_opposite_split": overlapping,
                "records_with_no_same_class_in_opposite_split": mismatched,
                "overlap_percent": round(100 * overlapping / len(indices), 4)
            })
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    summary = {
        "audit_version": "0.2.0",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256_train": train_info["sha256"],
        "dataset_sha256_test": test_info["sha256"],
        "source_step3_manifest_sha256": file_sha256(manifest_path),
        "source_commit": git_sha,
        "row_equality": "SHA-256 of type-preserving canonical JSON of parsed feature values",
        "feature_count": len(feats),
        "train_rows": len(train), "test_rows": len(test),
        "metrics": overlap_counts,
        "time_audit": "No reliable timestamp/session metadata in these 42 features; row order unvalidated",
        "interpretation": "Matching feature values do not alone establish acquisition duplication or train/test leakage. No rows were removed.",
        "training_performed": False, "data_modified": False,
        "python_version": platform.python_version(), "pandas_version": pd.__version__
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "overlap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(per_class).to_csv(outdir / "per_class_overlap.csv", index=False)
    md = ["# Exact-value feature overlap audit", "",
          "**Computed from verified official train/test files; no training.**", "",
          f"- Training: {len(train):,}; testing: {len(test):,}",
          f"- Shared exact parsed-feature signatures: {len(common):,}", ""]
    md += [f"- {k}: **{v:,}**" for k, v in overlap_counts.items()]
    md += ["", "## Interpretation", "",
           "These are exact parsed **feature-value** matches, ignoring ID and targets. "
           "They are not automatically proof of a data split mistake or illicit leakage.",
           "Compare classes and label conflicts before defining the training protocol.",
           "**No dataset records or individual digests were published.**",
           "No deduplication, model training, imputation or split changes were performed.",
           "CSV row order has not been proven chronological; do not assume temporal continuity.",
           ""]
    (outdir / "overlap_report.md").write_text("\n".join(md), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(args.manifest, args.output)
    print("EXACT-VALUE AUDIT COMPLETED (NO TRAINING)")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
