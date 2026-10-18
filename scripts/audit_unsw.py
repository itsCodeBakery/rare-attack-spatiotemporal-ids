"""Read-only audit of Kaggle UNSW-NB15 files. Produces aggregate metadata only."""
import argparse, hashlib, json, os, platform, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

TARGETS = {"id", "label", "attack_cat", "attack_category"}
TIME_NAMES = {"stime", "ltime", "timestamp", "time", "start_time",
              "end_time", "flow_start_time", "flow_end_time", "datetime", "ts"}

def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def label_counts(s):
    values = s.astype("string").str.strip().fillna("<MISSING>")
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}

def audit(train_path, test_path, out):
    train_path, test_path, out = map(Path, (train_path, test_path, out))
    for p in (train_path, test_path):
        if not p.is_file():
            raise FileNotFoundError("Input unavailable: " + str(p))
    if train_path.resolve() == test_path.resolve():
        raise ValueError("Train and test must be different files")
    tr = pd.read_csv(train_path, low_memory=False)
    te = pd.read_csv(test_path, low_memory=False)
    if tr.empty or te.empty:
        raise ValueError("Both splits must contain records")
    if not tr.columns.is_unique or not te.columns.is_unique:
        raise ValueError("Duplicate CSV column names are not permitted")
    warnings = []
    if list(tr.columns) != list(te.columns):
        warnings.append("Train/test column lists or ordering differ; review before training.")
    common = [c for c in tr.columns if c in te.columns]
    features = [c for c in common if c.lower() not in TARGETS]
    if len(features) != 42:
        warnings.append(f"Detected {len(features)} common feature columns; reference study uses 42.")
    time_cols = [c for c in common if c.lower() in TIME_NAMES or "timestamp" in c.lower()]
    if time_cols:
        warnings.append("Timestamp candidates exist but require a manual ordering/session audit: " + ", ".join(time_cols))
    else:
        warnings.append("No obvious timestamp columns: CSV row adjacency is not proof of temporal continuity.")
    if not features:
        raise ValueError("No common feature columns")
    # Duplicate counts are diagnostics based on fast, non-cryptographic 64-bit fingerprints.
    htr = pd.util.hash_pandas_object(tr[features], index=False)
    hte = pd.util.hash_pandas_object(te[features], index=False)
    ftr, fte = htr.value_counts(), hte.value_counts()
    common_hashes = ftr.index.intersection(fte.index)
    dup = {"fingerprint_method": "pandas 64-bit features-only; potential collisions/dtype effects",
           "train_duplicate_rows_beyond_first": int(htr.duplicated().sum()),
           "test_duplicate_rows_beyond_first": int(hte.duplicated().sum()),
           "common_feature_fingerprints": int(len(common_hashes)),
           "train_rows_sharing_test_fingerprint": int(ftr.loc[common_hashes].sum()),
           "test_rows_sharing_train_fingerprint": int(fte.loc[common_hashes].sum())}
    if dup["common_feature_fingerprints"]:
        warnings.append("Possible cross-split feature duplicates: verify exact rows before modeling.")
    schema, classes = [], []
    label_summary = {}
    inf_summary = {}
    for name, frame in (("train", tr), ("test", te)):
        for col in frame.columns:
            schema.append({"split": name, "column": col, "dtype": str(frame[col].dtype),
                           "missing": int(frame[col].isna().sum()),
                           "unique": int(frame[col].nunique(dropna=True))})
        label_summary[name] = {}
        for kind, aliases in (("attack_category", {"attack_cat", "attack_category"}),
                              ("binary_label", {"label"})):
            target = next((c for c in frame if c.lower() in aliases), None)
            if target is None:
                warnings.append(f"{name}: missing {kind} target column.")
                continue
            counts = label_counts(frame[target])
            label_summary[name][kind] = counts
            for cls, num in counts.items():
                classes.append({"split": name, "target": kind, "class": cls, "count": num,
                                "percent": round(100*num/len(frame), 6)})
        inf_summary[name] = {}
        for col in frame.select_dtypes(include="number").columns:
            count = int(np.isinf(frame[col].to_numpy(dtype=np.float64, copy=False)).sum())
            if count:
                inf_summary[name][col] = count
        if inf_summary[name]:
            warnings.append(f"{name}: non-finite infinity values observed.")
    try:
        source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_sha = None
    files = {}
    for name, path, frame in (("train", train_path, tr), ("test", test_path, te)):
        files[name] = {"path": str(path), "sha256": checksum(path),
                       "bytes": path.stat().st_size, "rows": len(frame), "columns": len(frame.columns)}
    summary = {"audit_version": "0.1.0",
               "observed_utc": datetime.now(timezone.utc).isoformat(),
               "source_commit": source_sha,
               "dataset": "Kaggle primus11/unsw-nb15-dataset (user-provided mount)",
               "inputs": files,
               "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
               "schema": {"common_feature_count": len(features), "common_feature_names": features,
                          "same_column_set": set(tr.columns) == set(te.columns),
                          "same_column_order": list(tr.columns) == list(te.columns),
                          "potential_time_columns": time_cols},
               "classes": label_summary, "duplicates": dup, "infinity_counts": inf_summary,
               "warnings": warnings, "preprocessing_performed": False, "training_performed": False}
    out.mkdir(parents=True, exist_ok=True)
    (out/"audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    pd.DataFrame(schema).to_csv(out/"column_schema.csv", index=False)
    pd.DataFrame(classes, columns=["split","target","class","count","percent"]).to_csv(out/"class_distribution.csv", index=False)
    report = ["# UNSW-NB15 Kaggle audit", "", "**Observed data audit; not model results.**", "",
              f"- Train: **{len(tr):,}** rows; test: **{len(te):,}** rows",
              f"- Common feature columns: **{len(features)}**",
              f"- Same column order: **{list(tr.columns) == list(te.columns)}**",
              f"- Potential timestamp columns: **{time_cols}**",
              f"- Training duplicate fingerprints beyond first: **{dup['train_duplicate_rows_beyond_first']:,}**",
              f"- Test records with a feature fingerprint also seen in training: **{dup['test_rows_sharing_train_fingerprint']:,}**",
              "", "## Warnings / blockers", ""]
    report.extend("- "+w for w in warnings)
    report += ["", "No imputing, normalization, label encoding, resampling, windowing, shuffling, fitting or training was performed.",
               "Fingerprint matches require exact verification; timestamp reliability remains a manual prerequisite.", ""]
    (out/"audit_report.md").write_text("\n".join(report))
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = audit(a.train, a.test, a.output)
    print("AUDIT COMPLETE:", r["inputs"]["train"]["rows"], "train /",
          r["inputs"]["test"]["rows"], "test; warnings:", len(r["warnings"]))
    print("No model training performed.")
