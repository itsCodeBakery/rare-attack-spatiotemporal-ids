"""Publisher-upload vs Kaggle-mirror provenance comparison.

This is a READ-ONLY data audit: hashes and all-column parsed-row equivalence.
Progress bars are displayed for hashing, reading, and per-record comparison.
No individual records, row hashes, training, or test predictions are exported.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import pandas as pd
from tqdm.auto import tqdm

EXPECTED = {
    "train": {
        "normal": 56000, "analysis": 2000, "backdoor": 1746, "dos": 12264,
        "exploits": 33393, "fuzzers": 18184, "generic": 40000,
        "reconnaissance": 10491, "shellcode": 1133, "worms": 130
    },
    "test": {
        "normal": 37000, "analysis": 677, "backdoor": 583, "dos": 4089,
        "exploits": 11132, "fuzzers": 6062, "generic": 18871,
        "reconnaissance": 3496, "shellcode": 378, "worms": 44
    }
}
EXPECTED_TOTAL = {"train": 175341, "test": 82332}
EXCLUDED_ID = {"id"}


def hash_file(path, desc):
    path = Path(path)
    size = path.stat().st_size
    sha = hashlib.sha256()
    with path.open("rb") as stream, tqdm(total=size, unit="B", unit_scale=True, desc=desc,
                                         dynamic_ncols=True) as bar:
        while True:
            data = stream.read(4 * 1024 * 1024)
            if not data:
                break
            sha.update(data)
            bar.update(len(data))
    return sha.hexdigest()


def read_csv_progress(path, rows, desc):
    chunks = []
    with tqdm(total=rows, unit="rows", desc=desc, dynamic_ncols=True) as bar:
        for piece in pd.read_csv(path, low_memory=False, chunksize=20000):
            chunks.append(piece)
            bar.update(len(piece))
    if not chunks:
        raise ValueError("Empty CSV input: " + str(path))
    frame = pd.concat(chunks, ignore_index=True)
    if len(frame) != rows:
        raise ValueError(f"{path}: expected {rows} records, found {len(frame)}")
    if not frame.columns.is_unique:
        raise ValueError(f"{path}: duplicate column names")
    if frame.isna().any().any():
        raise ValueError(f"{path}: missing parsed CSV values; manual audit required")
    return frame


def attack_counts(frame):
    col = next((c for c in frame if c.lower() in {"attack_cat", "attack_category"}), None)
    if col is None:
        raise ValueError("Attack-category label missing")
    names = frame[col].astype(str).str.strip().str.casefold().replace({"backdoors": "backdoor"})
    return {str(k): int(v) for k, v in names.value_counts().items()}


def compare_columns(left, right):
    same_set = set(left.columns) == set(right.columns)
    return {"same_column_set": same_set,
            "same_column_order": list(left.columns) == list(right.columns),
            "left_columns": list(left.columns), "right_columns": list(right.columns)}


def canonical_numeric(value):
    if pd.isna(value):
        return ["missing", None]
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Non-numeric value in numeric column") from exc
    if not number.is_finite():
        raise ValueError("Non-finite numeric feature value")
    if number == 0:
        number = Decimal(0)
    return ["n", str(number.normalize())]


def row_multiset(frame, columns, numeric_columns, desc):
    numeric = [col in numeric_columns for col in columns]
    counts = Counter()
    digests = []
    rows = frame[columns].itertuples(index=False, name=None)
    for row in tqdm(rows, total=len(frame), unit="rows", desc=desc, dynamic_ncols=True):
        encoded = []
        for item, is_numeric in zip(row, numeric):
            if is_numeric:
                encoded.append(canonical_numeric(item))
            else:
                encoded.append(["s", str(item)])
        data = json.dumps(encoded, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        fingerprint = hashlib.sha256(data).digest()
        counts[fingerprint] += 1
        digests.append(fingerprint)
    return counts, digests


def compare_pair(role, original_path, mirror_path, mirror_sha, output_dir):
    original_path, mirror_path = Path(original_path), Path(mirror_path)
    for src in (original_path, mirror_path):
        if not src.is_file():
            raise FileNotFoundError("Kaggle dataset attachment missing: " + str(src))
    original_hash = hash_file(original_path, f"{role}: publisher-upload hash")
    mirror_hash = hash_file(mirror_path, f"{role}: mirror hash")
    if mirror_hash != mirror_sha:
        raise ValueError("Existing mirror file differs from previously audited SHA-256: " + role)
    rows = EXPECTED_TOTAL[role]
    original = read_csv_progress(original_path, rows, f"{role}: publisher-upload CSV")
    mirror = read_csv_progress(mirror_path, rows, f"{role}: Kaggle mirror CSV")
    orig_counts, mirror_counts = attack_counts(original), attack_counts(mirror)
    if orig_counts != EXPECTED[role]:
        raise ValueError(f"{role}: publisher-upload class distribution differs from official designated split: {orig_counts}")
    if mirror_counts != EXPECTED[role]:
        raise ValueError(f"{role}: Kaggle mirror class distribution differs from designated split")

    cols = compare_columns(original, mirror)
    # Structural differences must stop the full-cell comparison; do not silently reorder.
    if not cols["same_column_set"]:
        return {
            "role": role, "decision": "NOT_EQUIVALENT_SCHEMA",
            "publisher_uploaded_sha256": original_hash, "mirror_sha256": mirror_hash,
            "publisher_uploaded_bytes": original_path.stat().st_size,
            "mirror_bytes": mirror_path.stat().st_size, "rows": rows,
            "class_counts_equal": True,
            "schema": cols, "byte_identical": original_hash == mirror_hash,
            "same_parsed_row_order_full": False, "same_parsed_multiset_full": False,
            "same_parsed_row_order_excluding_id": False,
            "same_parsed_multiset_excluding_id": False,
            "unmatched_publisher_rows_full": None, "unmatched_mirror_rows_full": None
        }
    common_cols = list(original.columns)
    mirror = mirror[common_cols]
    numeric_cols = {
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(original[c])
        and pd.api.types.is_numeric_dtype(mirror[c])
    }
    mismatched_kinds = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(original[c])
        != pd.api.types.is_numeric_dtype(mirror[c])
    ]
    if mismatched_kinds:
        raise ValueError(f"{role}: numeric/text column-type mismatches: {mismatched_kinds}")
    all_orig, all_orig_order = row_multiset(
        original, common_cols, numeric_cols, f"{role}: original full rows")
    all_mirror, all_mirror_order = row_multiset(
        mirror, common_cols, numeric_cols, f"{role}: mirror full rows")
    same_order_full = all_orig_order == all_mirror_order
    same_multiset_full = all_orig == all_mirror
    # ID differences can occur when a public dataset is repackaged.
    # Only calculate ID-independent signatures if full parsed table differs.
    if same_multiset_full and same_order_full:
        same_multiset_no_id = True
        same_order_no_id = True
    else:
        without_id = [c for c in common_cols if c.lower() not in EXCLUDED_ID]
        noid_a, seq_a = row_multiset(
            original, without_id, numeric_cols, f"{role}: original without ID")
        noid_b, seq_b = row_multiset(
            mirror, without_id, numeric_cols, f"{role}: mirror without ID")
        same_multiset_no_id = noid_a == noid_b
        same_order_no_id = seq_a == seq_b
    if original_hash == mirror_hash:
        decision = "BYTE_IDENTICAL"
    elif same_order_full and cols["same_column_order"]:
        decision = "PARSED_ROWS_SAME_ORDER"
    elif same_multiset_full:
        decision = "SAME_PARSED_ROWS_DIFFERENT_ORDER_OR_COLUMNS"
    elif same_order_no_id and cols["same_column_order"]:
        decision = "SAME_PARSED_ROWS_EXCEPT_ID_SAME_ORDER"
    elif same_multiset_no_id:
        decision = "SAME_PARSED_ROWS_EXCEPT_ID_DIFFERENT_ORDER"
    else:
        decision = "CONTENT_DIFFERENCE_INVESTIGATE"
    return {
        "role": role, "decision": decision,
        "publisher_uploaded_sha256": original_hash, "mirror_sha256": mirror_hash,
        "publisher_uploaded_bytes": original_path.stat().st_size,
        "mirror_bytes": mirror_path.stat().st_size,
        "rows": rows, "class_counts_equal": orig_counts == mirror_counts == EXPECTED[role],
        "schema": cols, "numeric_columns_compared": sorted(numeric_cols),
        "byte_identical": original_hash == mirror_hash,
        "same_parsed_row_order_full": same_order_full,
        "same_parsed_multiset_full": same_multiset_full,
        "same_parsed_row_order_excluding_id": same_order_no_id,
        "same_parsed_multiset_excluding_id": same_multiset_no_id,
        "unmatched_publisher_rows_full": int(sum((all_orig - all_mirror).values())),
        "unmatched_mirror_rows_full": int(sum((all_mirror - all_orig).values())),
        "source_identity_note": "Paths are user-provided publisher-upload candidates, not independent proof of origin."
    }


def run(manifest_path, official_train, official_test, output_dir):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["inputs"]["train"]["rows"] != 175341 or manifest["inputs"]["test"]["rows"] != 82332:
        raise ValueError("Audited mirror roles/sizes changed")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    pairs = [
        ("train", official_train, manifest["inputs"]["train"]["path"],
         manifest["inputs"]["train"]["sha256"]),
        ("test", official_test, manifest["inputs"]["test"]["path"],
         manifest["inputs"]["test"]["sha256"])
    ]
    results = []
    for role, original, mirror, checksum in tqdm(pairs, desc="Source partitions", unit="split"):
        results.append(compare_pair(role, original, mirror, checksum, output_dir))
    verdicts = {x["role"]: x["decision"] for x in results}
    valid_rows = all(x["same_parsed_multiset_full"] and x["same_parsed_row_order_full"]
                     and x["schema"]["same_column_order"] for x in results)
    no_id_equivalent = all(x["same_parsed_multiset_excluding_id"] for x in results)
    identity = {
        "audit_version": "0.3.0",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "official_source_claim": "User reports these files were downloaded from UNSW and uploaded to Kaggle. Independent origin attestation unavailable.",
        "publisher_landing_page": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "uploaded_dataset_path": "/kaggle/input/datasets/dsshayan/unsw-nb15",
        "mirror_dataset_path": "/kaggle/input/datasets/primus11/unsw-nb15-dataset",
        "comparison": results,
        "verdicts": verdicts,
        "same_full_parsed_rows_in_same_order_across_both_splits": valid_rows,
        "same_full_parsed_row_multisets_both_splits": all(
            x["same_parsed_multiset_full"] for x in results),
        "same_without_id_multisets_both_splits": no_id_equivalent,
        "publication_provenance_confirmed_independently": False,
        "no_model_training_or_inference": True,
        "python": platform.python_version(),
        "pandas": pd.__version__
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_comparison.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    notes = [
        "# UNSW-NB15 publisher-upload vs Kaggle mirror",
        "",
        "**Observed content comparison only; no model training/test inference.**",
        "**Source identity:** these paths were supplied by the user as publisher originals; "
        "a publisher signature or independent origin attestation has not been obtained.",
        "",
        "| Split | Original upload vs mirror | Original rows | Byte identical | Parsed row-order equality | Parsed multiset equality |",
        "|---|---|---:|---|---|---|"
    ]
    for x in results:
        notes.append(
            f"| {x['role']} | {x['decision']} | {x['rows']:,} | "
            f"{x['byte_identical']} | {x['same_parsed_row_order_full']} | "
            f"{x['same_parsed_multiset_full']} |"
        )
    notes += [
        "",
        "Both complete data partitions contain matching parsed data in the same order: "
        f"**{valid_rows}**.",
        "",
        "If CSV byte hashes differ but parsed rows match, differences may arise from CSV "
        "serialization, encodings, or float formatting. This **does not establish publisher origin**.",
        "If parsed records differ, stop and inspect aggregate differences before training.",
        "Identical row multisets with different row orders must not be treated as equivalent "
        "for sequential windowing/row-serialization experiments.",
        "This notebook does not prove CSV row order is chronological traffic order.",
        ""
    ]
    (output_dir / "source_comparison_report.md").write_text(
        "\n".join(notes), encoding="utf-8")
    print("SOURCE COMPARISON COMPLETE")
    print("Train/test outcomes:", verdicts)
    print("Identical parsed rows in order:", valid_rows)
    print("Independent publisher attestation: NOT performed")
    return identity


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--official-train", required=True)
    p.add_argument("--official-test", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args.manifest, args.official_train, args.official_test, args.output)
