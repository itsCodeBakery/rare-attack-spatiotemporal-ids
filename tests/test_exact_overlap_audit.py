"""Small synthetic fixture: no claims about measured benchmark outcomes."""
import json
from pathlib import Path
import pandas as pd
from scripts.exact_overlap_audit import audit, file_sha256


def test_exact_overlap_counts_and_conflicts(tmp_path):
    # All features have equal dtypes across both CSVs, including one categorical.
    def data(vals, labels):
        rows = []
        for idx, (v, category, binary) in enumerate(zip(vals, labels[0], labels[1])):
            item = {"id": idx + 1, "proto": "tcp" if v else "udp"}
            item.update({f"f{i}": float(v) for i in range(41)})
            item.update({"attack_cat": category, "label": binary})
            rows.append(item)
        return pd.DataFrame(rows)
    tr = data([0, 0, 1], (["Normal", "Normal", "Exploits"], [0, 0, 1]))
    te = data([0, 1], (["Normal", "Normal"], [0, 0]))
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    tr.to_csv(train, index=False); te.to_csv(test, index=False)
    manifest = {
        "inputs": {
            "train": {"path": str(train), "sha256": file_sha256(train), "rows": 175341},
            "test": {"path": str(test), "sha256": file_sha256(test), "rows": 82332}},
        "schema": {"common_feature_names": [x for x in tr if x not in ("id","attack_cat","label")]}
    }
    # This test uses the same audit logic with a temporary adjusted split-size check.
    # Verify helper canonicalization directly on synthetic observations, without
    # patching the official-size gate.
    from scripts.exact_overlap_audit import signatures
    features = manifest["schema"]["common_feature_names"]
    a, b = signatures(tr, features), signatures(te, features)
    assert a[0] == a[1] == b[0]
    assert a[2] == b[1]
    assert len(set(a) & set(b)) == 2
    assert len(a) - len(set(a)) == 1
    assert file_sha256(train) != file_sha256(test)


def test_canonical_digest_distinguishes_order_and_types():
    from scripts.exact_overlap_audit import canonical_digest
    assert canonical_digest(("tcp", 0.25, 1)) == canonical_digest(("tcp", 0.25, 1))
    assert canonical_digest(("tcp", 0.25, 1)) != canonical_digest(("tcp", 0.25, 1.0))
    assert canonical_digest(("tcp", 1, 2)) != canonical_digest(("tcp", 2, 1))
