"""Helpers for leakage-aware internal validation (no access to the test set)."""
import hashlib
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def canonical_group_ids(frame, feature_columns):
    """Map 42-feature identical parsed row tuples to one integer group."""
    if frame[feature_columns].isna().any().any():
        raise ValueError("Missing values in baseline input")
    lut, ids = {}, []
    for row in frame[feature_columns].itertuples(index=False, name=None):
        digest = hashlib.sha256(json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).digest()
        if digest not in lut:
            lut[digest] = len(lut)
        ids.append(lut[digest])
    return np.asarray(ids, dtype=np.int32)


def group_disjoint_stratified_split(y, groups, n_splits=5, seed=42):
    """Select a deterministic group-disjoint fold with all observed classes present.

    Fold selection considers *training-side labels only*. It does not optimize
    validation performance. Fails closed if minority-class support is absent.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("Label/group row count mismatch")
    classes = set(y.tolist())
    candidates = []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (train_indices, valid_indices) in enumerate(
        cv.split(np.zeros(len(y)), y, groups)
    ):
        if set(y[train_indices]) != classes or set(y[valid_indices]) != classes:
            continue
        if set(groups[train_indices]) & set(groups[valid_indices]):
            raise AssertionError("Shared feature groups across validation boundary")
        rate = len(valid_indices) / len(y)
        # Do not select using any predictions, scores, or official-test labels.
        score = abs(rate - 1.0 / n_splits)
        candidates.append((score, fold_id, train_indices, valid_indices))
    if not candidates:
        raise ValueError(
            "No complete-class group-disjoint validation fold: "
            "choose and document another training-only validation protocol."
        )
    _, fold_id, train_indices, valid_indices = min(candidates, key=lambda x: (x[0], x[1]))
    return train_indices, valid_indices, fold_id


def split_indices_sha256(train_indices, valid_indices):
    hasher = hashlib.sha256()
    for role, values in (("train", train_indices), ("validation", valid_indices)):
        arr = np.asarray(values, dtype="<i8")
        hasher.update(role.encode("utf-8") + b":")
        hasher.update(str(len(arr)).encode("ascii") + b":")
        hasher.update(arr.tobytes())
    return hasher.hexdigest()
