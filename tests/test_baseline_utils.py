"""Synthetic invariants; not benchmark results."""
import numpy as np
import pandas as pd
from rareids.baseline_utils import (canonical_group_ids, split_indices_sha256,
                                   group_disjoint_stratified_split)


def test_exact_identical_features_never_receive_different_group():
    frame = pd.DataFrame({"proto": ["tcp", "tcp", "udp", "udp", "icmp", "icmp"],
                          "dur": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]})
    ids = canonical_group_ids(frame, ["proto", "dur"])
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[4] == ids[5]
    assert len(set(ids)) == 3


def test_group_disjoint_validation_can_keep_all_classes():
    n = 100
    frame = pd.DataFrame({"numeric": np.arange(n, dtype=float) // 2})
    groups = canonical_group_ids(frame, ["numeric"])
    labels = np.array(["normal", "attack"] * (n // 2))
    train, valid, fold = group_disjoint_stratified_split(
        labels, groups, n_splits=5, seed=42
    )
    assert 0 <= fold < 5
    assert set(groups[train]).isdisjoint(groups[valid])
    assert set(labels[train]) == set(labels[valid]) == {"normal", "attack"}
    assert not set(train).intersection(valid)
    assert split_indices_sha256(train, valid) == split_indices_sha256(train, valid)
