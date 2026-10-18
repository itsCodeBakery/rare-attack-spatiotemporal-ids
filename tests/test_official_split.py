"""Synthetic unit tests for data-file role verification (not measured dataset results)."""
from pathlib import Path
import pandas as pd
import pytest
from rareids.splits import attack_counts, resolve_by_class_counts, OFFICIAL_TRAIN, OFFICIAL_TEST


def test_counts_and_normalization(tmp_path):
    f = tmp_path / "tiny.csv"
    pd.DataFrame({"attack_cat": ["Backdoors", "backdoor", "Normal"]}).to_csv(f, index=False)
    assert attack_counts(f) == {"backdoor": 2, "normal": 1}


def test_reference_totals():
    assert sum(OFFICIAL_TRAIN.values()) == 175341
    assert sum(OFFICIAL_TEST.values()) == 82332


def test_refuse_nonofficial_distribution(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"attack_cat": ["normal"]}).to_csv(a, index=False)
    pd.DataFrame({"attack_cat": ["analysis"]}).to_csv(b, index=False)
    with pytest.raises(ValueError, match="NEITHER arrangement"):
        resolve_by_class_counts(a, b)
