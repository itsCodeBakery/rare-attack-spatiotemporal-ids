"""Provenance gate for the published official UNSW-NB15 train/test partition.

The user-supplied Kaggle mirror may misname the two CSV files. Confirm *per-class*
histograms against the official 175,341/82,332 split before any preprocessing.
"""
from pathlib import Path
import pandas as pd

# Source: UNSW Canberra UNSW-NB15 dataset page (official row totals);
# published class distribution from the official split (Moustafa & Slay).
OFFICIAL_TRAIN = {
    "normal": 56000, "analysis": 2000, "backdoor": 1746,
    "dos": 12264, "exploits": 33393, "fuzzers": 18184,
    "generic": 40000, "reconnaissance": 10491,
    "shellcode": 1133, "worms": 130
}
OFFICIAL_TEST = {
    "normal": 37000, "analysis": 677, "backdoor": 583,
    "dos": 4089, "exploits": 11132, "fuzzers": 6062,
    "generic": 18871, "reconnaissance": 3496,
    "shellcode": 378, "worms": 44
}
assert sum(OFFICIAL_TRAIN.values()) == 175341
assert sum(OFFICIAL_TEST.values()) == 82332


def attack_counts(csv_path):
    path = Path(csv_path)
    header = pd.read_csv(path, nrows=0).columns
    col = next((c for c in header if str(c).strip().casefold() in
                ("attack_cat", "attack_category")), None)
    if col is None:
        raise ValueError(f"{path.name} has no attack category column")
    s = pd.read_csv(path, usecols=[col], dtype={col: "string"})[col]
    if s.isna().any():
        raise ValueError(f"{path.name} has missing attack labels")
    normalized = s.str.strip().str.casefold().replace({"backdoors": "backdoor"})
    return {str(k): int(v) for k,v in normalized.value_counts().items()}


def resolve_by_class_counts(named_train, named_test):
    """Return (verified_train_path, verified_test_path, mapping_description, counts)."""
    a, b = Path(named_train), Path(named_test)
    ca, cb = attack_counts(a), attack_counts(b)
    if ca == OFFICIAL_TRAIN and cb == OFFICIAL_TEST:
        return a, b, "named_files_match_official", {"named_train": ca, "named_test": cb}
    if ca == OFFICIAL_TEST and cb == OFFICIAL_TRAIN:
        return b, a, "named_files_are_reversed", {"named_train": ca, "named_test": cb}
    raise ValueError(
        "NEITHER arrangement matches official UNSW-NB15 per-class counts. "
        "Do not train or reinterpret the split; inspect mirror provenance. "
        f"Named train counts: {ca}; named test counts: {cb}"
    )
