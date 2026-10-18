# UNSW-NB15 publisher-upload vs Kaggle mirror

**Observed content comparison only; no model training/test inference.**
**Source identity:** these paths were supplied by the user as publisher originals; a publisher signature or independent origin attestation has not been obtained.

| Split | Original upload vs mirror | Original rows | Byte identical | Parsed row-order equality | Parsed multiset equality |
|---|---|---:|---|---|---|
| train | BYTE_IDENTICAL | 175,341 | True | True | True |
| test | BYTE_IDENTICAL | 82,332 | True | True | True |

Both complete data partitions contain matching parsed data in the same order: **True**.

If CSV byte hashes differ but parsed rows match, differences may arise from CSV serialization, encodings, or float formatting. This **does not establish publisher origin**.
If parsed records differ, stop and inspect aggregate differences before training.
Identical row multisets with different row orders must not be treated as equivalent for sequential windowing/row-serialization experiments.
This notebook does not prove CSV row order is chronological traffic order.
