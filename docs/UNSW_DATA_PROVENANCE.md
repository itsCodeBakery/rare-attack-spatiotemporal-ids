# UNSW-NB15 source provenance and official-file comparison

**Status (2026-10-18):** **Original-publisher equivalence NOT YET CONFIRMED.** Suspend new model training until this question is investigated and documented. Nothing in this document invalidates previous read-only audits.

## Publisher source

- Publisher: UNSW Canberra / UNSW Research.
- Official dataset description: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Official download link: the link labeled **HERE** on that page, currently leading to UNSW's SharePoint / institutional Microsoft service. Authentication or link-access restrictions may apply; we have **not** downloaded publisher files.
- Published designated partitions: `UNSW_NB15_training-set.csv` (175,341 rows) and `UNSW_NB15_testing-set.csv` (82,332 rows).
- The complete original collection, four CSVs totaling 2,540,044 records, is **not identical in scope** to the designated standard train/test benchmark. Retain it separately if studying true event ordering.

## Current Kaggle mirror (version 1 as provided by user)

- https://www.kaggle.com/datasets/primus11/unsw-nb15-dataset/versions/1
- Kaggle paths used:
  - `/kaggle/input/datasets/primus11/unsw-nb15-dataset/unsw_test.csv` = designated **training** by *class-distribution identity*.
  - `/kaggle/input/datasets/primus11/unsw-nb15-dataset/unsw_train.csv` = designated **testing** by *class-distribution identity*.
- Mirror training SHA-256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.
- Mirror testing SHA-256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`.
- Verified: both designated row counts, exact 10-class distributions, 42 network feature names, 45 total columns, intra-/inter-split exact-feature overlap audit.
- **NOT verified:** whether publisher-supplied original file bytes (or their normalized table contents and row order) match these mirrored CSVs.
- Cross-split feature-value overlaps and label conflicts are documented in [EXACT_OVERLAP_FINDINGS.md](EXACT_OVERLAP_FINDINGS.md). They must not automatically be attributed to the uploader without comparing publisher originals.

## Planned independent-source comparison (required before we claim 'official-original equivalent')

1. Download both designated `UNSW_NB15_*ing-set.csv` files from the publisher's **HERE** page in a browser, if accessible.
2. Upload the two original publisher files to a **private Kaggle dataset**, preserving filenames exactly. Attach it as a second input to the research notebook. Do not overwrite the mirror or commit raw records to Git.
3. Record publisher landing-page URL, acquisition date, file sizes, SHA-256 and exact CSV schema.
4. Compare publisher and mirror by **role**, not file name: row counts, column names/order, class distributions, parsed **all-column** content fingerprints/multisets and ideally exact row-by-row equality and ordering. If byte hashes differ because of newline/encoding/formatting differences, compare parsed values and explain the difference.
5. Re-run exact-feature label-conflict/overlap diagnostics on the publisher files and compare with the mirror report.
6. If equal, choose publisher version as the canonical source and record the equivalence evidence, making the existing Kaggle mirror a documented byte/parsed-content equivalent for reproducibility. If unequal, stop and audit the differences before changing code or training.
7. Keep all source hashes and comparison scripts in GitHub, but **do not** redistribute the dataset files through the public code repository.

### Rights and citations

The UNSW page permits academic research use, subject to authors' requested citations and restrictions on commercial use. Cite the dataset creators Nour Moustafa and Jill Slay and other requested dataset references, rather than attributing the dataset to the Kaggle uploader.

### Training policy

The mirror is *plausibly derived* from the designated original split given exact counts and labels, but **plausible is not publisher-verified**. A baseline performed on it before equivalence is confirmed must state the mirror provenance and must not be misrepresented as a publisher-original file experiment.
