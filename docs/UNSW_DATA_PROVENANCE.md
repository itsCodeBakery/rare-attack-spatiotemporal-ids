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

### Publisher-upload Kaggle paths (provided 2026-10-18)

- Original-designated training candidate: `/kaggle/input/datasets/dsshayan/unsw-nb15/UNSW_NB15_training-set.csv`.
- Original-designated test candidate: `/kaggle/input/datasets/dsshayan/unsw-nb15/UNSW_NB15_testing-set.csv`.
- Run [Notebook 04](../notebooks/04_verify_publisher_source.ipynb) **after uploads finish and both datasets are attached in Kaggle**. It displays progress bars, verifies full-row parsed-content equality and sequence ordering, and syncs aggregate results.
- **Important provenance distinction:** the paths are user-provided as university-source files; even perfect equality with the mirror cannot independently authenticate a publisher download. Keep the acquisition record.

## Completed source-comparison audit (2026-10-18)

Kaggle [source-comparison commit](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/6ee35ada942df078d6c7883b8a377279b7bf349b) reports `BYTE_IDENTICAL` for **both** user-uploaded designated UNSW files and their corresponding Kaggle mirror files. Raw bytes, file sizes, 45-column schema and ordering, every parsed record, duplicate multiplicities, and every record's relative row order match. Both splits have **0 unmatched records**. Training file SHA-256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` (175,341 rows, 32,293,018 bytes). Testing file SHA-256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` (82,332 rows, 15,380,800 bytes).

**Scientific conclusion:** The two Kaggle datasets are reproducibly *the same digital data* by matching SHA-256 digests. This supports reusing earlier read-only audits and the baseline validation without repeating them for source-migration reasons. The user reports the `dsshayan` files were obtained from the original university source; the audit did **not** independently authenticate that distribution source. Byte-identity does not establish chronological traffic continuity or resolve identical feature signatures with inconsistent labels.

**Source decision:** Prefer the explicitly labeled `dsshayan` paths for subsequent experiments. Keep both original SHA-256 identities and original run-specific source paths in experiment manifests; never revise history to pretend earlier experiments used the new path. The untouched designated test set remains held out for a final locked-model evaluation.
