# E7 CIC-IDS2017 — Append-only research log

Every new stage must have its own run ID, exact source file hash, Git commit, split, seed, code version, validation scope and private archive checksum. Prior runs must not be overwritten. Failure/negative results are preserved.

| Date (PKT) | Stage | Evidence | Status | Action |
|---|---|---|---|---|
| 2026-10-19 | E7 created | [Frozen E7 protocol](docs/E7_EXTERNAL_VALIDATION_PREREGISTRATION.md); [literature audit](docs/E7_NOVELTY_PRIOR_ART_AUDIT_2026.md) | PROTOCOL_REGISTERED | Do not fit until publisher, version, timestamp and session G0 checks |
| 2026-10-19 | Dataset mounted by user | Eight CSV paths reported beneath `/kaggle/input/datasets/dsshayan/dscicids/`, in a directory rendered as `TrafficLabelling ` with apparent trailing space | FILES_DISCOVERED_NOT_AUDITED | Inspect exact directory path, headers and timestamp strings read-only |
| 2026-10-19 | E7 isolated folder | [Folder README](README.md) and original E7 source/notebooks/tests copied from root with git history retained | PRESERVED_ON_GITHUB | New public E7 results and new scripts belong here |

**Scientific guard:** M1 global risk calibration was INFEASIBLE at seed 42; C4=C0. E7 cannot be considered an official UNSW retest, and no E7 model scores yet exist.

**Next run ID:** `E7-EXT-VALIDITY-001-G0` (source/schema audit only). Do not use `E7-EXT-VALIDITY-001-G1` for training until G0 checks and split preregistration are signed off.

**Backups:** Before long training, verify pushRare, auto-upload private model/prediction archives to unpublished draft GitHub releases after **each** expensive model/seed, commit public progress/results and verify remote checksums. Do not assume auto-backup exists merely because an uploader script is present.

| 2026-10-19 | E7 Notebook 26 prepared | [Eight-CSV G0 Kaggle notebook](notebooks/26_e7_cicids2017_eight_csv_g0_github_sync.ipynb); [per-file GitHub publisher](scripts/e7_g0_publish.py); [six uploader synthetic tests](tests/test_e7_g0_publish.py) passed locally; original 8-file directory and leading-space time header confirmed by user's Kaggle inspection | READY_FOR_USER_G0_RUN | Execute Notebook 26 only; full-source hashes, endpoint schema and chronology/parse anomalies, append-only GitHub public records; never train or modify prior model artifacts |

| 2026-10-19 | Notebook 26 timestamp parse stopped on Monday | User's first three Friday CSVs parsed `%d/%m/%Y %H:%M`, but `Monday-WorkingHours.pcap_ISCX.csv` preview did not. No full scans, E7 models or public result files were produced. | INPUT_FORMAT_MISMATCH_STOPPED_SAFELY | [Notebook 26A timestamp-only diagnostic](notebooks/26a_e7_cicids2017_timestamp_format_diagnostic.ipynb) committed; inspect eight files' literal timestamp strings and filename-day consistency, then register a separate corrected full G0 notebook instead of silently broadening original format. |

| 2026-10-19 | E7 Notebook 26A completed by user | Eight source files from `TrafficLabelling `; Monday has seconds, seven others minute format; day/month/year confirmed by filename weekday checks on 20 sample timestamps per file. Three Afternoon files have `1:00`/`3:30` with no AM/PM. See [original diagnostic report](data_provenance/G0_TIMESTAMP_DIAGNOSTIC_26A.md). | READ_ONLY_PREVIEW_AUDITED_CLOCK_UNRESOLVED | Freeze per-file formats; never silently reinterpret afternoon times. |
| 2026-10-19 | E7 Notebook 26B committed | [Notebook 26B](notebooks/26b_e7_g0_per_file_timestamp_autosync.ipynb), [eight-file format registry](configs/e7_g0_timestamp_registry_001.json), [four synthetic registry tests](tests/test_e7_g0_timestamp_registry.py). Uses existing GitHub publisher and provenance audit, auto-sync per CSV with remote content verification. | READY_USER_FULL_G0_RUN_NOT_EXECUTED | User runs Notebook 26B. Inspect live G0 outputs before Notebook 27; no classifier training. |

| 2026-10-19 | Notebook 26B Kaggle initialization failed | After `git fetch --depth 1 origin main`, shallow local checkout at `/kaggle/working/rare-attack-e7-cicids` rejected `git merge --ff-only FETCH_HEAD` with `refusing to merge unrelated histories`. This occurred in setup **before G0 data scan**, no changes to prior models or original official UNSW test. GitHub E7/UNSW/M1 originals separately verified still accessible. | SHALLOW_GIT_SETUP_FAILED_NO_DATA_SCAN | Corrected in [Notebook 26C](notebooks/26c_e7_g0_safe_checkout_verified_sync.ipynb): brand-new `/kaggle/working/rare-attack-e7-cicids-26c-source` clone, no fetch/merge/reset/delete of old folder; reuse isolated clean snapshot. Original Notebook 26B retained for audit history. |
