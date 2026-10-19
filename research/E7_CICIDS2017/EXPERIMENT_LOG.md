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
