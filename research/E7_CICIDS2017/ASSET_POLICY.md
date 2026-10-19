# E7 artifact and GitHub backup policy

**Public study directory:** `research/E7_CICIDS2017/`.

### Commit to GitHub

- Source, version-locked configs, notebooks, synthetic tests, method protocols and decisions.
- Publisher URL, version attribution, original archive and CSV SHA-256, byte sizes, column names, safe aggregate label/day/support counts and **verified** split digests.
- Evaluation accuracy/macro-F1/false-alarm metrics, classwise counts, confusion matrices, uncertainty intervals, figures generated from real outputs, latency and environment tables.
- Progress manifests with run status `PLANNED/RUNNING/FAILED/COMPLETED/VERIFIED`, source commit, completed seed/stage, private draft release ID and original SHA256.

### Never commit publicly

- Original CIC-IDS2017 CSV/ZIP, raw IP/host/port/session identifiers, feature rows or raw packets.
- Private fitted models, per-example prediction/label pairs, calibrator secrets, Kaggle token, GitHub token and private download URLs.
- Any relabeled corrected-vs-original mapping containing identifying flow-level source rows.

### Recovery contract

1. Preserve original generated local checkpoint without replacing it, even if Kaggle interrupted.
2. Write deterministic archive with manifest/checksum; verify archive locally.
3. Upload to **unpublished** GitHub draft release with `pushRare`; verify GitHub asset digest and checksum sidecar.
4. Commit aggregate progress only after successful private upload. On upload failure, preserve local file and **halt expensive successive training** until recovered.
5. Restore only after validating configuration/source split, exact file hashes, manifest contents and expected model identity. No silent partial overwrite.
6. Public release publication is prohibited for these private research assets.

Prior E3/E4/E5/E6/M1 source files/results and original official UNSW test remain unchanged. This folder is additive.
