# SCV-IDS-REPRO-001 — Verified archived experiment backup

**Verification date:** 2026-10-18  
**Verification:** GitHub Release REST API; source of evidence: user's Kaggle upload report and GitHub release/asset metadata independently fetched after upload.  
**Scope:** Complete packaged experiment artifacts, including checkpoint binaries and frozen per-row predictions, stored as a **draft GitHub Release asset**, **NOT** as ordinary Git repository files.

## Verified GitHub release

- [Draft release page](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-52b64328f8a128caaee7)
- Release ID: `416471878`
- Intended tag name: `scv-ids-repro-001-artifacts`
- Status: **draft = true**, **prerelease = true**; temporary URL includes `untagged-52b64328f8a128caaee7`, as expected while a release is unpublished.
- Archive asset ID: `653575961`
- Archive filename: `SCV-IDS-REPRO-001_FULL_ARTIFACTS.tar`
- GitHub-reported archive size: **136,366,080 bytes** (130.05 MiB).
- GitHub-reported archive upload state: `uploaded`.
- GitHub-reported SHA-256: `644629dd24e142906569ac771ec8c9ec4b86b47c6605f01cf96c307b8fc3bd00`, **exactly matching the checksum supplied by Kaggle**.
- Checksum asset ID: `653576178`
- Checksum filename: `SCV-IDS-REPRO-001_FULL_ARTIFACTS.tar.sha256`
- Checksum asset size: 103 bytes; state `uploaded`.
- Archive asset link (draft asset may require authenticated collaborator access): [download from draft release](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-52b64328f8a128caaee7).
- Verification method: matching GitHub's independently returned **SHA-256 release-asset digest metadata** against the full local checksum; the raw 130 MiB binary was **not independently downloaded and rehashed inside ChatGPT**.

## Contents and limitations

The [archive construction script](../scripts/archive_scv_outputs.py) packages every existing file from `/kaggle/working/research_outputs/SCV-IDS-REPRO-001` and `/kaggle/working/research_outputs/checkpoints` with an internal SHA-256 file manifest; it explicitly excludes the UNSW raw CSV inputs and GitHub access token. The archive is intended to preserve:
- 50-epoch training checkpoints, including the validation-selected best weights from epoch 41;
- actual full validation predictions/probabilities and input-row lineage;
- training/validation split indices, embedding vectors and numerical source data;
- 28 figures, each exported in PNG, SVG, PDF;
- executable experiment configuration, plotting provenance and recovery metadata.

The public Git repository independently stores compact plot sources and all 84 figure files. The full TAR is on an **unpublished draft Release**, not in Git history. Draft-release assets are not necessarily accessible anonymously and links may change if GitHub later publishes or renames the Release.

**Security:** Do not publish this draft Release without reviewing the binary archive for licensing, sensitive data, secrets or unintended files. The hosting repository is public; a *draft release is not a substitute for a dedicated private backup*. Keep a separate downloaded copy or private Kaggle-storage copy if long-term retention is critical.

## How to verify or restore later

1. Access the draft release via the linked GitHub account in the repository's Releases area.
2. Download `SCV-IDS-REPRO-001_FULL_ARTIFACTS.tar` and its `.sha256` sidecar.
3. Verify SHA-256 of the downloaded TAR is exactly `644629dd24e142906569ac771ec8c9ec4b86b47c6605f01cf96c307b8fc3bd00`.
4. Inspect the TAR manifest `SCV-IDS-REPRO-001_ARCHIVE_MANIFEST.json` and check each archived file's recorded size and digest against restored bytes.
5. Restore into the original relative `research_outputs/SCV-IDS-REPRO-001` and `research_outputs/checkpoints` structure. Use the saved checkpoints for **diagnostic-only** CNN activation analysis. **Do not retrain just to regenerate figures**.

## Research constraints

- Confirmed internal-validation metrics: **55.8664% macro-F1**, **82.7396% accuracy** on **35,005 serialized training-derived validation windows**, not official UNSW-NB15 test results.
- The original `environment.txt` was missing; only the **separately documented recovery environment** is captured. Do not retroactively claim the original training dependencies were fully frozen.
- CNN sample embeddings exhibited zero measured variation in 128 dimensions; investigate using frozen best checkpoint under [DIAG-001](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/issues/4) before launching further model training.
- Never represent performance targets for the original SCV-IDS article as verified outperformance from internal validation.
