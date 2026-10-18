# Research Decision Log

All material changes in the research direction, evaluation protocol, or claimed outcomes must be recorded here in a Git commit.

| Date (UTC) | Decision | Rationale | Status |
|---|---|---|---|
| 2026-10-18 | Maintain a new dedicated research repository | Avoid mixing with unrelated intrusion-detection projects | Adopted |
| 2026-10-18 | Use the official UNSW-NB15 split as the primary benchmarking anchor | Match the reference paper and protect against uncontrolled resplitting | Official split identity verified and synced 2026-10-18 |
| 2026-10-18 | Make macro-F1 and rare-class outcomes first-class endpoints | Published macro-F1 is much lower than overall F1 | Planned; no new results |
| 2026-10-18 | Keep raw datasets and large checkpoints out of the repository | Legal, size and version-control constraints | Adopted |
| 2026-10-18 | User-run audit reported 82,332 rows in named train and 175,341 in named test; enforce official per-class split identity before training | The official standard split has 175,341 training and 82,332 testing rows. Names alone are unreliable. | Verified from synchronized Kaggle audit commit a62bdba; user-supplied filenames were reversed |
| 2026-10-18 | Verify feature-value overlap with SHA-256 and label consistency before training | The original 64-bit fingerprint report showed extensive repeated patterns but did not establish literal equality or label contradictions | Verified from Kaggle exact-value audit commit 3c99950; 8,541 overlapping test records, 4,294 without matching training label tuple |
| 2026-10-18 | Lock unchanged official test as primary, group-disjoint training-only validation and secondary overlap diagnostics | Avoid test-set selection bias and optimistic validation from identical feature vectors | Protocol v1 committed; CatBoost single-fold training-only validation observed in commit 4c5956d; publisher-original equivalence pending |
| 2026-10-18 | Recompute first CatBoost internal validation metrics from saved 10-class confusion matrix | Verify reproducible quantitative evidence before any modeling claims | 55.41% macro-F1 and 82.40% accuracy on 35,009 validation examples; publisher data check open; no official-test score |
| 2026-10-18 | Add progress-bar comparison for user-uploaded publisher CSVs, paired with the existing Kaggle mirror | Verify source equivalence, column order, exact parsed content and class distributions before further experiments | Notebook 04 prepared; execution pending; performance targets remain aspirational |
| 2026-10-18 | Accept user-uploaded publisher-source candidate as byte-for-byte equivalent to the existing Kaggle mirror | Notebook 04 and SHA-256 digests confirm both file identities, all rows and ordering | Verified copy equality in Kaggle commit 6ee35ad; publisher origin is user-attested, not independently authenticated; use dsshayan labeled paths going forward |
| 2026-10-18 | Implement paper-aligned SCV-IDS CNN–ViT baseline with 28 reproducible figures and resumable training | Verify base-paper architecture before any proposed novelty; preserve all validation evidence and figure sources | Code + Kaggle notebook committed; synthetic CI plotting test passes; GPU training **pending**; test untouched |
