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
| 2026-10-18 | Lock unchanged official test as primary, group-disjoint training-only validation and secondary overlap diagnostics | Avoid test-set selection bias and optimistic validation from identical feature vectors | Protocol v1 committed; first CatBoost control prepared; training pending |
