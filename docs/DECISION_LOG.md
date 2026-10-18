# Research Decision Log

All material changes in the research direction, evaluation protocol, or claimed outcomes must be recorded here in a Git commit.

| Date (UTC) | Decision | Rationale | Status |
|---|---|---|---|
| 2026-10-18 | Maintain a new dedicated research repository | Avoid mixing with unrelated intrusion-detection projects | Adopted |
| 2026-10-18 | Use the official UNSW-NB15 split as the primary benchmarking anchor | Match the reference paper and protect against uncontrolled resplitting | Planned; audit pending |
| 2026-10-18 | Make macro-F1 and rare-class outcomes first-class endpoints | Published macro-F1 is much lower than overall F1 | Planned; no new results |
| 2026-10-18 | Keep raw datasets and large checkpoints out of the repository | Legal, size and version-control constraints | Adopted |
| 2026-10-18 | User-run audit reported 82,332 rows in named train and 175,341 in named test; enforce official per-class split identity before training | The official standard split has 175,341 training and 82,332 testing rows. Names alone are unreliable. | Blocked pending verified counts and Kaggle rerun |\n