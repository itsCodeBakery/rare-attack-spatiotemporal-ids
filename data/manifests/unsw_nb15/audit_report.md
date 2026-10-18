# UNSW-NB15 Kaggle audit

**Observed data audit; not model results.**

- Train: **175,341** rows; test: **82,332** rows
- Common feature columns: **42**
- Same column order: **True**
- Potential timestamp columns: **[]**
- Training duplicate fingerprints beyond first: **74,301**
- Test records with a feature fingerprint also seen in training: **8,541**

## Warnings / blockers

- No obvious timestamp columns: CSV row adjacency is not proof of temporal continuity.
- Possible cross-split feature duplicates: verify exact rows before modeling.

No imputing, normalization, label encoding, resampling, windowing, shuffling, fitting or training was performed.
Fingerprint matches require exact verification; timestamp reliability remains a manual prerequisite.
