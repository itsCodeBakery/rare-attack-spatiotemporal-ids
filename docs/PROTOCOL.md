# Study Protocol (v0.1, 2026-10-18)

> **Pre-registered planning document.** Hypotheses and targets are not results. Update via committed decisions when evidence warrants changes.

## Phase A: dataset provenance and audit

1. Obtain official UNSW-NB15 training and testing splits; record provider URLs, file hashes, row counts and original column names. Do not commit raw rows.
2. Audit duplicate records and identifier columns; distinguish label leakage, cross-split duplicates and legitimate repeated traffic; document any removal with before/after counts.
3. Verify labels, benign definition, missingness, datatype and category sets. Create class-frequency tables for train/test.
4. Audit the presence, reliability and scope of timestamps or group/session identifiers. Row adjacency is **not automatically evidence of true temporal adjacency**.
5. Only after provenance checks, define the windowing protocol. Fit transformations exclusively on permitted training data. Split training from validation **before** making overlapping windows to prevent shared-event contamination.

## Phase B: reference reproduction

- Implement the SCV-IDS preprocessing and CNN-ViT crossattention baseline as far as supported by the paper, with deviations disclosed.
- Reference: 5-row windows, 42 features, 210 mapped feature slots, 16×16 RGB input, 8×8 ViT patches, and CNN–ViT bidirectional crossattention. Details and encoding ambiguity must be confirmed experimentally and documented.
- Evaluate binary and ten-class multiclass tasks separately. Preserve the official test set untouched until model and hyperparameters are frozen.
- Disclose the exact averaging convention of every F1 score. Report at least accuracy, macro-/weighted-F1, per-class precision/recall/F1, confusion matrix, benign false-positive rate, and bootstrap uncertainty where appropriate.

## Phase C: proposed innovations (hypotheses)

- Test rare-signal-preserving representations independently of class-balancing choices.
- Benchmark against strong non-image/tabular and sequence baselines, not only vision variants.
- Quantify inference throughput, latency, memory and model size under stated hardware.
- Explore open-set rejection only under explicitly defined held-out attack families. Do not use an open-set headline without a held-out protocol and threshold calibration strictly on permitted validation data.

## Phase D: rigorous ablations

- Sliding-window size and ordering validity; semantic vs arbitrary feature layout; categorical encoding; feature masking/padding.
- Backbone branches (CNN-only, ViT-only, fused) and candidate fusion improvements.
- Losses, minority sampling, signal-preservation components, inference threshold, and calibration.
- Repeated seeds (target at least 5 for key claims), confidence intervals and paired tests when appropriate.
- Cross-dataset training/testing, if included, must be distinguished from training and evaluating separately on each dataset; incompatible class schemas must be reconciled before any shared-label claims.

## Every experiment records

- `run_id`, UTC start time, purpose, status (planned/running/failed/completed/verified).
- Dataset provider/version and SHA-256 file hashes; split index or algorithm; exact train/val/test counts.
- Git commit SHA, notebook/script, full config and seed.
- Python/CUDA/PyTorch/sklearn versions, GPU model, available VRAM and execution time.
- Metrics from actual saved labels/predictions; paths + checksums for artifacts not stored in Git.
- Known limitations, deviations, errors and the follow-up decision.

## Evidence rules

Never report fabricated, inferred, or table-derived confusion matrices as model outputs. All figures derive from audited experimental data, are appropriately captioned, and expose data/code provenance. Do not present numbers from different datasets or evaluation splits as directly comparable without a caveat.
