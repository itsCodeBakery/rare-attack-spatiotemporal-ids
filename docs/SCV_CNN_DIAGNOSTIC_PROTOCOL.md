# DIAG-001 — Frozen CNN branch and crossattention diagnostics

**Prepared:** 2026-10-18. **Status:** Kaggle execution pending.  
**Notebook:** [07_scv_cnn_diagnostic.ipynb](../notebooks/07_scv_cnn_diagnostic.ipynb)  
**Reference:** The original [SCV-IDS-REPRO-001 run](SCV_IDS_001_RESULTS_AUDIT.md), specifically the best **epoch-41** checkpoint; **not the official test**.

## Observation, not yet explanation

Our [projection diagnostics](../results/SCV-IDS-REPRO-001/projection_diagnostics.json) found **zero numerically informative dimensions in the 128-D CNN pooled representation** among a limited sample of frozen internal validation examples. ViT and fused embeddings were nonconstant. This raises a concern that the CNN arm may contribute little or no instance-varying information, but it **does not prove universal collapse**, or prove the network architecture from the paper is defective. The current implementation is a *paper-aligned independent reconstruction*, not verified author source code.

The same run shows **DoS→Exploits = 2,130/2,509** and **Backdoor→Exploits = 286/350** internal validation confusions, warranting evidence-based follow-up.

## Primary diagnosis — *no training*

**Pinned checkpoint:** `1ff947f795162b747115fee6e3131eb10fd3955445a3040902a274fa71457c65`  
**Pinned full 130.05 MiB archive:** `644629dd24e142906569ac771ec8c9ec4b86b47c6605f01cf96c307b8fc3bd00`  
**Original designated training CSV:** `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`

1. Use the original Kaggle `/kaggle/working/research_outputs` files if present. Otherwise download the pinned **draft GitHub Release asset** using the authorized `pushRare` Kaggle secret, verify the complete archive's SHA-256, and restore all files with individual checksums and safe TAR member validation. No download if the files are already available. This is a read-only restore.
2. Load the untouched official **TRAIN** CSV; recover the original train-side validation split indices, original label sequence, and fitted categorical vocabularies plus MinMax params from the saved experiment. Do **not** call `fit_preprocessor` or use the official testing CSV.
3. Form a deterministic sample: the previously saved class-stratified validation embeddings plus up to 512 additional random training-derived validation windows, maximum 1,800. The sampling seed is 42.
4. Load the original saved `SCVIDS` epoch-41 checkpoint strictly. Replay all sampled windows using exactly the saved 16×16 RGB serialization and preprocessing.
5. **Integrity gate:** require every argmax prediction to match saved frozen probabilities, with **max absolute probability deviation ≤ 0.002**. Compare the original sampled CNN pooled embeddings against the re-extracted ones to tolerance 1e-4. If either fails, stop and investigate model/serialization mismatch before any interpretation.
6. Measure every original `CNNEncoder.layers` Conv/ReLU/MaxPool activation stage: fraction of exactly zero activations, between-image channel mean variability, and within-image spatial variability. Retain full per-channel aggregate values (not images, raw flow rows, or per-record predictions in GitHub).
7. Evaluate variability in pooled CNN, **spatial CNN tokens**, pooled ViT, both attention-output directions, and fused embeddings; distinguish constant pooled values from the possibility of variable CNN tokens.
8. Conduct paired **inference-only** zeroing of (a) CNN K/V branch, (b) ViT branch, and (c) both; measure logit changes and per-class validation metrics. These are out-of-distribution interventions and should not be claimed to be causal evidence about contribution during training.
9. Apply a controlled **horizontal flip** of the serialized input image to test CNN/ViT representation sensitivity without gradients. This is a diagnostic perturbation, **not an actual network attack** nor a robustness metric.
10. Store observed aggregate data, source metadata, metric snapshots, and SHA-256 hashes; regenerate **12 scientific plots × PNG/SVG/PDF = 36 figure files** via versioned `scripts/plot_scv_cnn_diagnostic.py`. The 12 graphs are: layer between-image variance, zeros, constant-channel fraction, spatial variance, branch variance, metrics under masked branches, per-class recalls, logit shifts, RGB channel distributions, flip sensitivity, channel-level distribution, and sampled confusion. All 12 plots derive exclusively from saved numeric CSVs and can be edited without retraining.

## Interpretation checklist

- **CNN truly constant across all observed layers?** Inspect which Conv/ReLU stage first collapses to zero variance. Possible hypotheses include ReLU saturation, loss-driven feature ignoring, or near-invariant serialized inputs. These are *hypotheses only* until actual activation statistics are observed.
- **CNN pooled constant, spatial tokens variable?** Attention K/V may still contain structure even if mean pooling is constant. Compare flattened CNN token variance versus pooled variance before claiming branch collapse.
- **Other branches compensate?** ViT and fused representation variance, plus logits/metrics sensitivity to zeroing, are diagnostic. If zeroing CNN hardly changes logits, it supports, but does not definitively prove, a limited instance-specific role.
- **Prediction replay mismatched?** Stop. It could indicate changed model code, saved config, preprocessing ordering, torch numeric behavior, or corrupted artifacts. Do not present any figures as results.
- **Subset versus full-validation distinction:** the diagnostic tests at most 1,800 internal validation windows, while the archived primary macro-F1 of **55.8664%** was computed on **35,005 internal validation windows**. Do not equate subset metrics with primary validation performance.
- **Chronology:** no timestamp proof for CSV row order, and no original author source-code equivalence.
- **Official UNSW test stays held out**; no model changes, calibration, experiments on official test, or claims of outperforming 2026 SCV-IDS from these diagnostics.

## Preservation

All diagnostic numerical CSVs and the 36 publication figure files will be pushed to `results/SCV-CNN-DIAG-001/` using `pushRare`. No original packet/flow records, row-level predictions, unpublished checkpoints or tokens will be uploaded publicly. The complete original trained-model archive is already SHA-256-verified on a **draft GitHub Release**; see [archive provenance](SCV_IDS_ARCHIVE_VERIFICATION.md).

**Next gate after execution:** inspect branch statistics and image variability; decide if a verified baseline reproduction can be used for subsequent proposed improvements or if the CNN arm needs a corrected independent reconstruction. Every modification must be versioned and evaluated on training-only validation first.
