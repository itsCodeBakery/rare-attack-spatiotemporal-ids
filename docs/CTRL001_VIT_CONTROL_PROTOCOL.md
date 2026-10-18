# CTRL-001 / V1 — First matched-target control: genuine ViT-only (seed 42)

**Pre-registered:** 2026-10-18.  
**Status:** Full Kaggle implementation ready; **GPU training and actual result pending**.  
**Notebook:** [notebooks/08_ctrl001_vit_only.ipynb](../notebooks/08_ctrl001_vit_only.ipynb).  
**Experiment:** `CTRL-VIT-001-S42` — its own separate checkpoint and publication-figure namespace.

## Motivating verified baseline observation

The [original SCV independent reconstruction](SCV_IDS_001_RESULTS_AUDIT.md) trained for 50 epochs and obtained **82.7396% accuracy**, **55.8664% ten-class macro-F1** using its best epoch 41 on **35,005 internal validation windows**. A [subsequent frozen-checkpoint diagnosis](SCV_CNN_DIAGNOSTIC_RESULTS_AUDIT.md) found the **final CNN ReLU output identically zero** on 1,600 training-derived validation windows, while ViT features varied. Inference-only CNN zeroing did not change logits on those diagnostic samples. These findings concern *our independent reconstruction*, not confirmed authors' code.

Before proposing anything new, test a **real trained standalone ViT**, rather than mistakenly treating zero-masked CNN-ViT logits as a standalone ViT baseline.

## Frozen split and experimental target parity

- Dataset: designate only `/kaggle/input/datasets/dsshayan/unsw-nb15/UNSW_NB15_training-set.csv` (175,341 records; verified SHA-256 `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`).
- Split reference: the **same archived original** `split_indices.npz`, cryptographic SHA-256 `01f6070e836317b09ba5f8ef662dba748ec19f9b7f6e211e4a5215d35bc4409c`. Group-disjoint original fold; 140,332 development training rows and 35,009 validation rows; **not regenerated or reselected**.
- Original validation prediction file SHA-256 `54f6aa6cb50c3c6a1533623e1c7b619f38dfa7ccbac8ce03cb9392e13e617612`; its 35,005 target labels and window-end indices are the exact required reference. No differences permitted.
- Original saved fitted `preprocessing.json` categorical vocabularies (proto/service/state) and 42-feature training-only MinMax parameters are reused **without refitting**. Quantization is the same zero-padded 5×42→16×16×RGB algorithm.
- Keep the original row ordering inside the **training-derived** development train/validation partitions. Windows cannot cross partitions, but some source rows were removed and row-order adjacency is *not proven chronological*.
- **Official designated test CSV is not read, fitted, validated, or used for model selection.**

## The control architecture (V1)

`ControlledViTOnly` reuses **the original `ViTEncoder` Python class** from the SCV reconstruction, so patch size 8×8, 4 transformer blocks, 8 attention heads, embedding width 128 and MLP hidden width 256 are unchanged. It mean-pools the ViT patch tokens and uses `128→128 (GELU/dropout 0.2)→10` as the classifier.

**Differences from V0**: V1 has **no CNN, no bidirectional crossattention and no fusion module**. It is newly trained from seed 42, not the pre-existing checkpoint with its CNN removed at inference. The classifier's input representation differs, so this is a *controlled architecture comparison*, not a one-parameter ablation. Compute and report parameter count and time.

## Training pre-registration

| Item | Locked configuration |
|---|---|
| Experiment | CTRL-VIT-001-S42 |
| Seed | 42 |
| Optimization | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Loss | 10-class multiclass focal, γ=2, no class α |
| Epochs | **50**; no premature early stopping |
| Batch size | 64 |
| Precision | Float32, not mixed |
| Selection | Maximum internal-validation macro-F1; ties lowest validation focal loss |
| Shuffle | Training DataLoader shuffle with checkpointed generator state |
| Checkpoints | Best, last resumable **every epoch**, and every five epochs |
| Targets | Same class order and **35,005 labels in identical order** |
| Outcome | Macro-F1, weighted-F1, balanced accuracy, 10-class support/recall/F1, Normal false-positive rate, DoS→Exploits, Backdoor→Exploits, training time, memory, parameter count |

**Only a single seed** is launched in this first screening step. Repeated seeds with uncertainty intervals are required before drawing a statistical conclusion.

## Separation of experimental stages and preservation

**Cell 1:** clone repository and safely restore the previously verified 130.05 MiB SCV archive from an unpublished draft Release if needed, checking archive and individual-file SHA-256.  
**Cell 2:** train this new standalone ViT with Kaggle GPU and nested progress bars, saving every epoch. When its status says `completed=true`, reexecuting the cell **does not retrain**.  
**Cell 3:** derive metrics, confusion, calibration and ROC/PR from **the frozen best-checkpoint probabilities**; generate all figures. No model training or inference here. An error at this stage does not invalidate the 50-epoch run.  
**Cell 4:** archive every new checkpoint, full 35,005×10 probability matrix, config, metadata, numeric tables and figures into a **new private draft GitHub Release**, with SHA-256; separately synchronize **only aggregate reports and plot source CSVs** and 42 image files into public GitHub.

Fourteen versioned figures (14×3=**42 exports**) include train/validation focal loss, accuracy, macro-F1, weighted/macro comparison, epoch runtime, original-split class support, ranked per-class F1, precision/recall, confusion counts and normalized confusion, reliability, ROC and PR curves, and true vs predicted class proportions. SVG and PDF preserve editable text/vector paths. Every chart has a signed source-table path in `figure_manifest.json` and is rebuildable from `scripts/plot_ctrl001_vit.py` **without GPU**.

**Retention limitation:** A successful GitHub public sync of small plots does not itself preserve the weight files. The notebook must reach the private draft Release upload or the entire checkpoint tree must be retained in a saved Kaggle version.

## Interpretation gate

The V0 SCV reconstructed checkpoint had a **zero-output CNN** on the diagnostic sample. If V1 equals it under the same labels, this provides useful evidence that the CNN branch was superfluous in this *implementation*. If V1 is worse or better, that must be evaluated with repeated seeds, same targets, optimization-budget considerations, and potential checkpoint/activation diagnostics before any claim.

**Next control after this one:** V2 healthy-CNN corrected fusion model, activation-health gate, then window-size-one and fixed-order-permutation controls. A rare-attack-aware novel method comes **later** and must be compared with V1, V2, and strong tabular classifiers under common targets. No claim of surpassing published SCV-IDS **85.77% accuracy/46.87% macro-F1** until the **locked official test** is appropriately evaluated.
