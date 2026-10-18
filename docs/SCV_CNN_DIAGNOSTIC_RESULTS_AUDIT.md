# DIAG-001 — Frozen CNN-branch diagnostic, independently audited

**Audit date:** 2026-10-18. **Source:** [Kaggle-synchronized observed results commit 2c18b88](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/2c18b88cba408cd8eb505436957af6d92f7ec143).

**Study scope:** 1,600 fixed sampled windows drawn exclusively from the original **training-derived group-disjoint internal validation partition** of UNSW-NB15, using the *unchanged epoch-41 SCV-IDS paper-aligned independent-reconstruction checkpoint*. This is **not official-test performance** and does not certify implementation identity with Chen et al.'s author model.

## Provenance and numerical validation

- Frozen checkpoint SHA-256: `1ff947f795162b747115fee6e3131eb10fd3955445a3040902a274fa71457c65`. It matches the full research archive previously independently checked in the GitHub draft Release.
- Designated train CSV SHA-256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.
- Original preprocessor (fitted on development-train only) reused from saved metadata; original validation row/window lineage checked; **official test not accessed**.
- Checkpoint replay against the previously saved full-window validation probabilities: **100.00% matching argmax** over 1,600 samples and **max absolute probability difference 1.1175870895e-08**.
- Frozen CNN pooled embedding replay max absolute vector difference: **0.0** (1,101 reference examples).
- Recomputed from the committed [sampled confusion matrix](../results/SCV-CNN-DIAG-001/sampled_confusion.csv): **1,079 correct of 1,600**, accuracy **67.4375%**, macro-F1 **56.8810753%**. These exactly agree with [ablation_overall.csv](../results/SCV-CNN-DIAG-001/ablation_overall.csv). This sample is enriched for rare classes, so its accuracy is **not directly comparable** to the full 35,005-window development validation result (82.7396%).
- Generated **12 distinct diagnostic figures**, exported in all **three formats (PNG/SVG/PDF)**, totaling **36 versioned files**, all defined using **9 committed aggregate numeric CSVs**. The figure manifest records every exported file's source data, filename and SHA-256. The archive has independent binary checkpoint/prediction retention.
- Training or optimizer operations: **none**. The record is forward-pass/frozen-model evidence.

## Primary result: CNN branch dead on the sampled windows

| Diagnostic quantity | Measured result |
|---|---:|
| CNN pooled mean standard deviation across examples, 128 dims | **0.0** |
| CNN full spatial tokens standard deviation across examples, 2,048 dims | **0.0** |
| CNN output mean vector norm | **0.0** |
| CNN final Conv-layer (Sequential index 10) spatial variation | 0.00125525 (small but nonzero) |
| CNN final ReLU (Sequential index 11) exact zero fraction | **100%** |
| CNN final ReLU per-channel inter-example SD | **0.0** |
| ViT pooled mean feature SD | **0.847953** |
| Fused representation mean feature SD | **0.415041** |
| CNN representation change after horizontal image flip | **0.0** L2 |
| ViT representation change after horizontal image flip | **28.8431** mean L2 |

Conv/ReLU stage statistics reveal progressively sparse activations, culminating in an **all-zero final CNN ReLU output** for the sampled 1,600 windows. This is a *specific failure in the trained checkpoint*; neither the CNN pooled features **nor the spatial CNN tokens** retain input-specific information. The final pre-ReLU convolution has small nonzero spatial variation, so the final ReLU removes the remaining signal in this sample. Earlier ReLUs are already near-saturated. This does **not** prove Chen et al.'s original CNN branch fails or that all inputs would be identical; exact cause (optimization, activation, initialization, scale, fusion behavior) remains a testable hypothesis.

## Frozen branch intervention (not causal training ablation)

| Inference intervention | Diagnostic subset accuracy | Diagnostic subset macro-F1 | Prediction disagreement with original |
|---|---:|---:|---:|
| Unmodified checkpoint | **67.4375%** | **56.8811%** | 0% |
| Replace CNN tokens with zeros | **67.4375%** | **56.8811%** | 0% |
| Replace ViT tokens with zeros | **14.6875%** | **2.5613%** | 60.75% |
| Replace both branches with zeros | **14.6875%** | **2.5613%** | 60.75% |

**CNN masking produced zero measured logit L2 difference and no prediction changes** in the 1,600 diagnostic windows, consistent with original CNN tokens already being exactly zero. ViT masking significantly changed logits (mean L2 22.03775). This supports the conclusion that the **learned predictor at this checkpoint is ViT-driven** for these windows, notwithstanding the presence of parameterized crossattention modules. Note the `vit_to_cnn_attention` branch norm is nonzero because attention projections have parameters/biases even when CNN keys/values are all zero; this is not instance-dependent CNN feature evidence.

**Do not present masking as a new architecture's performance**: zeroing branches is an out-of-distribution inference-time diagnostic.

## Rare attacks still fail

Within this **rare-enriched diagnostic sample**, Backdoor recall is **6/123 = 4.88%**, DoS recall **21/156 = 13.46%**, Analysis **29/124 = 23.39%**, and Worms **3/21 = 14.29%**. Misclassifying these attacks as Exploits remains a primary challenge. Full earlier validation DoS→Exploits is 2,130/2,509; do not substitute the sampled subset count for full validation.

## What we have *not* established

1. **We have not outperformed the published SCV-IDS paper.** Its reported 85.77% multiclass accuracy and 46.87% macro-F1 were measured in a different evaluation context. Our trained checkpoint was evaluated only on a training-derived internal-validation fold.
2. No 2026 journal-worthy methodological novelty is demonstrated by diagnosing an all-zero CNN branch alone; it is primarily a *baseline-implementation finding*. A corrected independent baseline is necessary before crediting any proposed novelty.
3. Original CSV records lack a verified timestamp/flow-session sequence. Five-row windows of filtered train/validation rows are **row-order constructions, not verified temporal neighborhoods**. This must be addressed by row-permutation and window=1 controls and candid framing.
4. The internal validation contains an original train-derived group-disjoint split, but official designated train/test duplicate signatures and occasional conflicting class labels have already been documented. Never alter the official benchmark test set to improve published scores.
5. A single seed and selective sampled diagnostic cannot establish confidence intervals, robustness, causal contribution or generalization. Do not interpret 56.88% sampled F1 as an improvement over 55.87% complete-validation F1.

## Scientific decision and next pre-registered experiment

**Do not overwrite or retrain the completed checkpoint.** Keep `SCV-IDS-REPRO-001` permanently labeled as the audited ViT-driven/zero-output-CNN independent reconstruction. The full archive, observed figure sources, and diagnostic CSVs remain reproducible.

Next prepare an **architecture-fidelity repair experiment** as a *new* controlled baseline, not a claimed new method: compare (i) frozen published-reconstruction implementation, (ii) corrected CNN activation path (e.g. non-saturating final activation or residual normalization), and (iii) ViT-only, trained using **exactly matched original train-derived folds**, explicit split/window lineage and consistent budget. Predefine activation health metrics, validation macro-F1, per-class recall (Analysis, Backdoor, DoS, Worms), parameter/FLOP/runtime cost, and at least 3–5 seeds for promising variants. Additionally run row-permutation/window=1 controls to establish whether serialized adjacency contributes information. These designs must be explicitly distinct from the paper's original method, with no official-test use until model selection is frozen.

Only after the corrected baseline and controls are proven should we claim and test a **novel rare-attack-aware contribution** (e.g. supervised class-conditional margins/prototype regularization and confusion-aware objectives). Such a proposal requires comparison against strong nonimage tabular controls and controlled ablations, not decorative complexity. GitHub issue [DIAG-001](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/issues/4) logs the evidence.
