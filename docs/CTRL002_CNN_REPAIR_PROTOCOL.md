# CTRL-CNNVIT-V2-S42 — pre-registered minimal CNN activation repair

**Date:** 2026-10-18. **Status:** code and Kaggle notebook ready, **no V2 GPU training has yet run**.

## Why this control is necessary

The original paper-aligned independent SCV reconstruction V0 achieved 55.8664% ten-class macro-F1 on 35,005 *training-derived group-disjoint validation windows*, but its final CNN ReLU output was all zero on a 1,600-window frozen diagnostic sample; CNN masking had zero logit effect there. A genuine separately trained ViT-only V1 achieved 57.4409% macro-F1 on **exactly the same target labels**. That result is a useful baseline finding, **not** an official published-paper outperformance or new method.

We now test **one and only one architectural change** to V0: final CNN nonlinearity `cnn.layers[11]` changes from `ReLU(x)=max(0,x)` to `LeakyReLU(x)=max(0,x)+0.1 min(0,x)`. This lets negative preactivations transmit nonzero information and directly probes whether final hard clamping caused the CNN dead-path observation. It does not guarantee good signal or more accurate predictions. We do not modify CNN conv weights/widths, ViT, crossattention, fusion, classifier, image serialization, focal gamma, optimizer or validation split. The repaired model is trained from scratch, never patched into V0's frozen weights. This is **engineering baseline repair, not methodological novelty**.

The entire model definition is versioned in [src/rareids/v2_model.py](../src/rareids/v2_model.py), the configuration is [configs/ctrl002_cnnvit_leaky_seed42.json](../configs/ctrl002_cnnvit_leaky_seed42.json), and [Notebook 09](../notebooks/09_ctrl002_cnnvit_leaky.ipynb) is the Kaggle entry point.

## Fair comparison controls

| Property | V0 independent SCV reconstruction | V1 standalone ViT | V2 CNN repair |
|---|---|---|---|
| ViT depth, width, heads, patches | 4 layers, 128, 8, 8×8 | Same ViTEncoder | **Same as V0** |
| CNN channels | 32, 32, 64, 64, 128 | No CNN | **Same as V0** |
| CNN final activation | ReLU | N/A | **LeakyReLU(0.1)** |
| Bidirectional crossattention/fusion/classifier | Original independent reconstruction | No fusion / independent classifier | **Exactly V0 structure** |
| Source dataset | Designated train 175,341 rows | Same | **Same** |
| Original group-disjoint development train/validation rows | 140,332 / 35,009 | Same saved indices | **Same saved indices** |
| Five-row RGB window train/validation targets | 140,328 / 35,005 | Identical label order | **Identical label order** |
| Train-fitted categorical/MinMax preprocessing | Yes | Replay original | **Replay original** |
| Seed/epochs/batch | 42 / 50 / 64 | Same | **Same** |
| Optimizer/loss | AdamW lr .001 wd .001, focal γ=2 | Same | **Same** |
| Selection | Internal validation maximum macro-F1; tie lowest loss | Same | **Same** |
| Official test | NOT used | NOT used | **FORBIDDEN** |

Original training CSV `bec7dd5ec88dc2a0ccc7a07879d338395ed742175fd0339e07dfe0648fa`; split indices `01f6070e836317b09ba5f8ef662dba748ec19f9b7f6e211e4a5215d35bc4409c`; original frozen validation predictions `54f6aa6cb50c3c6a1533623e1c7b619f38dfa7ccbac8ce03cb9392e13e617612`. Mismatch aborts, rather than using a different split. Only original training CSV read.

**Caveat:** five-row windows preserve original retained CSV order within each partition; original row adjacency is not independently verified temporal continuity, and partition filtering causes gaps. We must separately evaluate window=1 and deterministic row permutations.

## Frozen health gate (declared before training)

Every epoch evaluates a **fixed, class-aware 512-window subsample** of *internal validation only*, seed 42, with at least up to 24 examples drawn per class (Worms naturally limited to its support of 21). It records:
- fraction of all CNN tokens with absolute activation above 1e-12;
- mean per-token feature standard deviation across these examples, constant-dimension fraction and pooled-dimension variance;
- mean logit L2 difference and prediction disagreement when CNN token values are replaced by zero at inference.

The **best checkpoint** passes the descriptive CNN activation-health gate only if:
1. CNN nonzero-token fraction is at least **0.05**, and
2. CNN mean between-example token-feature std is at least **1e-5**.

If the gate fails, we **still preserve/publish the measured run** with explicit failure. A passed gate is not proof of that CNN branch increasing F1; inference-only zeroing is an out-of-distribution diagnostic. We will compare per-class F1, especially **DoS, Backdoor, Analysis, Worms**, and absolute DoS→Exploits confusion against V0 and V1 using identical labels.

## Reproducibility and figures

Kaggle [Notebook 09](../notebooks/09_ctrl002_cnnvit_leaky.ipynb) separates:
1. clone repo / pinned reference archive restore with SHA-256 and verified source;
2. new 50-epoch V2 GPU training, nested progress bars, epoch-wise source activation logs, best/last/five-epoch full checkpoints and original training Python package freeze;
3. **no-training finalizer** from saved probabilities: 10×10 confusion, 10-class precision/recall/F1, accuracy, macro/weighted-F1, balanced accuracy, reliability, ROC/PR numeric source data, activation-health status, V0/V1 reference numbers;
4. **full private checkpoint/probability/figures archive** uploaded to a new checksum-verified **draft GitHub Release**, plus public GitHub atomic sync of only aggregate CSVs, metadata and figures.

The [plotting script](../scripts/plot_ctrl002_cnnvit.py) builds **18 figures** × **PNG/SVG/PDF** (**54 files**), with 8 versioned numeric CSV plot sources: original 14 standard V1-style classification and learning-curve figures plus 4 CNN health plots across 50 epochs (nonzero-token fraction, token variation, CNN-mask logit changes, CNN-mask disagreement). A figure-only restyle does not repeat any GPU training. Numeric output, configs, row-lineage hashes, source commit, private prediction hashes and archive verification support later scholarly edits.

**Statistical limit:** initial V2 uses **one pre-registered seed, 42**, for controlled screen. Do not claim statistical significance or official-test paper superiority. Repeat V1/V2 across at least three and preferably five seeds only after first health/evaluation check. Do not invent performance or figure outcomes before Kaggle execution.

## Next user action

Run only [Notebook 09](../notebooks/09_ctrl002_cnnvit_leaky.ipynb) in Kaggle with dataset `dsshayan/unsw-nb15`, GPU, Internet and private existing `pushRare` secret. Then provide BOTH `VERIFIED CTRL-V2 SYNC COMMIT:` and `CTRL-V2 PRIVATE RELEASE VERIFIED:` URL outputs. **Do not rerun training if finalization/upload encounters an error.** Preserve the original 50-epoch V0 and V1 archives unchanged.
