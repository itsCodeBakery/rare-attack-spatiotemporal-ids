# SCV-IDS-REPRO-001 — audited experiment results (2026-10-18)

**Audit evidence:** [Kaggle-synchronized observed outputs](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/d3dcbe98a3ddfb395058b3c23b3e47a8be87c21c). This study is **paper-aligned independent reproduction on an internal group-disjoint development validation fold**, not a verified official author-code reproduction and **not** an official UNSW-NB15 test result.

## Numeric checks — passed

Results were independently recomputed from [observed confusion matrix](../results/SCV-IDS-REPRO-001/plot_data/confusion_validation.csv) and compared with [metrics_validation.json](../results/SCV-IDS-REPRO-001/metrics_validation.json), [per_class_validation.csv](../results/SCV-IDS-REPRO-001/plot_data/per_class_validation.csv), and [50-epoch training history](../results/SCV-IDS-REPRO-001/plot_data/training_history.csv):

| Quantity | Verified |
|---|---:|
| Validation predictions | 35,005 |
| Correct predictions | 28,963 |
| Validation accuracy | 82.739609% |
| Macro-F1 | 55.866391% |
| Weighted-F1 | 81.208883% |
| Balanced accuracy | 54.897358% |
| Normal → any attack rate (false positive) | 8.727371% |
| Best checkpoint epoch | 41 of 50 |
| Final epoch (not checkpoint selected) macro-F1 | 54.320066% |
| Effective training windows | 140,328 |
| Internal validation windows | 35,005 |
| Sum of logged epoch times | 2,080.99 seconds |
| Checkpoint binary files referenced in manifest | 12 |

All five measures (accuracy, macro-F1, weighted-F1, balanced accuracy, per-class precision/recall/F1) agree exactly with the saved aggregate metrics to numerical precision. The best epoch also matches the saved training trace, and the manifest states the designated official test was not used for fitting or inference.

## Per-class validation

| Class | Support | Recall (%) | F1 (%) |
|---|---:|---:|---:|
| Analysis | 428 | 18.46 | 28.78 |
| Backdoor | 350 | 5.43 | 9.03 |
| DoS | 2,509 | 11.92 | 18.54 |
| Exploits | 6,684 | 91.94 | 73.29 |
| Fuzzers | 3,679 | 81.73 | 77.21 |
| Generic | 7,894 | 97.44 | 98.68 |
| Normal | 11,103 | 91.27 | 94.37 |
| Reconnaissance | 2,116 | 67.72 | 78.87 |
| Shellcode | 221 | 68.78 | 57.69 |
| Worms | 21 | 14.29 | 22.22 |

### Observed confusion bottlenecks

- **DoS → Exploits:** 2,130 of 2,509 DoS cases (84.89%) mislabeled as Exploits. Only 299 DoS cases correctly detected.
- **Backdoor → Exploits:** 286 of 350 Backdoor cases (81.71%). Only 19 correctly detected.
- **Analysis → Exploits:** 289 of 428 Analysis cases (67.52%). Only 79 correctly detected.
- **Worms → Exploits:** 14 of 21 Worms cases. Only 3 correctly detected.
- **Normal → Fuzzers:** 843 of 11,103 Normal cases. False-positive rate across all attack categories: 8.73%.

These strongly favor investigating *class confusion and feature ambiguity*, not simply collecting more global-accuracy gains.

### Unexpected architectural diagnostic

[projection_diagnostics.json](../results/SCV-IDS-REPRO-001/projection_diagnostics.json) records **zero informative dimensions out of 128** for the sampled **CNN-only** feature vectors: all sampled outputs are numerically constant at the precision examined. Raw input, ViT and fused features show measurable variation.

This does **not** prove the CNN encoder is globally dead over all training or test samples, but it raises a **high-priority reproducibility and architecture-fidelity question**. Before proposing crossattention improvements, instrument the frozen best checkpoint on a broader sample: per-stage activation/gradient variance, ReLU sparsity, attention input/output variance, CNN branch ablation, and class-conditional results. Avoid changing the checkpoint. A constant branch cannot provide instance-varying local signal to fusion on that sampled subset.

### Baseline comparison limits

- CatBoost control had **55.41% macro-F1 on 35,009 single-row** training-derived validation records; SCV-IDS here has **55.87% macro-F1 on 35,005 five-row windows**. The label/window evaluation sets differ; the 0.46-point numeric gap is **not** a valid paired or statistically meaningful superiority claim.
- The source SCV-IDS article reports **85.77% multiclass accuracy** and **46.87% macro-F1** on its reported benchmark test evaluation. This run's 82.74%/55.87% are on a different *internal validation protocol*, so there is **no established outperformance**, despite macro-F1 being numerically higher.
- This independent reproduction uses reconstruction choices for quantization, focal-loss gamma/alpha and crossattention projection/classifier details; no verified authors' implementation equivalence.
- Five-row windows are from the surviving CSV rows in train or validation and are not verified chronological flow windows. The designated official test remains locked.

## Figure audit — passed at manifest level

- **28 distinct figure stems** × **PNG/SVG/PDF** = **84 exported figure entries**, and these paths appear in the synchronized commit.
- **9 observed plotting CSVs**, their complete filenames and SHA-256 values appear consistently in `figure_manifest.json` and `artifact_manifest.json`.
- All 28 figure definitions reference recorded source table names; every referenced figure export has a recorded SHA-256, byte length, source table(s) and caption.
- Figures 25–28 are **deterministic PCA**, not t-SNE. Figure 26 flags the collapsed CNN branch as uninformative instead of inventing a scatterplot.
- This audit inspected exported GitHub paths and recorded hashes; **it did not independently download and rehash every PNG, SVG and PDF binary**. Commit integrity provides versioning, while per-export SHA-256 verification against raw bytes remains possible from the saved Kaggle archive.

### Original source provenance and preservation

- The training file's recorded SHA-256 matches the previously audited designated UNSW training dataset.
- The original `environment.txt` is missing. The observed recovery software versions are saved in `recovery_environment.json`, explicitly **not** attributed to the original training. The exact original training code commit is *not* recoverable from this file, although the repository retains the relevant historical commits.
- The manifest records SHA-256 for 12 PyTorch checkpoint binaries, prediction arrays, split indices and embedding arrays, but **none of those large binaries is stored on GitHub**.
- **Backup status unconfirmed:** successful GitHub figure sync does not guarantee the full archive `/kaggle/working/SCV-IDS-REPRO-001_FULL_ARTIFACTS.tar` was downloaded or uploaded to persistent private Kaggle storage.

## Next gate (one action)

**Preserve the full Kaggle archive outside the temporary session** and verify its checksum. Only after the user confirms persistent backup should we launch a no-training CNN-branch diagnostic using the saved best checkpoint. This protects 35 minutes of completed training and all figure-reproduction assets.
