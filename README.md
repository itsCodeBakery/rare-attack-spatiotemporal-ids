# Rare-Attack-Aware Spatiotemporal Intrusion Detection

[![Research status](https://img.shields.io/badge/status-research%20setup-blue)](#research-status) [![Platform](https://img.shields.io/badge/compute-Kaggle%20GPU-informational)](#execution-environment)

Reproducible research on **rare-attack-sensitive and leakage-audited network intrusion detection** using serialized network traffic representations and deep learning.

> **Status:** Research setup. No model-performance claims have been experimentally verified by this repository.

## Research motivation

Our reference study is J. Chen *et al.*, "**SCV-IDS: CNN-ViT Crossattention on Serialized Traffic Images for Spatiotemporal Intrusion Detection**," *IEEE Internet of Things Journal*, 2026, DOI: [10.1109/JIOT.2026.3695912](https://doi.org/10.1109/JIOT.2026.3695912).

The article reports the following **UNSW-NB15** results:

| Task / metric | Published SCV-IDS result | Interpretation |
|---|---:|---|
| Multiclass accuracy | 85.77% | Overall correct predictions |
| Multiclass F1 | 84.59% | The paper's reported summary F1; averaging convention must be checked before matching |
| Multiclass macro-F1 | 46.87% | Equal weight for each class; our primary rare-attack performance reference |
| Binary accuracy | 96.41% | Separate binary classification task |

The paper also reports **Analysis-class F1 = 0.53%**, pointing to a significant rare-class detection weakness. The study's separate CIC-IDS2017 experiment uses a Wednesday subset and an 80:20 split; its high score must not be described as zero-shot transfer from a model trained on UNSW-NB15.

## Research objective

Our **primary objective** is to improve verified multiclass macro-F1 and rare-class recall/F1 on **the official UNSW-NB15 test set** without making unacceptable trade-offs in overall accuracy, benign false-positive rate, calibration, or inference efficiency.

**Candidate hypotheses (not yet proven)**:
1. Attack-signal-preserving temporal representations may reduce dilution of minority-attack cues.
2. Class-imbalance-aware representation learning or decision mechanisms may improve minority classes without destabilizing majority-class detection.
3. Reliability-aware inference and explicitly evaluated unknown-attack handling may improve practical security performance.

The final model, manuscript title, and novelty claim are subject to evidence from baseline reproduction, literature verification, ablation studies, and repeated-seed experiments.

## Reproducibility and scientific integrity

- **No data leakage:** Split before fitting imputers, scalers, categorical vocabularies, feature selection, class balancing or overlapping windows. Never use test labels for tuning.
- **Respect dataset limits:** Investigate whether trustworthy timestamps / traffic continuity are available before claiming that adjacent rows represent true temporal behavior.
- **Matched comparison:** Keep datasets, classes, split, metric definitions and evaluation scope aligned when comparing with SCV-IDS or competing methods.
- **Repeat experiments:** Record seeds, uncertainty intervals, versions, hardware, runtime and protocol deviations.
- **Publish the evidence:** Store code, scripts, configs, compact result tables, plots, dataset manifests, training logs and links/checksums for large external artifacts.
- **Never fabricate results:** Planned/failed/verified statuses are explicit, and figures must be generated from saved predictions or experiment outputs.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) and [docs/DECISION_LOG.md](docs/DECISION_LOG.md).

## Repository structure

```text
configs/                 Versioned experiment configurations
data/                    Dataset access instructions and manifests (no raw datasets)
docs/                    Scientific protocol and project decisions
experiments/             Run index, configs, notes and ablation specifications
notebooks/               Numbered Kaggle notebooks
scripts/                 Preparation, training, validation and sync tools
src/rareids/             Importable Python package
tests/                   Unit and data-integrity tests
results/                 Metrics, figures, summary tables and compact artifacts
reports/                 Manuscript drafts, figures and publication assets
```

## Execution environment

Experiments will run on **Kaggle** using a declared dataset source and GPU type. Dataset inputs are read-only; generated experiment outputs must be backed up and traceable to a Git commit. The GitHub repository is the **source of truth for text, code, configuration and small artifacts**, not a storage location for raw datasets or multi-GB model checkpoints.

## Research status

**Source-provenance cross-check complete (2026-10-18):** [Notebook 04 comparison](data/manifests/unsw_nb15/publisher_comparison/source_comparison.json), synchronized in [commit 6ee35ad](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/6ee35ada942df078d6c7883b8a377279b7bf349b), confirms that the user-uploaded `dsshayan/unsw-nb15` train and test files are **byte-for-byte identical** to their corresponding `primus11` mirror files. SHA-256 file digests, all rows, column order, and row order match; zero unmatched records. This confirms equality **between the two Kaggle copies**. Independent proof of the files' publisher origin and genuine chronological row order is **not** established.

**Model-control results:** The [CATBOOST-GROUPVAL-001](docs/CATBOOST_GROUPVAL_001_REVIEW.md) internal, group-disjoint validation yielded **55.41% macro-F1** and **82.40% accuracy** on **35,009 validation rows**. These results are **not an official-test score** and must not be compared as a claimed win over SCV-IDS's published test values. Extensive within-split duplicate feature tuples and cross-split label contradictions remain documented in [DATA-001](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/issues/1).

**Preferred source for future Kaggle notebooks:** use `/kaggle/input/datasets/dsshayan/unsw-nb15/UNSW_NB15_training-set.csv` (train, 175,341 rows) and `/kaggle/input/datasets/dsshayan/unsw-nb15/UNSW_NB15_testing-set.csv` (test, 82,332 rows). Source file SHA-256 hashes are mandatory checks. Earlier experiments on the `primus11` mirror retain their original provenance and need **no rerun solely to switch to the identical dataset copy**.

**Step 7 complete — verified paper-aligned validation:** [Recovered result commit d3dcbe9](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/d3dcbe98a3ddfb395058b3c23b3e47a8be87c21c) stores 50 epochs of recorded history, **82.74% accuracy and 55.87% ten-class macro-F1** on **35,005 internal validation windows** from the **best epoch 41**. Both numbers were independently recomputed from the saved confusion matrix. **This is NOT the unchanged official-test result**, so no claim of beating the original SCV-IDS paper is justified. See the [independent audited report](docs/SCV_IDS_001_RESULTS_AUDIT.md).

**Reproducible figures:** 28 figures × three formats (PNG, SVG, PDF) = **84 committed figure files**, all with declared CSV source tables and SHA-256 manifests. [Figure scripts](scripts/plot_scv_figures.py) regenerate images from nine versioned numeric `plot_data/*.csv` sources **without rerunning the network**. PCA plots replace the failed t-SNE stage and explicitly flag the CNN feature branch's **zero measured variance in the sampled validation representations**. This requires frozen-checkpoint diagnosis in [DIAG-001](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/issues/4), not an assumption that the complete architecture is functioning normally.

**Full reproducibility backup now verified on GitHub:** the complete 130.05 MiB archive and checksum are present as assets in a **draft Release**, with GitHub-reported SHA-256 identical to the Kaggle-computed archive digest. See [verified archive and restoration instructions](docs/SCV_IDS_ARCHIVE_VERIFICATION.md). The draft is not public and should not be published without reviewing licensing/sensitive files; an additional private/local backup is advisable. The missing original training-environment freeze remains documented transparently. **Next research gate:** [DIAG-001 — frozen-CNN feature-collapse investigation](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/issues/4) with the saved checkpoint; no retraining yet.

**Step 8 prepared — frozen CNN/attention diagnostic:** [Notebook 07](notebooks/07_scv_cnn_diagnostic.ipynb) and [DIAG-001 protocol](docs/SCV_CNN_DIAGNOSTIC_PROTOCOL.md) use the verified checkpoint archive with **no retraining** and no official-test access. A strict replay check must match the saved original validation probabilities and CNN embeddings before any diagnostic interpretation. The planned output is 12 numeric-source plots in PNG/SVG/PDF. If Kaggle has lost its old session, the notebook automatically restores the SHA-256-pinned archive from the draft Release with authenticated access. Results are **pending execution**.

**Step 8 observed diagnosis — confirmed CNN zero output:** [SCV-CNN-DIAG-001 synchronized evidence](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/2c18b88cba408cd8eb505436957af6d92f7ec143) and [independently audited diagnosis](docs/SCV_CNN_DIAGNOSTIC_RESULTS_AUDIT.md) found the last CNN ReLU produces **all-zero features across 1,600 frozen internal-validation windows**. CNN masking does not change logits or predictions; ViT masking severely degrades performance. The **12 diagnostic figures × PNG/SVG/PDF**, 9 input CSVs and manifest are committed. Strict frozen prediction replay succeeded. This diagnosis concerns **our independent reconstruction checkpoint**, not the original paper's author code. **Next:** matched-fidelity repaired CNN versus ViT-only controls, window-order ablations and multi-seed validation. No official-test performance claims or novel model gain yet.

**CTRL-001 genuine ViT-only control V1 verified:** [Observed real Kaggle commit 8b7baea](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/8b7baeada660b0203f48237b6f8ed9f793c55b77) records **57.4409% macro-F1 and 83.2367% accuracy** on the *same 35,005 internal-validation window labels* as V0, compared with V0's 55.8664% macro-F1 and 82.7396% accuracy (**+1.5746 / +0.4971 percentage points**). **Rare-class tradeoffs:** DoS and Backdoor improve; Analysis worsens. [Audited 10-class comparison](docs/CTRL_VIT_001_RESULTS_AUDIT.md) and [numeric source CSV](results/comparisons/V0_CNNViT_vs_V1_ViTOnly/classwise_deltas.csv) provide exact per-class support and F1 deltas. **42 PNG/SVG/PDF figures and seven CSV tables** are versioned. New ViT weights/predictions are preserved separately in a [checksum-verified unpublished draft Release](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-7be70698aa7f3dd7a5de). This is one-seed internal validation, **not** published SCV-IDS official-test outperformance. **Next control:** healthy CNN–ViT V2.

**CTRL-001 V2 minimal CNN activation repair is READY, not yet trained:** [Kaggle Notebook 09](notebooks/09_ctrl002_cnnvit_leaky.ipynb), [full preregistered protocol](docs/CTRL002_CNN_REPAIR_PROTOCOL.md), and [versioned V2 model definition](src/rareids/v2_model.py). Versus V0, **only** the final CNN ReLU changes to `LeakyReLU(0.1)`; training remains seed=42, 50 epochs, same original archived train/validation split and identical 35,005 internal validation targets. Every epoch logs CNN token nonzero rate, between-example variance and zero-CNN masking effects; best-epoch CNN activation health uses two predeclared thresholds, with failures preserved. Training/plotting separated, checkpoints frozen, **18 figures × PNG/SVG/PDF** with eight numeric CSV sources, and new private draft model Release archival planned. **V2 results are pending; no novelty, official-test, or significance claim.**

**V2 negative result independently verified; focus shifted to competitive metrics:** 50-epoch repaired CNN–ViT best macro-F1 **55.2464%**, accuracy **82.4654%**, below existing 57.4409% ViT-only. CNN tokens are nonzero but numerically constant across a frozen 512-window sample and CNN masking changes zero predictions. [Complete V2 verification](docs/CTRL002_V2_VERIFICATION_AND_PIVOT.md), [V2 actual measured results](results/CTRL-CNNVIT-V2-S42/metrics_validation.json), [V2 checksum-verified draft archive](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-b86560b60bc745c6e454). **Do not conduct additional incremental CNN repairs.** [Next one-cell rapid LightGBM benchmark notebook 10](notebooks/10_fast_tabular_benchmarks.ipynb) trains two real 42-feature classifiers on the EXACT 35,005 original internal-validation target labels; **results pending, no promises**. Preserve original official test untouched.

**Outcome targets, not achieved results:** SCV-IDS published **85.77% multiclass accuracy** and **46.87% macro-F1**. We aim to exceed both under a fair protocol, with materially better rare-class recall/F1, full ablations, and repeated-seed uncertainty reporting. No guaranteed improvement or unsupported claim is permitted.

## Citation and rights

The reference article is cited bibliographically; its publisher PDF is **not redistributed** here. Dataset and third-party software licenses must be checked before public redistribution.

## Maintainer

[itsCodeBakery](https://github.com/itsCodeBakery)
