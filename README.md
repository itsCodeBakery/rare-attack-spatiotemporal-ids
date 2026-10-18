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

**Step 7 ready — SCV-IDS CNN–ViT paper-aligned reproduction:** [Notebook 05](notebooks/05_scv_ids_baseline.ipynb), [declared experiment configuration](configs/step07_scv_ids_repro_001.json), [source fidelity and figure-reproduction protocol](docs/SCV_IDS_REPRODUCTION_PROTOCOL.md). Implemented five-row traffic serialization, 16×16 RGB, ViT, CNN and bidirectional crossattention, train-only MinMax/index encoding, focal loss, AdamW, 50 epochs, exact-feature-group-disjoint internal validation, tqdm progress bars and resumable checkpoints. **Not yet executed on Kaggle GPU; no SCV validation or official-test results claimed.**

**Preservation policy:** The notebook creates 28 **data-backed** diagnostic figures, each exported as PNG/SVG/PDF with independent numeric CSV plot sources, reproducible plotting code, SHA-256 manifest and frozen dependency list. GitHub automatically receives small metrics, plot source CSVs, three figure formats and full provenance. Large model checkpoints, original feature rows, split indices and individual validation probabilities remain in the **saved versioned Kaggle outputs**, with GitHub linking their SHA-256 hashes. Figures can be re-styled from saved inputs **without model retraining**. The official test set remains locked; CSV row order is not established chronological.

**Outcome targets, not achieved results:** SCV-IDS published **85.77% multiclass accuracy** and **46.87% macro-F1**. We aim to exceed both under a fair protocol, with materially better rare-class recall/F1, full ablations, and repeated-seed uncertainty reporting. No guaranteed improvement or unsupported claim is permitted.

## Citation and rights

The reference article is cited bibliographically; its publisher PDF is **not redistributed** here. Dataset and third-party software licenses must be checked before public redistribution.

## Maintainer

[itsCodeBakery](https://github.com/itsCodeBakery)
