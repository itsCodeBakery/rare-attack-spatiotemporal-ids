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

**Phase 1 — official-split dataset audit verified and synced.** Kaggle manifests confirm the official class distributions, SHA-256 file hashes, 175,341 train / 82,332 test records and 42 features. The Kaggle mirror's filenames are reversed. **Phase 2 — exact-value feature-overlap and label-consistency audit prepared, awaiting execution.** No model has been trained.

**Next gate:** execute [02_exact_overlap_audit.ipynb](notebooks/02_exact_overlap_audit.ipynb) in Kaggle and synchronize its aggregate SHA-256 equality/label-consistency report. Do not create temporal windows or start SCV-IDS training until overlap and temporal-order limitations are documented. The [Step 3 source audit](data/manifests/unsw_nb15/audit_summary.json) remains immutable evidence of the original Kaggle partition mapping.

## Citation and rights

The reference article is cited bibliographically; its publisher PDF is **not redistributed** here. Dataset and third-party software licenses must be checked before public redistribution.

## Maintainer

[itsCodeBakery](https://github.com/itsCodeBakery)
