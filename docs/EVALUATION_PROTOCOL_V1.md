# Evaluation Protocol Lock — Version 1 (2026-10-18)

This protocol was recorded **before running any proposed-model experiments**.

## Research questions

- RQ1: Does a rare-signal-preserving representation improve macro-F1 and rare-category F1 compared with the published SCV-IDS method under the same official test partition?
- RQ2: Do any measured improvements survive comparisons with strong tabular baselines and analyses restricted to feature-nonoverlapping test records?
- RQ3: Are purported spatiotemporal improvements robust to perturbing CSV row order, and which results are legitimately attributable to time?

## Primary evaluation

The official UNSW-NB15 partition consists of **175,341 train** and **82,332 test** examples; label taxonomy: nine attacks plus Normal. The Kaggle mirror's train/test filenames are reversed, so roles must be resolved by reference hashes and exact per-class counts, never names alone.

**Primary reported metric:** ten-class macro-F1 on the untouched official test set. Also report overall accuracy, weighted-F1, balanced accuracy, each of the ten class F1/recall/precision and counts, Normal-to-attack false-positive rate, inference speed and memory. Distinguish any prior-paper summary F1 from macro-F1 unless the published aggregation convention is explicitly verified.

No test-set resampling, deduplication, synthetic balancing, validation-based filtering, early stopping, or threshold selection.

## Model selection and cross-validation

- Candidate model development uses only official train examples.
- Group internal train/validation folds using a SHA-256 canonical signature of the 42 features. Identical feature vectors remain together even if they have different labels. Record seed, folds, indices digest, label proportions and group intersections.
- Split **before** fitting preprocessing or building windows. No overlapping raw events across the validation boundary.
- CatBoost baseline receives typed tabular rows without ViT/CNN encoding. This provides a fair representation-control experiment, not yet a direct claimed reproduction of SCV-IDS.
- For final comparisons, freeze the protocol, hyperparameters, model/checkpoint hash and threshold using training/validation only before the first final test pass.

## Predeclared additional test analyses

Alongside the unchanged official test results, independently score exact-feature-overlap and nonoverlap subsets and by-class results. Include the denominator/support for each subgroup. Overlap groups and label conflicts were identified in an unsupervised/quality audit; these subsets must not be used for iterative model selection.

## Temporal validity

The 42-feature Kaggle CSVs contain no obvious reliable timestamps. Sliding windows formed over rows are an *ordered-record encoding* but should not be claimed to preserve physical temporal continuity. When reproducing SCV-IDS, explicitly include:
- Row-order windows as a **published-protocol approximation**, with original ordering and label-of-latest-row.
- Window size 1 (single row), randomized-within-split order control, and an optional class-preserving permutation control.
- Clear note that any true packet/flow chronological model requires verified raw-timestamp source data.

## Evidence and statistical standards

- Run ≥5 different seeds for key final comparisons after screening.
- Save full configurations, git SHA, data hashes, seeds, package versions, Kaggle hardware identity, metrics/predictions provenance, and model checksum.
- Store small metrics, aggregate confusion matrices, runtime tables and plots in GitHub. Store model checkpoints and large per-row artifacts in versioned Kaggle outputs with SHA-256 references; **never include secrets**.
- Never confuse published SCV-IDS multiclass **accuracy (85.77%)** and its overall reported **F1 (84.59%)** with the explicitly reported **macro-F1 (46.87%)**. The latter is the appropriate primary rare-class comparison.
- No performance claims before executed results are uploaded and independently inspected.
