# Exact-Feature Overlap: Audited Findings (2026-10-18)

**Status:** Read-only data audit complete; no model has been trained. This is **not** an intrusion-detection performance result.

## Provenance

- Source: Kaggle user-provided mirror \`primus11/unsw-nb15-dataset\`.
- Original file names reverse the officially designated roles: \`unsw_test.csv\` = **175,341 training records**, \`unsw_train.csv\` = **82,332 testing records**.
- The SHA-256 file hashes and all ten class distributions matched the official benchmark definitions.
- [Initial synchronized audit](../data/manifests/unsw_nb15/audit_summary.json)
- [Exact-value overlap audit](../data/manifests/unsw_nb15/exact_overlap/overlap_summary.json) and [per-class table](../data/manifests/unsw_nb15/exact_overlap/per_class_overlap.csv)
- Audit evidence committed by Kaggle: [3c99950](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/3c99950b5d44cdac62dfc2b9abc80890ddd02571).

## What was compared?

SHA-256 digests were computed from **type-preserving canonical JSON of all 42 parsed feature values** in their column order. The \`id\`, \`attack_cat\`, and \`label\` columns were excluded from feature signatures. Individual traffic rows and digests were **not** published.

This verifies equality of *parsed feature tuples* and is **not** evidence that equal-valued examples came from the same captured network event. There are no reliable timestamps/session IDs in the distributed CSV feature schema.

## Quantitative findings

| Diagnostic | Observed count |
|---|---:|
| Feature signatures shared between official train and test | 1,302 |
| Official training records with feature signatures present in test | 10,279 |
| Official test records with feature signatures present in train | 8,541 (10.37% of test) |
| Test records matching features **and** a matching (attack label, binary label) tuple in train | 4,247 |
| Test records matching training features, but lacking the same label tuple in train | 4,294 (5.22% of test) |
| Shared feature-signature groups containing at least one differing label tuple | 548 |
| Shared groups whose training/test label-tuple sets are disjoint | 425 |
| Within-training signature groups with contradictory label tuples | 1,772 |
| Within-testing signature groups with contradictory label tuples | 575 |
| Within-training repeated feature records beyond first | 74,301 |
| Within-testing repeated feature records beyond first | 28,386 |

### Selected classes in the **official test set**

| Class | Test support | Exact feature overlap with train | Percent |
|---|---:|---:|---:|
| DoS | 4,089 | 2,004 | 49.01% |
| Reconnaissance | 3,496 | 1,138 | 32.55% |
| Analysis | 677 | 125 | 18.46% |
| Exploits | 11,132 | 1,802 | 16.19% |
| Generic | 18,871 | 1,722 | 9.13% |
| Worms | 44 | 0 | 0.00% |

See the full CSV for all ten classes and both splits.

## Scientific interpretation — boundaries of the evidence

**Supported:**
- Many traffic records are *indistinguishable under the 42 available features*. Some have inconsistent supervised labels; the same input tuple is not a reliable one-to-one key to the reference label.
- A feature-only deterministic classifier cannot distinguish two exactly equal feature vectors based on information absent from its input; simply increasing model complexity does not resolve that ambiguity.
- Treating a random split of rows within the official training partition as an independent validation set can allow exact-value duplicates to cross the internal train/validation boundary.
- Feature-window construction from adjacent CSV rows cannot, without further evidence, be claimed to encode genuine chronological traffic context.

**Not established:**
- Whether overlapping feature tuples come from the *same underlying PCAP event*, separate equivalent events, or information lost during dataset feature extraction.
- That the official designated split is invalid; we preserve it for standard published benchmark comparability.
- That specific prior published models exploited overlap, that any leaderboard number is inflated by a known amount, or that SCV-IDS's published results are incorrect.
- That a model trained on these fixed-feature CSVs has learned network traffic's physical time dynamics.

## Locked evaluation decisions (before proposed-model training)

1. **Primary benchmark:** unchanged official test set; full ten-class accuracy, macro-F1, weighted-F1, per-class F1, per-class recall, false-positive rate, and confusion matrix, using identical label definitions.
2. **Model development:** only official training partition; internal validation groups every identical parsed 42-feature tuple into the *same fold*. Fit all preprocessing, resampling and calibration using training-side data only.
3. **Overlap-sensitivity:** report secondary test metrics separately for (a) all official test rows, (b) exact feature-overlapping rows, (c) non-overlapping rows, and optionally (d) overlap rows with/without training-label agreement. These are *diagnostics*, not substitutes for primary benchmark results.
4. **Temporal caveat:** row-adjacency is merely an ordering assumption. If serialized windows are later used to reproduce SCV-IDS, we explicitly label these as **CSV row-order windows** and introduce row-order perturbation and window-size controls. Genuine temporal claims require verified packet/flow timestamp metadata.
5. **Test discipline:** published reference audit counts can inform the evaluation plan, but do not tune architectures, thresholds, model weights, or hyperparameters on the official test set. One final locked-model evaluation and a fully versioned report are required.

## Next experimental gate

Establish a **single-row, group-disjoint CatBoost benchmark on the official training partition**, with no test-set predictions yet. This is an important non-image control: improved rare-attack scores must ultimately justify the extra image-serialization and attention overhead relative to strong tabular modeling. See notebook \`03_catboost_groupval.ipynb\`.
