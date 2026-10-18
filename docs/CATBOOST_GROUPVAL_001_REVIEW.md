# CATBOOST-GROUPVAL-001 — independent results audit

**Verified:** 2026-10-18. Evidence commit: [4c5956d](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/4c5956d1add7819f000bd20cbedbba4c3217522e). This is an **internal training-partition validation run**, **NOT** the official UNSW-NB15 test benchmark. The original UNSW publisher-vs-Kaggle mirror equivalence check remains open as [DATA-002](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/issues/2).

## Evidence verified

All figures below were independently recalculated from [the saved confusion matrix](../results/CATBOOST-GROUPVAL-001/confusion_matrix.csv), matched against [metrics.json](../results/CATBOOST-GROUPVAL-001/metrics.json) and [per_class_metrics.csv](../results/CATBOOST-GROUPVAL-001/per_class_metrics.csv). The manifest records 140,332 training-partition development rows, 35,009 internal validation rows, 0 shared exact 42-feature groups across development/validation, seed 42, StratifiedGroupKFold fold 2 of 5, CatBoost 1.2.10, and Kaggle Tesla T4 GPU. It states the official test was not used for inference.

| Metric (validation only) | Measured |
|---|---:|
| Ten-class macro-F1 | 55.4107% |
| Accuracy | 82.4045% |
| Weighted-F1 | 79.9037% |
| Balanced accuracy | 53.6054% |
| Normal-to-attack false-positive rate | 9.0934% |
| Correct predictions | 28,849 / 35,009 |
| Logged training time | 11.475 seconds |
| CatBoost selected iteration | 599 of 600 configured |
| Training checkpoint SHA-256 | `139cc1eed239b5b55c4b59ce3783a59d4ae82a1c5be587a218ca45def36174ad` |

### Per-class validation quality

| Attack class | Support | Recall | F1 |
|---|---:|---:|---:|
| Analysis | 428 | 13.79% | 22.78% |
| Backdoor | 350 | 10.29% | 18.23% |
| DoS | 2,509 | 4.42% | 8.24% |
| Exploits | 6,684 | 94.24% | 73.94% |
| Fuzzers | 3,679 | 75.24% | 73.99% |
| Generic | 7,894 | 97.47% | 98.70% |
| Normal | 11,107 | 90.91% | 92.29% |
| Reconnaissance | 2,116 | 77.74% | 82.66% |
| Shellcode | 221 | 62.44% | 65.87% |
| Worms | 21 | 9.52% | 17.39% |

**Particularly notable:** 2,318 of 2,509 actual DoS validation examples were predicted as Exploits. The rarest validation class has only 21 Worms examples, making individual-class estimates imprecise. Our planned methodological emphasis on class confusion, not just rarity alone, is supported by these observations.

## Scientific interpretation and limitations

- **DO NOT** directly compare validation macro-F1 55.41% against the published SCV-IDS 46.87% **official test** macro-F1; different evaluation protocols. No outperformance claim is permitted.
- The model has no image serialization or crossattention. It is a strong **single-record tabular control**, not a reproduction of SCV-IDS.
- One seed and one selected validation fold are insufficient for final claims. Better macro-F1 must be verified with repeated runs and locked test evaluation once protocols are finalized.
- No original publisher-file equivalence check has been completed; these results are from the **primus11 Kaggle mirror**, which matches official class counts and partition sizes but has swapped file names.
- The source commit in the run manifest is `2fa73f0e8eda5a8f056b341580b14b9d222224ed`. This preserves the precise code version used even though research documentation has advanced since the run.
- The run manifest reports a checkpoint SHA-256 and Kaggle path; this GitHub repository does **not** contain the checkpoint itself.
- Training selection finished at iteration 599, the last of 600 allowed iterations; a follow-up hyperparameter study could test longer training, but **do not tune until publisher provenance and the experimental plan have been finalized**.

**Next gate:** verify the two CSV inputs against the publisher original files, as defined in [UNSW_DATA_PROVENANCE.md](UNSW_DATA_PROVENANCE.md). Do not silently treat a popular Kaggle mirror as publisher-verified identical. Maintain the unchanged official test set for final comparison after model selection.
