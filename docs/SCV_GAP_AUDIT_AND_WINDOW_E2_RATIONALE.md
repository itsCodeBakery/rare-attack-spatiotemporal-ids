# SCV-IDS scientific gap audit and E2 experiment rationale (19 October 2026)

## Concrete source and implementation comparison

**Base publication:** Chen et al., *SCV-IDS: CNN-ViT Crossattention on Serialized Traffic Images for Spatiotemporal Intrusion Detection*, IEEE Internet of Things Journal 13(15), 2026; DOI 10.1109/JIOT.2026.3695912. The source PDF was reviewed; this repository does not redistribute the copyrighted PDF.

**Paper-reported, official-test regime:** multiclass accuracy 85.77%, aggregate precision 85.16%, aggregate recall 85.77%, aggregate F1 84.59%, and additionally explicitly reported ten-class macro-F1 **46.87%**. The paper states class **Analysis F1 <1%** and notes imbalance, overlapping Exploits patterns and possible index-encoding bias. Do not conflate the aggregate F1 with macro-F1 or assert aggregate averaging without reproducing Table V.

**Paper's window-size ablation:** *its own* window-size 1 accuracy **75.82%** versus window-size 5 accuracy **85.77%**. The paper constructs a 5 × 42 = 210-value window, attaches the newest row's class, maps values to 24-bit RGB and pads to 16×16. The paper first separates official training/test partitions and then windows within each partition. This is evidence for a paper-specific window effect, **not evidence that our LightGBM will improve by 9.95 percentage points**, and not evidence that CSV order is truly chronological.

**Our frozen internal benchmark:** from the original UNSW-NB15 *training* file (175,341 rows), group-disjoint internal split of 140,332/35,009 source records, then fixed five-row target offsets, leaving 140,328/35,005 training/validation labels. These are **not the base paper's official test conditions**. Two strong previously trained LightGBM models used only 42 features of the *newest window record*. Last-record unweighted accuracy 83.2595%, macro-F1 60.2494%; rare-weighted accuracy 80.7142%, macro-F1 63.6691%. The separate five-seed CatBoost/LGBM mixture maximizes internal macro-F1 around 63.97%, at around 81.21% accuracy.

## What earlier experiments taught us

- Corrected CNN–ViT, pairwise CARES, multiclass CARES, class-pair routing and GPU residual contrastive MLP did **not** establish a robust simultaneous gain in both accuracy and macro-F1.
- Class reweighting shifts many samples between DoS and Exploits. This is a measurable classification trade-off, not solved by additional thresholds.
- Critical control missing from the strong tabular benchmark: **use of all five input records of the frozen SCV window**. E2 directly changes the available information while holding the 35,005 target rows fixed.

## E2 (preregistered) design and scientific safeguards

1. Reconstitute training/validation windows separately using original frozen partition order. Confirm SHA256 of training CSV, split and validation target-probability archive.
2. Materialize 5 × 42 = 210 true raw feature values (oldest to newest) for each labeled window, preserving categorical types; no labels or record IDs are input.
3. Train unweighted and rare-class-sqrt LightGBM variants, comparing with **unchanged, SHA256-frozen**, last-record LightGBM. Real progress shown each 50 boosting rounds and via tqdm; every 100 rounds save a local partial booster.
4. Automatically create an **unpublished private GitHub release** for each completed model (fitted booster, per-row probabilities), with hash verification. Public GitHub includes only numeric metrics, classwise scores, genuine confusion matrices and provenance.
5. Report **accuracy, macro-F1, weighted F1, per-class DoS/Analysis/Backdoor recall** and exact FP/FN counts. No comparison is marketed as official-test superiority.
6. If both accuracy and macro-F1 improve over the best *frozen baselines*, proceed to shuffled-context/window-order negative controls, multiple seeds, and eventually frozen-method official-test evaluation. If either deteriorates, document the negative result and **stop claiming the serialized context helped**.

## Key experimental uncertainties

- UNSW-NB15 CSV row order is **not verified to be chronological**. Hence E2 window suffix `lagN` is a **position** only. A chronological or per-flow physical IDS cannot be claimed without a reconstructed and validated time basis.
- A group-disjoint split created inside official training data is useful for model selection but is NOT the original paper's evaluation distribution; high-quality final work needs a reproducible comparable official-test evaluation and matched scoring definitions.
- Training/validation are overlapped by construction *within each partition* as windows share historical rows; label leakage must be addressed by evaluating the held-out newest row, never crossing partitions, and reporting that adjacent window samples are statistically dependent.
- High-impact journal suitability requires more than a score increase: validated temporal semantics, external-domain robustness, calibration/uncertainty, classwise security risk and rigorous reproducibility.

**Decision:** E2 is an evidence-based screening experiment, not a proposed journal-worthy novel architecture or guaranteed improvement. Its outcome determines whether full-window representations are worth developing.
