# Exact-value feature overlap audit

**Computed from verified official train/test files; no training.**

- Training: 175,341; testing: 82,332
- Shared exact parsed-feature signatures: 1,302

- exact_shared_feature_signatures: **1,302**
- train_records_sharing_test_feature_values: **10,279**
- test_records_sharing_train_feature_values: **8,541**
- duplicate_train_feature_records_beyond_first: **74,301**
- duplicate_test_feature_records_beyond_first: **28,386**
- within_train_feature_groups_with_label_conflict: **1,772**
- within_test_feature_groups_with_label_conflict: **575**
- shared_feature_groups_with_label_conflict: **548**
- shared_feature_groups_with_disjoint_train_test_labels: **425**
- test_records_with_exact_feature_and_label_match_in_train: **4,247**
- test_records_with_feature_match_but_no_matching_train_label: **4,294**

## Interpretation

These are exact parsed **feature-value** matches, ignoring ID and targets. They are not automatically proof of a data split mistake or illicit leakage.
Compare classes and label conflicts before defining the training protocol.
**No dataset records or individual digests were published.**
No deduplication, model training, imputation or split changes were performed.
CSV row order has not been proven chronological; do not assume temporal continuity.
