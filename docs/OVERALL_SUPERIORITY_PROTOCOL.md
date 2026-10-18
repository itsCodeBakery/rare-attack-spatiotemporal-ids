# SCV-IDS overall-superiority acceptance protocol (UNSW-NB15)

**Established from base-paper PDF**, IEEE Internet of Things Journal 2026, Tables I, IV and V, page 34623: official UNSW-NB15 training set **175,341 records** and test set **82,332 records**. SCV-IDS reports multiclass **accuracy 85.77%, precision 85.16%, recall 85.77%, F1 84.59%**. The authors separately report ten-class **macro-F1 46.87%**. Their binary task reports accuracy **96.41%** and F1 **96.40%**. Other cross-dataset results must be verified individually before making a comprehensive performance claim.

## Strict goal

Demonstrate *jointly* on the matched, untouched designated **official test split** (with precisely documented count and five-record target handling):
1. Multiclass accuracy **>85.77%**.
2. Aggregate precision **>85.16%**.
3. Aggregate recall **>85.77%**.
4. Aggregate F1 **>84.59%**.
5. Ten-class macro-F1 **>46.87%**, with materially higher performance than the frozen **63.6691% development macro-F1** desired as a research-development goal (not an apples-to-apples official threshold).
6. No catastrophic decrease in Backdoor, Analysis, DoS, Worms recall; report support counts, precision, recall, F1, confusion, significance and CIs. Aim for good balanced accuracy, calibration, inference efficiency and model size.
7. If claiming superiority over **all experiments** rather than multiclass only, evaluate binary and external dataset generalization separately against the exact paper metrics.

**Comparable evaluation warning:** Paper's Table V does not explicitly state which class-average scheme was used for its aggregate precision and F1. Our script reports **support-weighted** precision/recall/F1 and *ten-class macro-F1*. Do not equate support-weighted F1 to paper F1 without verifying the exact averaging protocol. No assertion of outperforming the paper can be made solely from internal validation.

## Development without test contamination

- Original source and split pinned by SHA-256; official test unopened.
- Archive frozen original LightGBM evidence: 63.6691% macro-F1, 80.7142% accuracy and 81.7327% **weighted F1**, **internal** 35,005 target validation.
- CARES-PAIR-001 six-variant proof is independently audited and negative overall.
- **New CTRL-CAT-001**: two CatBoost multiclass training candidates (one unweighted, one moderately rare-weighted), plus **eight predefined probability mixtures** involving the two original frozen LightGBM candidates; training fitted to exactly 140,328 original development target rows, observed metric evaluation on the same 35,005 validation labels. Once candidate checkpoints are archived, do not refit the original 50-epoch CNN/ViT controls.
- Mark **both** the paper's numerically crossed thresholds (different split, purely exploratory) and current best validation-baseline gains. A Pareto screen reports accuracy and macro-F1 simultaneously, not merely their weighted sum.
- A score crossing paper thresholds on internal validation **does NOT establish superiority**. Model selection and all further iterations must be completed before official test.
- Test after freezing protocol, specifying official test row handling, per-class metrics, all dataset transformations trained using training subset only, repeat seeds with uncertainty, and match or qualify the paper's sample/averaging settings.

**Next executable step:** [Notebook 12](../notebooks/12_catboost_overall_superiority_benchmark.ipynb), code [CTRL-CAT-001](../scripts/run_ctrl004_catboost_pareto.py), [immutable config](../configs/ctrl004_catboost_pareto_seed42.json).

**Ethics of score reporting:** New model accuracy cannot be promised. CatBoost + ensembles are conventional comparison baselines, not by themselves a publishable novel architecture. After a Pareto-suitable backbone is established, a distinct, literature-differentiated and training-derived risk-aware method must be implemented and ablated.
