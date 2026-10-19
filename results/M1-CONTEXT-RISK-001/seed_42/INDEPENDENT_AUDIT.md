# M1-CTX-RISK-001 seed 42 — Independent public-artifact numerical audit

**Audited:** 2026-10-19 | **Scope:** All six public 10x10 confusion matrices, per-class precision/recall/F1 tables, frozen protocol, calibration grid, source code, public provenance/summary, and remote private archive metadata. **Not inspected:** private `npz` probability arrays, private fitted checkpoints, training environment internals, or actual original raw CSV bytes. No additional benchmark training/evaluation was conducted for this audit.

Original result files: [seed-42 results](./). Original immutable private archive (GitHub unpublished draft release): [private release metadata](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-c333b2cb4a9e1c45a796). The remote archive asset SHA-256 `00c440f79a3f88e71dc831281fd10776a06af4e57569f62d0b25a8c27910ab88` agrees with the researcher's originally reported local archive hash. Public source + exact experiment reports committed by the researcher in [36b404a793b20ea208e263361922d070ba3ebd4d](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/36b404a793b20ea208e263361922d070ba3ebd4d).

## A. Reconstruction from six authentic confusion matrices

For each stored confusion matrix, the audit independently computed `accuracy=sum(diagonal)/N`, class precision `TP/predicted support`, class recall `TP/true support`, and class F1 `2*TP/(true support+predicted support)`, with **fixed ten classes**, then macro-F1 as the unweighted arithmetic mean of all ten F1 scores. All six matrices have exactly **36,139 target records** with the **same** actual class support vector `[418,381,2553,6805,3661,8485,11319,2231,251,35]` in protocol class order. Every row's 10-class precision/recall/F1 from `*_per_class.csv` matched its confusion matrix to **1e-12**, with no discrepancies. Both aggregate metrics agree with saved `final_public_summary.json` to floating-point precision.

| Control | Accuracy | Macro-F1 | Benign FPR | Analysis recall | Backdoor recall | DoS recall | Worms recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0_TARGET | 0.810094358 | 0.641491192 | 0.128986660 | 0.200956938 | 0.181102362 | 0.792792793 | 0.485714286 |
| C1_CONTEXT | 0.844766042 | 0.650154400 | 0.067762170 | 0.232057416 | 0.157480315 | 0.560908735 | 0.457142857 |
| C2_BLEND | 0.841528543 | 0.656184499 | 0.083487941 | 0.241626794 | 0.165354331 | 0.643556600 | 0.428571429 |
| C3_GATE | 0.842801406 | 0.660669170 | 0.076861914 | 0.236842105 | 0.175853018 | 0.685468077 | 0.457142857 |
| C4_RISK | 0.810094358 | 0.641491192 | 0.128986660 | 0.200956938 | 0.181102362 | 0.792792793 | 0.485714286 |
| C5_NOCONTEXT | 0.810094358 | 0.641491192 | 0.128986660 | 0.200956938 | 0.181102362 | 0.792792793 | 0.485714286 |

**Exact independent equality check:** C0, C4 and C5 all have numerically identical complete public confusion matrices and per-class reports. Since both registered calibrations selected `alpha=0.0`, the equality is the mathematically expected target-only posterior, not evidence of an additional model's gain.

**Exploratory contrast:** C3 exceeds C1 macro-F1 by **0.010514770** (1.0515 percentage points), exceeds C0 by **0.019177978** (1.9178 pp), and exceeds C2 by **0.004484671** (0.4485 pp). It is **not** the preregistered primary C4 endpoint. C3 has 0.6855 DoS recall versus 0.7928 for C0; the macro-F1 gain is **not** uniform improvement across rare classes. Its benign FPR 0.0769 lies above context-only 0.0678 but below target-only 0.1290.

## B. Exact calibration diagnosis: no feasible alpha

Per original `calibration_public.json`: `M1.status=INFEASIBLE`, `M1.alpha=0.0`, `NO_CONTEXT.status=INFEASIBLE`, `NO_CONTEXT.alpha=0.0`. OOF predictions: **139,182** records; gate fitting: **103,397**; separate calibration: **35,785**; public gate/calibration group intersection: **zero**. Outer held-out partition: **36,143 raw** rows -> **36,139 window targets**, separate from original 139,198 raw training-side rows, both from the original *training CSV*.

The registered cap feasibility conditions simultaneously require:
- calibration accuracy at least **context-only accuracy −0.010**;
- calibration benign false alarm rate at most **target-only rate +0.005**;
- each of Analysis, Backdoor, DoS recall at least its target-only recall **−0.020**.

**Failure mechanism, deduced from saved calibration grid plus exact published `calibrate` implementation**:

- `alpha=0.00`: calibration accuracy **0.818052257**. Since predictions match target-only by construction, benign-FPR and minority-recall limits are met **exactly**. The *only* reason it is infeasible must be the **minimum context-relative accuracy** requirement.
- `alpha=0.25`: calibration accuracy **0.833365935** and DoS recall **0.709850993** versus target DoS **0.715231788**. Analysis, Backdoor, DoS and benign-FPR all satisfy their target-relative limits. This candidate likewise fails **accuracy** relative to the context-only expert.
- `alpha=0.50`: calibration DoS recall **0.687086093**, **2.8146 pp below** target-only **0.715231788**; breaches the 2 pp DoS-tolerance condition regardless of accuracy.
- `alpha=0.75`: calibration DoS recall **0.656870861**, **5.8361 pp below** target; breaches DoS tolerance.
- `alpha=1.00`: calibration DoS recall **0.639072848**, **7.6159 pp below** target; breaches DoS tolerance.

Therefore, **none of the five grid points satisfy all guardrails**, and the code correctly records the negative result and selects its explicit target-only fallback. The fallback **does not meet the context-relative accuracy guardrail**, so this run must **not** be described as having demonstrated or guaranteed risk-constrained performance. It is deliberately labeled a failure/fallback, not a successful feasibility solution.

**Unavailable for exact reconstruction from public-only data:** the calibration subset's unblended context-only accuracy and confusion matrix are not included in `calibration_public.json`; the accuracy failures for the first two grid values can nevertheless be logically identified from the published feasibility rule. Exact model probabilities/Brier scores require private evidence review.

## C. Predeclared hypothesis verdict and provenance limits

The **primary C4 hypothesis failed** at seed 42: relative macro-F1 delta vs C0 = **0.0**, vs C1 = **−0.008663208** (−0.8663 pp). It is inappropriate to retrospectively re-label C3 as the primary winning proposed method, or relax the original risk constraints on this holdout. A future alternative is a **new preregistered method** with train-derived model selection and a fresh held-out/external test, not method development on the current used fold.

The public provenance reports `source_group_intersection=0`, `calibration_uses_outer_heldout=false`, `official_test_accessed=false`; source runner reads only verified original `UNSW_NB15_training-set.csv` and forbids test-file paths. These are source/manifests checks, **not** a byte-level audit of Kaggle's execution environment. M1 contextual windows use split-local **CSV position**, not verified timestamps/session identifiers; no temporal causality or external generalization established. All M1 results are distinct from the earlier frozen official UNSW test.

**Research decision:** retain original source/model/metrics and negative C4 outcomes; do not rerun seed 42 or touch official test. Before further training, record whether remaining fixed seeds are used solely for robustness of the **unchanged** frozen M1 protocol or whether a **separately registered M2** changes its constraints and must be tested on genuinely fresh held-out data. Validate context temporal semantics externally before temporal claims.
