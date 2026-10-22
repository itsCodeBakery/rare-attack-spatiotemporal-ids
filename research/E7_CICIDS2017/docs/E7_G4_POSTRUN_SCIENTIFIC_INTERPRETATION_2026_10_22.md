# E7 G4 — Post-run scientific interpretation and limits

**Date:** 22 October 2026  
**Status:** G4 results independently retrieved and verified: **20/20** immutable seed reports and **1/1** consolidated four-fold/five-seed summary.  
**Provenance:** [G4 archived results](../results/G4_RISK_ABSTENTION/G4_FOUR_FOLD_FIVE_SEED_SUMMARY.json); [locked G4 protocol](../configs/G4_RISK_CALIBRATION_PROTOCOL_001.json); [Notebook 31](../notebooks/31_e7_g4_validation_risk_and_abstention.ipynb).  
**Scientific classification:** **RETROSPECTIVE EXPLORATORY REUSED G3 TEST**. G3 test-day outcomes were already inspected before G4 was designed. This prevents treating G4 test-day comparisons as untouched confirmatory evidence, even though G4 thresholds were selected on validation data only. Do not call the G4 values verified cross-day risk bounds.

## 1. Locked methods and reported experimental results

M0 = original frozen G3 B1 LightGBM detector with threshold 0.5. M1 = validation-only 1% alarm FPR-constrained threshold using an empirical-CDF (DKW) margin. M2 = M1 high threshold with independently chosen lower auto-benign threshold, deferring intermediate scores. Same 20 previously trained private G3 checkpoints. **G4 performs no new model fitting.**

| Held-out source day | M0 original binary Macro-F1 (5-seed mean) | M1 Macro-F1 (5-seed mean) | M1 unseen-attack alert recall (5-seed mean) | M1 test FPR (5-seed mean) | M2 accepted coverage (5-seed mean) | M2 selective error, nonempty decisions only |
|---|---:|---:|---:|---:|---:|---:|
| Tuesday | 0.492171 | 0.722964 | 0.355186 | 0.004833 | 0.015704 | 0.326974 |
| Wednesday | 0.400951 | 0.388468 | **0.000000** | 0.000000 | **0.000000** | **undefined** (zero decisions) |
| Thursday | 0.659836 | 0.831968 | 0.897473 | 0.003908 | 0.008223 | 0.468886 |
| Friday | 0.381150 | 0.616383 | 0.281365 | 0.006245 | 0.119276 | 0.029776 |

These are per-fold seed means over the **same test-day records**, not 20 independent out-of-domain test sets. Per-fold baseline-to-M1 differences are descriptive and **may not be reported as independent gains or evidence of a superior algorithm**.

## 2. Forensic checks across every individual seed report

- **20/20** original G4 seed reports were fetched and inspectable; each has `RETROSPECTIVE_EXPLORATORY_REUSED_G3_TEST` status and references a SHA-256-checkpointed private G3 model.
- **20/20 M2 runs made zero automatic benign decisions.** Every selected `auto_benign_low` was exactly `0`; no test score satisfied auto-benign `p<=0`. Thus implemented M2 degenerates to **alarm or abstain**, not a three-way detector.
- **20/20** M2 runs were **below the prespecified 50% minimum test coverage**: Tuesday mean 1.5704%, Wednesday 0%, Thursday 0.8223%, Friday 11.9276%. Wednesday coverage zero means selective error is **undefined**, not 0%, and should not be plotted as error-free.
- The M1 no-alarm sentinel threshold `1.000001` was selected in **all five Wednesday seeds**. Their validation FPR feasibility flag was `true`, because always outputting no alert trivially respects the false-positive bound; this is a **vacuous feasible solution with zero attack-detection utility**.
- **One of the 20 original test runs violated the nominal 1% alarm false-positive limit**: Friday test, seed 45, FPR `0.019965147880151187` (~1.9965%). Therefore a validation-only FPR constraint cannot be advertised as guaranteed after day shift. Moreover, flow-level dependence violates the iid premise of the calibration-day DKW inequality.
- Tuesday individual attack recall varies greatly across seed models (10.99% to 45.27%), despite stable original M0 macro-F1. Friday recall is consistently ~28.1%. Thursday mean recall 89.75% occurs on a very small attack prevalence within its capture day, so the high Macro-F1 does not justify indiscriminate deployment or rare-family reliability claims.
- Four distinct test days, each associated with different original capture-day attack labels and likely environmental conditions, confound novelty with day-specific distribution shift. Low-support attack labels such as Heartbleed (11), Infiltration (36), and SQL Injection (21) remain unfit for confident family-specific claims.

## 3. What is and is not supported

**Supported by the archived experiment:**
- M1 thresholds based exclusively on the corresponding G3 validation day **retrospectively** improve binary Macro-F1 on Tuesday, Thursday and Friday; Wednesday drops, detects no attacks, and becomes no-alarm.
- Low test false-positive rates on average do not establish non-vacuous detection; they must be accompanied by recall and individual-seed violations.
- M2 currently has no effective auto-benign region; its apparent selective error must be shown together with very low/zero coverage.
- Risk-control feasibility on calibration data must be distinguished from operational feasibility and actual shifted-day performance.

**Not supported:**
- A formal **cross-day** FPR/FNR, selective-risk, or coverage guarantee.
- G4 independence from already-examined G3 test outcomes or superiority over M0 based on independent test evidence.
- Any new spatiotemporal feature contribution, continual learning, open-set multiclass recognition, corrected attack ground truth, or comparison with the original SCV-IDS random test split.
- Successful operational abstention: 20/20 G4 M2 runs violate the declared 50% minimum coverage.

## 4. Research gap grounded in the observed data

A **simultaneous high-level alert, auto-benign and abstain** IDS decision layer under novel-attack/capture-day shifts needs to balance:

1. Detect enough true rare/unseen attacks (**non-vacuous recall**) while limiting false positives.
2. Ensure **usable coverage**, including safe automatic benign decisions, instead of vacuous always-abstain or always-benign/always-attack strategies.
3. Quantify **calibration uncertainty under distribution shift and dependence**; never claim iid theoretical bounds continue to hold on shifted data without verified conditions.
4. Maintain frozen provenance, source quality, day/session leakage controls and original label uncertainty.

G3 + G4 establish a concrete failure case for this line of research; they **do not yet establish novelty or a solved problem**.

## 5. Recommended next phase: G5 independently confirmed validation

Before implementing a new risk-aware method, independently verify the corrected CIC-IDS2017 source and provenance (including attack labels, acquisition, and day semantics), or source an external flow corpus with documented ground truth. Freeze **all** source hashes, family support, train/calibration/test partitions, threshold-selection objective, minimum coverage, primary joint safety-and-utility metric, and expected failure behavior **before opening independent test labels or inspecting their results**.

A potential design goal for a future model (not a claimed result) is a **non-vacuous operating-point selection with explicit rejection-cost/coverage constraint**, coupled to out-of-distribution or distribution-shift detection and conservative fallback when calibration is unrepresentative. Evaluate against frozen B0/B1/B2 and M1/M2 under the **same** new split, plus ablations. If no threshold meets the joint constraints, report an infeasible operating region rather than relaxing them after observing test outcomes.

Never tune a new design on the reused G3 Tuesday/Wednesday/Thursday/Friday test-day results and then claim confirmation on those same days.

## Preservation and reproducibility

This is an **additive interpretation document only**. The G0, G1, G2, G3 and G4 original model, metrics and source evidence, M1 runs, unpublished private GitHub draft model release, and frozen UNSW-NB15 official test all remain unchanged. No additional Kaggle compute or Git model writes are required to complete the G4 interpretation.
