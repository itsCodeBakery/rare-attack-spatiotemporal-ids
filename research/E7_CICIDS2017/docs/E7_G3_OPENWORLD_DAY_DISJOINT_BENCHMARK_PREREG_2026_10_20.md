# E7 G3 — Preregistered day-disjoint open-world source IDS benchmark

**Experiment:** `E7-G3-OPENWORLD-DAYHOLDOUT-001`  
**Preregistered:** 20 October 2026, **before any G3 model training**  
**Source:** G0/G1/G2 verified original CIC-IDS2017 `TrafficLabelling` reports; all original hashes frozen in [machine-readable protocol](../configs/G3_OPENWORLD_DAYHOLDOUT_PROTOCOL_001.json).  
**Research status:** G2 evidence established, **G3 model outputs and classification scores do not yet exist**.

## 1. Source-derived conclusions, not inferences

- Eight original CSV files together have **3,119,345 apparent source rows**. G2 confirmed a **contiguous 288,602-row fully empty tail** in `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`, beginning at source row **170,367**. All 288,602 tail records have *zero nonempty fields*. This explains their simultaneous missing labels, timestamps and endpoints. It is appropriate to **exclude only structurally empty rows in a separate derived analysis table**, resulting in **2,830,743 populated source rows**; do not modify the original 458,968-row CSV or G0/G1/G2 audits. This does **not** prove the root cause of the tail.
- Source file SHA256 remains the authority. Thursday WebAttacks requires strict `cp1252` decoding; the others passed strict UTF-8.
- Monday, Tuesday and Wednesday original CSV order exhibits **208,847**, **159,424**, and **163,908** timestamp reversals. G2 separates simple preceding-hour-12 to following-hour-1 transitions (4,245 / 1,610 / 2,497) from other reversals (204,602 / 157,814 / 161,411). **No source row index is treated as a reliable temporal or causal predecessor.**
- Original ground-truth labels, timezone, full 24-hour convention and flow-completion timestamps remain **unverified**. G3 is a **provisional, exploratory benchmark on published original labels**, not a verified ground-truth or independent corrected-dataset performance claim.

## 2. Effective capture-day support after strictly blank-row exclusion

| Capture day | Populated flows | Benign | Attack families observed |
|---|---:|---:|---|
| Monday | 529,918 | 529,918 | None |
| Tuesday | 445,909 | 432,074 | FTP-Patator (7,938), SSH-Patator (5,897) |
| Wednesday | 692,703 | 440,031 | DoS Hulk (231,073), GoldenEye (10,293), Slowhttptest (5,499), slowloris (5,796), Heartbleed (11) |
| Thursday | 458,968 | 456,752 | Infiltration (36), Web Attack Brute Force (1,507), XSS (652), SQL Injection (21) |
| Friday | 703,245 | 414,322 | DDoS (128,027), PortScan (158,930), Bot (1,966) |
| **Total** | **2,830,743** | **2,273,097** | **557,646 attack flows across day-confined families** |

Source-category support is descriptive. The attack families are **day-confined by the recorded class names**, but capture conditions are also day-specific. Performance on held-out days cannot disentangle attack novelty from network/environment shifts.

## 3. Four frozen evaluation folds

Each fold uses **disjoint, whole capture days**, not row-wise random splits:

| Fold | Training days | Validation day | Test day (unseen attack families) |
|---|---|---|---|
| G3-1 | Monday + Thursday + Friday | Wednesday | Tuesday: FTP/SSH-Patator |
| G3-2 | Monday + Tuesday + Friday | Thursday | Wednesday: DoS/Heartbleed |
| G3-3 | Monday + Tuesday + Wednesday | Friday | Thursday: WebAttacks/Infiltration |
| G3-4 | Monday + Wednesday + Thursday | Tuesday | Friday: DDoS/PortScan/Bot |

For a fixed fold, no original source day may appear in more than one partition. All test-day attack class names are absent from training. The validation day contains distinct attack class names, and is used solely for threshold calibration/selection, **never** for fitting feature transformations or the classifier. The **test day remains untouched until training and validation choices are frozen**. This is cross-source-day binary *unseen-attack* evaluation; it is not zero-shot multiclass recognition, proof of active attack detection chronology, or a direct comparison against the base SCV-IDS paper's CIC Wednesday random 80:20 result.

## 4. Fully specified initial controlled baseline

**Task:** Binary BENIGN vs ATTACK. Attack subtypes are retained for **per-family recall only**; no supervised subtype class target is evaluated in the first G3 benchmark.

**Feature policy:** Use static original flow statistics only. Strictly exclude `Flow ID`, source/destination IPs, source/destination ports, raw `Timestamp`, `Label`, source filename, day of week, record index, all future-context features or target-adjacent rows. Retain `Protocol` and numerical flow statistics, pending per-feature ablation. Fix column alignment before fitting; never infer a missing source column silently. Convert nonfinite feature values to missing; training-only fitting for any imputation/normalization. LightGBM tree learners can natively handle NaNs; calibrators may not inspect test labels. Warn if any derived labeled-but-featureless source record exists.

**Model/control matrix (start with reproducible baseline):**
1. **B0:** Non-learning train-derived prior/majority sanity control, reporting FPR/recall/PR-AUC baseline for the actual class distribution.
2. **B1:** Static-flow LightGBM binary classifier, fixed source-independent hyperparameters, 5 seeds (`42–46`) if Kaggle quota permits.
3. **B2:** B1 with validation-derived binary threshold, same fitted model. Predeclared false-negative cost (5) vs false-positive cost (1) on the validation day; threshold selected on validation only. Report threshold and actual test costs/FPR.
4. **B3 (future):** Selective/rejection control; threshold derived from validation and report risk-coverage curves without claiming a formal risk bound under distribution shift.
5. **Only after reproducible G3 baselines:** Propose a novel method such as **risk-aware open-world attack detection with abstention and stage-wise continual updates**, with mathematically explicit novelty/ablation and separate independent testing. The term `RACE-Net` is a working project label, not yet experimentally supported.

**Fixed evaluation:** per held-out day ROC-AUC, average precision, balanced accuracy, positive/negative precision/recall, binary and macro-F1, confusion counts, per-unseen-family recall and exact support, FPR at the selected validation threshold, and mean/std across seeds. Show `Heartbleed n=11`, `Infiltration n=36`, and SQL Injection `n=21` explicitly as low-support results; no unsupported significance claims. Include all decisions made *without observing test outcomes*.

**Preregistered safety/leakage audits:** ensure no feature uses raw endpoints, timestamp or day identity; no train/test day overlap; no fitted transformation sees validation or test; reject any missing label on populated rows; remove **only completely blank records** in the derived table; verify per-file original hash and per-file class counts; investigate cross-partition identical-feature fingerprints as *candidate* overlap, not automatically exact traffic identity. G3 results are **exploratory** until independently corrected CIC-IDS2017 source and labeling checks are available.

## 5. Reproducibility and publication safeguards

- One new, versioned Kaggle notebook at a time. Commit the code and a runnable synthetic validation suite to GitHub **before** the real source run. Sync at least the aggregate configuration, source hashes, split membership, per-fold metrics and any valid ablation summary immediately after each completed fold. Never upload raw flows, endpoints, data-derived row hashes, Kaggle credentials or original source CSVs to the public repo.
- No modification, movement, replacement or reevaluation of the already frozen UNSW-NB15 official test, M1 six-control results or G0/G1/G2 original evidence.
- **Novelty warning:** The baseline G3 experiment is essential research infrastructure, but is **not itself a novel journal contribution**. It cannot credibly establish superiority over an accuracy number measured on a different dataset split/task. Subsequent method comparisons must use the **same** frozen G3 protocol and corrected-source sensitivity experiments.
- **Future validation:** separately acquire/verify the independently corrected Engelen–Rimmer–Joosen CIC-IDS2017 source; document license and source hashes, attack labeling changes and feature alignment; do not assume it is row-wise interchangeable with these originals.

**Next implementation artifact:** Notebook 30 — first frozen B0/B1/B2 day-disjoint model experiments, subject to synthetic test and code review before Kaggle deployment. **Not run yet.**
