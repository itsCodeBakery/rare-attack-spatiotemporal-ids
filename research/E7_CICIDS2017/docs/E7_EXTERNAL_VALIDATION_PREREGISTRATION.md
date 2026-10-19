# E7-EXT: timestamp-authenticated external validation — preregistration (2026-10-19)

**Experiment family:** `E7-EXT-VALIDITY-001` (new, not a replacement for E6 or M1); **status: PROTOCOL ONLY / NO DATA AUDITED, NO MODELS FITTED, NO SCORES**. Original frozen official UNSW-NB15 model/evaluation are never reopened. Any correction to this plan after external results requires a **new experiment ID** and independent future evaluation.

## Critical discovery from the uploaded SCV-IDS base paper

SCV-IDS **already reports additional CIC-IDS2017 evaluation**: the paper describes the `Wednesday-WorkingHours.csv` subset, drops 11 Heartbleed flows, and makes an 80:20 train/test partition; its reported CIC multiclass accuracy is **99.81%**. Thus calling an ordinary Wednesday random 80:20 evaluation an unprecedented external validation is invalid. In particular, we must not compare our E7 future-day binary detection against SCV-IDS's **different** Wednesday 5/6-class supervised task numerically as though matched.

SCV-IDS's UNSW-NB15 source-order windows are not independently proven chronological. E5 found severe original row-label clustering. M1's global risk cap was infeasible at all five calibrated points (seed 42), and C4 fell back to the C0 model. The next study needs genuine source/flow timing and strong controls, not another cosmetic MoE.

## Datasets: authorized candidate sources, and why

| Role | Candidate and authoritative source | Required provenance gate | Experimental scope |
|---|---|---|---|
| Primary external context-time benchmark | [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), **`GeneratedLabelledFlows.zip`** | Keep actual Timestamp, Src/Dst IP+ports, Protocol+Label; log publisher/license/source checksum; must verify start/flow-finish semantics; **do not silently substitute `MachineLearningCSV.zip`** | Genuine prior related completed flows on chronological episodes; stronger than SCV's Wednesday random 80/20 |
| Mandatory ground-truth sensitivity | [Engelen–Rimmer–Joosen corrected CICIDS2017 / fixed CICFlowMeter](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html) | Separate original/corrected versions, labels including **X - Attempted**, flow-generation rules and raw feature schema. The versions are **not row-wise interchangeable**. | Repeat common experiment type on corrected dataset, report original-vs-corrected sensitivity **separately** |
| Second independent external environment (later) | [UNB CIC IoT-DIAD 2024](https://www.unb.ca/cic/datasets/iot-diad-2024.html) | Validate timestamp and endpoint fields on actual downloaded archive; device/attack schedule and consent/licence check; data volume may exceed default Kaggle memory | IoT-source generalization and source/context validity; **not** an assumption that cross-dataset raw schema matches UNSW |
| Optional harmonization-only transfer suite | [UQ NetFlow V2/V3 data](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Preserve common schema, source-dataset identity, no target fit; V3 includes published temporal aggregates **but does not itself prove flow start/end chronology** | Binary zero-shot source-domain → target-domain transfer when features genuinely align |

**CIC-IDS2017 quality warning:** [Engelen et al.](https://intrusion-detection.distrinet-research.be/WTMC2021/) showed label and CICFlowMeter construction errors; avoid equating original labels with correct ground truth. **Dataset suitability does not imply trustworthy time by default**.

## G0 — no model training until complete

Before any model or window work:
1. Dataset name, exact original file(s), authoritative publishing URL, licence, source/mirror provenance attestation, bytes/sha256, CSV encoding, columns, rows, label list/support, day-of-week/day counts, null/inf, duplicates, conflicting labels, exact IP/port/protocol coverage, timestamp parse, timezone/clock skew and start/end duration semantics. **Keep only hashed/high-level aggregates in public repo.**
2. Verify label taxonomy and scope against source attack schedule, annotate attack `attempted` vs `effective` in corrected version. **Do not relabel attempted attacks as simply benign without a prespecified, documented policy**.
3. Verify there are endpoint pairs (or flow IDs) connecting related flows and **the preceding record is completed before the target begins**. Formal history condition: `end(previous) <= start(target)` with validated duration unit/time semantics, plus same predeclared endpoint-pair or session relationship; never use target/future labels for history selection. If end cannot be established, context quality FAILS and only target-only/binary-data-quality baselines can proceed.
4. Chart per-day class support, attack-time blocks, class label versus row position and time, endpoint separation/overlap, duplicate hashes. Declare whether **chronological splitting leads to absent classes** before defining endpoint/metric.
5. Freeze train/calibration/test date and group boundaries and label mapping, SHA256 of source and split indices, feature list, random seeds, inclusion criteria and computational budget in new config. **Failure in G0 stops model training.**

In particular, a CSV's `Timestamp` field is **not** proof its sequential rows are network-causal observations; sessions and flow completion must be independently checked.

## G1 — primary authenticated-timing *within external dataset* replication

**Primary scientific question:** Does verified previous-flow context improve detection on **future chronological episodes** versus target-only features, for the **same target flows**? This tests methodology portability by retraining on an external dataset (call it **independent-dataset replication**, not zero-shot transfer from UNSW). All original UNSW performance figures stay frozen.

- From genuinely validated flow records, construct `x_i` current flow feature vector and `H_i` up to **four previous related completed flows** (strictly from earlier time in that source/session; no label-based filtering). Mask when fewer than four exist; require missing-context baseline. Use reproducible strictly increasing timestamp+stable within-time tiebreaker, never future flow features.
- Split first by **capture time/campaign**, not random rows/windows. Explicit closed-set experiment may be possible only with sufficient support for *all* chosen attack classes in train/calibration/test; otherwise state binary/seen-classes and mark excluded/unseen labels as **unknown**, separately measured. Whole-day holdout in CIC often yields unseen attack families and therefore measures open-world generalization, not ordinary ten-class closed-set scoring.
- Inner time-ordered train/fitting/calibration + forward test, with no session/window crossing boundaries. If endpoint-campaign segregation and chronology cannot both hold, report both as different sensitivity protocols and acknowledge domain shift. Purge boundary windows and close all session contexts at boundaries; no random row split for main results.
- Freeze class mapping **before** model training. Primary endpoint: binary attack macro-F1 and attack PR-AUC with benign FPR, plus class/novel-attack supports and counts of previously unseen families. Secondary closed-set macro-F1 only if class support is present; never silently equate CIC labels with UNSW's ten classes.
- All model variants share identical target flow indices, same train-only feature transforms, model budgets and `seed={42,43,44,45,46}`, no test label reuse for calibration.

**Mandatory controls on each matched split:**
  1. **T**: target-only LightGBM, 58-round budget as an initial fixed budget, then a separately registered honest *training-only* hyperparameter benchmark if required.
  2. **R**: authentic same-entity **related, prior-completed** context model.
  3. **U**: unrelated but past completed flow history with matched 4-slot availability, preserving prevalence/time strata without using labels at inference.
  4. **P**: same related past flows **permuted within the context** (both train and test with predeclared seeds, plus test-only perturbation separately).
  5. **B**: fixed 50/50 probability blend T/R.
  6. **G**: unrestricted gate trained only on time-respecting train-side out-of-fold predictions. C3's existing architecture is **not** a novelty claim.
  7. **C4**: identical frozen M1-style global risk cap with **zero-violation reporting** and explicit infeasible fallback. **Do not tune alpha grid/limits from E7 test**.
  8. **M2** (optional): separate freshly preregistered candidate with distinct novelty and all design selection strictly inside the training horizon; if M2 is invented after seeing E7 test, require another independent test dataset.
  9. Strong class imbalance baseline matching [HICID](https://doi.org/10.1016/j.comnet.2026.112532), at least a reproducible hierarchical rare-class baseline; report faithful reproduction deviations.

**Ablations:** history length `{0,1,2,4}`; same-endpoint vs same-conversation; shuffle within valid related past; random unrelated past; label-independent quality gate; scarcity/missing context; feature leakage e.g. source/dest IP, time, port, day and flow IDs **not used as classifier features unless a separate predeclared confound ablation**.

**Outcome:** report paired macro-F1 with uncertainty, FPR at fixed target-side attack recall, time/host-matched attack PR-AUC, per-class support/recall/F1, calibration/ECE/Brier, peak RAM, latency and size. Significance: paired group-block bootstrap on sessions or days where there are adequate independent units, otherwise avoid claiming precise independent CIs.

## G2 — strict cross-dataset zero-shot transfer (optional, separate from G1)

**Two independent tasks must never be conflated:**
- **Train-from-scratch on CIC**: independent-dataset *replication*. Refit the learner on that dataset's own training partition with no test exposure. This does **not** show the UNSW pretrained checkpoint transfers.
- **UNSW → CIC zero-shot**: feasible **only after** shared feature/schema conversion or common NetFlow extraction is independently validated. Fit encoding, imputation, normalization and thresholds on **source only**. Freeze model, evaluate target once; don't use target labels or target examples for training/threshold calibration. Report transfer as binary attack/benign only where semantics align, including unknown family handling.

Never feed CIC's ~80 features to the UNSW model's 42-feature checkpoint as though the architectures/schemas matched. No incomparable 99.81% Wednesday base-paper metrics used as direct denominator for G1/G2.

## G3 — labels-corrected sensitivity (mandatory before strong claims)

Reproduce G1 using the publisher original and separately the corrected 2021/2022 release, each with its *own* audited flow construction and permitted features. Compare *direction* of relative paired differences and failures; don't merge corrected/source labels without a documented 1-to-1 flow mapping. Pay attention to `- Attempted` samples and missing Heartbleed or WebAttack class support.

## Stop/go & Kaggle engineering

**No G0 audit result currently exists in this repo.** Read-only tool source will be `scripts/e7_g0_external_source_audit.py`, its tests at `tests/test_e7_g0_external_source_audit.py`, and public manifests under `results/E7-EXT-VALIDITY-001/g0/`. No raw flows/IP values are committed.

Compute: Kaggle **CPU** sufficient for G0 and lightweight LightGBM; GPU only when fixed neural baselines need it. Log SHA256 of original datasets and source commit, progress/manifest every candidate, and checksum-verified **GitHub draft releases per completed stage** from the beginning (E6 established practice). On connection failure halt after local private checkpoint; never proceed for hours unbacked up. GitHub `pushRare` secret required for autosync jobs; no secret printed. Gate smoke tests on synthetic CSV before real input.

**Current next action for the researcher:** download **original `GeneratedLabelledFlows.zip`** from the UNB publisher and attach it as a Kaggle dataset **without editing/re-sorting/relabeling it**; *alternatively* share an independently documented corrected-version archive, clearly named as corrected. Run G0 audit only. Once G0 is checked, freeze the actual G1 chronological split and implement models. We cannot select timestamp format or exact split/labels until real source column/value audit succeeds.

## Publication feasibility

This plan does not guarantee improved performance or methodological novelty. A negative result from genuinely authenticated context can support a dataset/protocol reliability paper; a novel risk-limited algorithm paper requires additional distinct mathematics and positive independently validated empirical evidence. Venue selection intentionally deferred.
