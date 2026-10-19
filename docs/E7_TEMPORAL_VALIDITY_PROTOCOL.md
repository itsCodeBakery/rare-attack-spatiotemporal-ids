# E7 — Timestamp-grounded, causal-order intrusion detection: preregistered protocol

**Protocol ID:** `E7-TEMPORAL-VALIDITY-001`  
**Prepared:** 2026-10-19, **status:** DESIGN / DATA-PROVENANCE GATE ONLY. **No dataset has been mounted or audited, no E7 model has been trained, and no scores exist.**  
**Motivation:** [E6 audited evidence](E6_ORDER_ROBUSTNESS_AUDIT.md) shows source-order sensitivity without verification that adjacent original UNSW rows constitute consecutive network events.

## Primary question and falsifiable hypothesis
Can five **actually prior and related completed network flows** improve intrusion detection over the **identical current-flow features**, on later chronological captures, when attack schedule, endpoint identity, duplicate rows, and shuffled-history controls are evaluated? The null hypothesis is that validated causal context provides no reproducible gain in binary attack detection and operational false alarm tradeoffs after these controls. A negative result will be retained and reported.

## G0: dataset-provenance gate (must PASS before fitting anything)

**Candidate source:** University of New Brunswick / Canadian Institute for Cybersecurity, *CIC-IDS2017* official [publisher](https://www.unb.ca/cic/datasets/ids-2017.html). Use publisher's **`GeneratedLabelledFlows.zip`** (labelled flows with timestamp and endpoint identifiers), **NOT** the more commonly mirrored `MachineLearningCSV.zip` if it lacks those fields. The source advertises source/destination IPs, ports, protocol and timestamp as flow-label metadata, and multiple campaign days. Do not treat the two archives as interchangeable. Use only source files for which SHA-256, column presence, actual row counts and label semantics have been verified. Kaggle input path is **pending**; no unverified Kaggle mirror is assumed.

**Quality warning:** Historical CIC-IDS2017 has documented CICFlowMeter and ground-truth errors; compare with Engelen, Rimmer & Joosen's [corrected tool/dataset and paper](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html). The corrected version is a separately versioned sensitivity dataset; do not merge labels from corrected and original without an audited row mapping. Maintain publisher data licences and no public redistribution of raw flows/IP addresses.

Required checks, performed **before train/validation/test or architecture choices**:
1. Audit external download provenance, exact filename/tree, SHA-256, sizes, format, CSV header raw and canonical aliases, labels, schema, IP/port/protocol fields, `Timestamp`, `Flow Duration` and event/source IDs. For every source shard report row counts, duplicate header names, missing/NaN/Inf, and source label support.
2. Parse timestamps with a documented time-zone assumption, date format, range, duplicates, and day-shard consistency. Preserve **original strings** privately and record parser statistics publicly. Validate duration units and finite nonnegative values; define `t_end = t_start + duration` only when the provider's units and semantics support this.
3. Audit equal-timestamp collisions, negative/out-of-day times, 5-tuple duplicates, contradictory labels for identical feature tuples, campaign/time concentration, apparent future leakage, host-pair reuse, and each attack family's day/time support. Time/day are **QC metadata**, never features. Label fields never create contexts.
4. Check capacity to form related histories (0–4 previous completed flows) and group/key coverage without using any labels. Verify nonzero coverage under predeclared recency bounds `{60, 300, 900}` seconds; choose a bound only from allowed **training/validation** diagnostics, never from a final test.
5. **Hard failure:** if timestamps or endpoints do not exist in the actual source, are unparseable, or cannot establish order and completion time, the protocol stops. Do not substitute row number, sorted label, original CSV sequence, or artificial time. If labels/classes are absent from a chronological split, do not fabricate classwise metrics.

**G0 output only:** source manifest, SHA-256, schema quality, time and duplication summary, history-availability plots/tables, label/day coverage and documented PASS/FAIL. No weights, predictions, classifier metrics or access to original UNSW official test. Raw IPs, rows, payloads, or large binaries stay private.

## G1: precommitted task and defensible splits

- **Primary task:** binary detection (benign versus attack); later days can contain unseen attack families. A multiclass analysis is **secondary** and restricted to labels with explicit coverage in all relevant splits. Do not merge incompatible class labels to artificially enable a score.
- **Prospective calendar-day holdout:** train on **Monday–Wednesday**, development/threshold selection on **Thursday**, final locked evaluation on **Friday**. Keep each day/shard intact; do not resample test or tune on Friday. If the source does not support this chronology, declare a protocol deviation and pre-register a replacement **before looking at test predictions**.
- The real capture attack schedule makes day, host and attack label correlated; day-based results represent **within-benchmark forward generalization, not open-world production validation**. Report label/day coverage, host overlap, source IP and destination IP strata, and minority/unseen-family outcomes. Report any unrepresented family as unknown/unseen, not as a known-class multiclass accuracy entry.
- Training-window construction is confined to each allowed partition; no shared source flows across partitions. Deduplicate exact 5-tuples and feature+label keys only under an explicitly described policy determined using the training portion, and audit duplicates across partitions without using held-out labels to adapt model selection. A prespecified embargo at boundaries may exclude flows whose duration overlaps the split; account for all removed counts.
- A separate **train-derived** group-disjoint validation sensitivity check may be used for known-family classwise comparisons, but cannot replace the locked chronological holdout.

## G2: causal representation (only after G0 PASS)

**Decision time:** flow completion. For each target flow `i`, sort candidate completed flows by validated `t_end`; a history item `j` is eligible iff `t_end(j) < t_end(i)`, in the same allowed split, satisfying the predeclared endpoint/protocol grouping and recency bound. Equal completion times must NOT be arbitrarily treated as prior events. History uses **features only**, not labels or learned test statistics.

**Relational grouping:** a canonical, order-independent source/destination **host pair + protocol** from identifiers; a `Flow ID` is an event identifier, not a session-history grouping key. Exclude source/destination IP, flow ID, timestamp, weekday, shard name, label and dataset row ID from predictive features. Port fields are evaluated under a declared included/excluded sensitivity analysis; do not use port as an implicit attack/time label without disclosure.

Use at most four eligible prior completed flows plus target (length 5); fixed zero/missing masks for shorter histories and explicit elapsed-time gaps. All models must score the **same target flow IDs**, including those with no valid history.

### Equal-budget controls (one family of learners; no architecture shopping on final holdout)

| ID | Representation | Why it is essential |
|---|---|---|
| C0 | Target flow only | Non-temporal baseline |
| C1 | Most recent four **global** completed flows | Tests generic timestamp context versus relational context |
| C2 | Most recent four related flows in endpoint-pair/protocol key, true completion-time order | Main proposed causal-context candidate |
| C3 | **Same four C2 events**, order randomly permuted (fixed seed; elapsed-time handling declared) | Isolates historical **order** from membership when compared to C2 |
| C4 | Target with training-distribution-matched unrelated histories sampled label-blind, with no future events | Negative control against generic history and extra dimensions |

Main contrast **C2 − C0**; robustness contrasts **C2 − C3** (order) and **C2 − C4** (related-history semantics). C1 is descriptive. Fix shared preprocessing and model budget. Fit imputers/encoders/scalers/class-balancing **on training flows only**. LightGBM is the computationally efficient first learner; future neural architectures require separately registered ablations and cannot be credited retroactively to E7.

## G3: frozen training and evaluation
- Pre-register selected learner, model features, random seeds (target `42,43,44,45,46`), maximum iterations, fitting subset and versioned hyperparameters **after G0**, but **before** Friday test access. The exact model training spec is not yet fixed and no trained-result claim is permitted.
- Strict train-only fitting; select hyperparameters and decision threshold on allowed development validation only. If Thursday lacks sufficient attacks for a meaningful threshold, declare threshold selection **infeasible**, use a prespecified training-internal validation strategy, and keep Friday sealed.
- At the same fixed score targets, report binary macro-F1, attack-class precision/recall/F1, benign false-positive rate, AUPRC with prevalence, AUROC when both classes are present, classwise known/unseen family recall, calibration/ECE and throughput/latency. Supply *both* natural-prevalence and class-normalized views, with subgroup denominators.
- Save original per-flow decision probabilities and labels **privately**, alongside immutable SHA-256 and code commit. Public reports: confusion matrices, metrics, classwise counts, source manifests, fixed-seed paired comparisons and publication-ready figures regenerated from recorded numeric sources.
- Repeated-seed summaries, paired confidence intervals using independent day/host/event-group clusters where support permits; overlapping flow-window predictions are not independent trials. Do not invent statistical significance when cluster count is insufficient.

## Interpretation/gates
**Go** to a temporal-dependence hypothesis only if C2 reproducibly beats C0 on predeclared primary metrics without unacceptable false-alarm cost, C2 exceeds its same-event order-permuted control where order is claimed, and the gain survives attack schedule/endpoint subgroup and duplication audits. Otherwise report null/negative evidence and do not retrofit a favorable representation to the locked test.

**Comparability constraint:** this **day-wise external benchmark** differs from SCV-IDS's selected Wednesday subset and random 80:20 CIC-IDS2017 split, and from our older UNSW official test. Its score must **not** be framed as a direct head-to-head victory over published 99.81% CIC values. A matched split/protocol reproduction would be a separate preregistered project, not a replacement for E7.

## Execution boundary and next user-facing action
**E7 currently stops at G0 preparation.** Obtain and mount publisher-origin `GeneratedLabelledFlows.zip` on Kaggle (or explicitly verified schema-identical archive with full metadata); record only its Kaggle path and uploaded archive's SHA-256 privately/publicly as appropriate. Then implement and run **G0 inspection only**, returning a PASS/FAIL and reproducible data-coverage report **before** any experiments are trained.
