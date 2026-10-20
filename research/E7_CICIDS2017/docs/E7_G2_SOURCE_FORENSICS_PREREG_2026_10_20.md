# E7 G2 — Original CSV Structural Integrity and Clock Forensics (Preregistered)

**Study:** `E7-G2-FORENSICS-001`  
**Date:** 20 October 2026  
**Status:** GitHub CI validated, **actual Kaggle G2 outputs pending**.  
**Source:** Eight completed and archived original CIC-IDS2017 G0/G1 CSV audits; original TrafficLabelling CSVs are held by the Kaggle dataset `dsshayan/dscicids`.

## Motivating verified evidence (G1 and G0, not G2 results)

1. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` has 458,968 records. The original G1 audit identifies 288,602 missing labels as **one contiguous source-row block**, positions **170,367–458,968**. The original G0 audit finds exactly 288,602 timestamp parse failures and missing endpoints. The source uses strict `cp1252` decoding because full-file UTF-8 decoding failed.
2. The 288,602 anomalous-row count numerically equals the 288,602 original records in `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`. **This coincidence is not evidence of a concatenation error, source-file mixing, or a shared set of rows.** No source records have been relabeled or deleted.
3. G1 counts adjacent parsed timestamp reversals in raw source row order: **Monday 208,847, Tuesday 159,424, Wednesday 163,908**. Mon/Tue/Wed hour distributions include `12` and `01` but no unambiguous `13`–`23` representation. A twelve-hour clock reset is a hypothesis, not a verified 12-hour clock convention.
4. **Known limitation of legacy G0 report:** `label_day_counts` can omit invalid-timestamp rows in chunks containing no valid timestamps. G1 missing-label totals and independent G2 row-density checks—not legacy `label_day_counts`—are the verification sources for missingness. Original G0 reports remain immutable.

## Frozen G2 tests and expected claims

- **G2-A: WebAttacks original CSV structure (full source scan).** Count truly empty records (all original fields blank), rows with missing labels but some other source fields populated, rows with missing timestamps, missing-label row runs, exact boundaries, trailing blank rows and original source-column nonempty density. Make no diagnosis of a source-generation cause without original capture evidence.
- **G2-B: Monday/Tuesday/Wednesday original clock-order decomposition (three full source scans).** Recount descending timestamp pairs and classify into source adjacent `12:xx → 01:xx` versus all other descending pairs; record reversal-magnitude bins and timestamp ties; do not change any original timestamp, re-sort rows or infer traffic causality.
- **G2-C: Authenticity.** SHA256-check all four source CSVs against both archived G0 and G1 evidence before parsing, and check G2 row counts and raw reversal/missing-label totals against preregistered G1 counts.
- **G2-D: Reproducibility.** Publish only strictly-schema-validated aggregate reports via append-only, byte-verified GitHub writes. One report after each CSV. On GitHub PUT followed by temporary GET 404, verify immutable Git blob/commit; retain local receipt and never issue blind repeated PUT.
- **G2-E: Safety.** Never publish raw flow data, IP addresses, row-level fingerprints, private secrets or fitted models. Frozen UNSW-NB15 official test and prior model evidence are never read or modified.

**Interpretation gate:** A fully blank 288,602-row tail supports **source-padding or structural-empty-row suspicion**, not an established origin or evidence of the correct label for any deleted/empty row. A mixture of blank and populated unlabeled rows warrants per-source acquisition/label review. Any non-`12 → 1` inversions mean raw CSV order is not monotonic even under the simplest clock-wrap explanation. G2 by itself **cannot** establish timezone, flow-completion order, correct attack labels or a defensible leakage-free training/test split.

## Deliverables

- [Notebook 29](../notebooks/29_e7_g2_source_integrity_clock_forensics.ipynb) — four-source G2 original-data audit and verified GitHub autosync
- [G2 source scanner](../scripts/e7_g2_source_clock_forensics.py) and [immutable public publisher](../scripts/e7_g2_publish.py)
- [GitHub Actions checks](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39403758615) — **38 synthetic tests passed**, all three Notebook 29 code cells compiled. This verifies code and synthetic cases, **not** live Kaggle G2 results.
- Output location after actual execution: `research/E7_CICIDS2017/results/G2_SOURCE_FORENSICS/`. There are no G2 experiment results until Kaggle finishes uploading them.

## Frozen follow-on requirements

After G2 outputs, source-forensic interpretation, independent original/publisher provenance, corrected CICIDS2017 sensitivity, source clock convention, start/end semantics and endpoint/session linkage must be resolved or the unsafe chronological claims must be explicitly abandoned. Predeclare any alternative leakage-controlled study design **before** training. No reuse of frozen UNSW-NB15 official test.
