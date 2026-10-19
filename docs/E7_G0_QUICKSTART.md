# E7 G0 external-source audit quickstart (no model training)

**Purpose:** verify provenance, headers, explicit timestamp format, label/day support and endpoint metadata for an actual external **publisher** labeled-flow archive. This is a data gate, not a trained IDS result. It cannot alone certify causal network history.

1. Download the publisher-original `GeneratedLabelledFlows.zip` from [University of New Brunswick CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), and retain the original archive unchanged. **Do not substitute `MachineLearningCSV.zip`.** For the mandatory corrected-flow sensitivity experiment use the separately versioned [Engelen corrected CIC-IDS2017](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html); do **not** combine versions.
2. Attach that file to a Kaggle notebook as an input dataset. Kaggle may show the ZIP or may expose the CSV files extracted, both are supported by the command-line tool, one mounted path at a time.
3. Open the version-controlled [Notebook 25](../notebooks/25_e7_external_g0_provenance_audit.ipynb). Its first cell clones this existing GitHub repo to a **different** directory than the M1 checkout and runs six synthetic gate tests. Keep CPU accelerator, Internet on for the GitHub clone.
4. List mounted external ZIP/CSV sources. Verify the **actual** timestamp string format from the raw source header and sample timestamps **without revealing IP addresses or flows in output**. Do not guess AM/PM, day/month order or timezone. Then set `DATA` and `TIMESTAMP_FORMAT` explicitly in Notebook 25.
5. Execute `--max-rows 1000` G0 smoke test. Its result is always `G0_NOT_CLEARED` by design. After fixing any real data schema/time issues, full scan omits `--max-rows` and writes a new audit JSON. Even its `G0_SCHEMA_CHECKED_TEMPORAL_VALIDITY_PENDING` is **not a full causal-certification PASS**: flow end/start semantics, session linkage and attack labels remain to be audited.
6. Public GitHub: commit only metadata/checksums, source day/class histograms, aggregate audit JSON and decisions. Keep raw endpoints, flow IDs, rows and private trained models outside the public Git history; original artifacts in checksum-verifiable private draft releases. This is the same E6/M1 provenance practice.
7. **Never** run external fitting before G0, no official UNSW test access, and do not claim external portability from an internally retrained external-data model; that is an independent-dataset replication, not true UNSW → CIC zero-shot transfer.

Example command **only after timestamp examples are inspected and format verified**:

```bash
python scripts/e7_g0_external_source_audit.py \
  --data /kaggle/input/YOUR_DATASET/GeneratedLabelledFlows.zip \
  --timestamp-format '%d/%m/%Y %H:%M:%S' \
  --max-rows 1000 \
  --output /kaggle/working/E7-EXT-VALIDITY-001/g0_smoke.json
```

**Warning:** The `--timestamp-format` string above is illustrative; it is **not** verified for the publisher dataset and must not be copied without checking real timestamp values.

**Status as of 2026-10-19:** six source-gate tests passed on synthetic data, no external real dataset mounted, no external models fitted, no external accuracy/macro-F1 measurements. See the [preregistration](E7_EXTERNAL_VALIDATION_PREREGISTRATION.md) and [novelty audit](E7_NOVELTY_PRIOR_ART_AUDIT_2026.md).
