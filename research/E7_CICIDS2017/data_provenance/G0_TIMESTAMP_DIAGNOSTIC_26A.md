# Notebook 26A — CIC-IDS2017 timestamp parser diagnosis

**Status:** User-executed Kaggle **read-only 20-row timestamp preview per file**, 2026-10-19. **Full file parsing not yet performed.** No original traffic, model training, or official UNSW test touched. Source: user supplied Notebook 26A console output.

## Actual attached source

- Mount base: `/kaggle/input/datasets/dsshayan/dscicids`.
- Parent directory (exact): `TrafficLabelling ` — the trailing space is part of the filename.
- Original timestamp header in first inspected file: ` Timestamp` — leading space.
- Eight CSV files identified in the user-provided listing. The upload is a user-asserted publisher-style CIC-IDS2017 labeled-flow archive, **not yet independently publisher-authenticated**.

| File | Original timestamp sample | Exact parser matching sample weekday | Weekday check |
|---|---|---|---|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | `7/7/2017 3:30` | `%d/%m/%Y %H:%M` | Friday |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | `7/7/2017 1:00` | `%d/%m/%Y %H:%M` | Friday |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | `7/7/2017 8:59` | `%d/%m/%Y %H:%M` | Friday |
| Monday-WorkingHours.pcap_ISCX.csv | `03/07/2017 08:55:58` | `%d/%m/%Y %H:%M:%S` | Monday |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | `6/7/2017 1:00` | `%d/%m/%Y %H:%M` | Thursday |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | `6/7/2017 8:59` | `%d/%m/%Y %H:%M` | Thursday |
| Tuesday-WorkingHours.pcap_ISCX.csv | `4/7/2017 8:54` | `%d/%m/%Y %H:%M` | Tuesday |
| Wednesday-workingHours.pcap_ISCX.csv | `5/7/2017 8:42` | `%d/%m/%Y %H:%M` | Wednesday |

**Ambiguity:** Friday's 7/7/2017 date is compatible with both month/day and day/month representations, but the other weekdays disambiguate the overall corpus in favor of **day/month/year**. This was verified only on the first 20 samples per file; format consistency across all records remains an open G0 check.

### Blocking clock-hour ambiguity

Friday Afternoon DDoS appears to begin with `3:30`, Friday Afternoon PortScan with `1:00`, and Thursday Afternoon Infiltration with `1:00`, **without AM/PM**. A 24-hour parser represents these as 03:30 or 01:00, whereas the file names suggest afternoon episodes and potentially 15:30 or 13:00. Do not silently add 12 hours or infer a timestamp timezone; further source/capture-time verification is mandatory before constructing causal history.

### Corrective action

- Original Notebook 26 safely stopped before the full audit when it tried the minute-only parser on Monday. **No E7 accuracy score exists.**
- [Frozen eight-file parser registry](../configs/e7_g0_timestamp_registry_001.json) registered based on Notebook 26A evidence, not tuned from any classifier result.
- [Notebook 26B](../notebooks/26b_e7_g0_per_file_timestamp_autosync.ipynb) applies the appropriate registered parser to all rows separately, validates weekday conformity, records clock ambiguity as `G0_NOT_CLEARED` and commits safe aggregate reports to GitHub after each full original CSV scan. No model fit.
- Next Notebook 27 must establish actual clock convention, flow completion time/duration unit, endpoint/session linkage, duplicates and corrected-source quality before chronology or external-validity claims.

Do not reinterpret the earlier frozen SCV-IDS/UNSW results or change M1's failed preregistered C4 endpoint.
