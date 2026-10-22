> **Verified research status — 22 October 2026: G4 complete.** All 20 validation-only risk-calibration/abstention reports and [the five-seed summary](results/G4_RISK_ABSTENTION/G4_FOUR_FOLD_FIVE_SEED_SUMMARY.json) are archived. See the [G4 post-run scientific analysis](docs/E7_G4_POSTRUN_SCIENTIFIC_INTERPRETATION_2026_10_22.md). M1 improves retrospective Macro-F1 on three of four original-label source days but **Wednesday attack recall is zero**, and Friday seed 45 exceeds the 1% test false-positive target. M2 makes **zero auto-benign decisions in all 20 runs** and fails the predefined minimum 50% test coverage in every run; do not claim successful operational abstention. The same G3 test sets were already inspected, so these are **exploratory findings only**. Next scientific gate: independent corrected/external capture source and fully preregistered confirmatory risk/recall/coverage evaluation; no G5 result or new model exists yet. All older experiments and unpublished private models are preserved.

> **CURRENT NEXT KAGGLE RUN — 22 October 2026:** G3 is complete: all 20 original four-fold/five-seed results and one aggregate summary verified in [G3 public results](results/G3_OPENWORLD_DAYHOLDOUT/). **Run [Notebook 31 — G4 validation-only risk calibration and abstention](notebooks/31_e7_g4_validation_risk_and_abstention.ipynb)**. [G4 protocol and Kaggle guide](docs/E7_G4_RISK_CALIBRATION_PREREG_AND_KAGGLE_GUIDE.md) explain the locked M1 false-alarm control and M2 abstention; 20 original private G3 models are restored read-only, not retrained. [CI](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39721921712) passed 73 synthetic tests, checked six source blob IDs and compiled all five notebook code cells. **Actual G4 execution has not yet occurred.** G3 test-day outcomes are already known: G4 reuse is retrospective exploratory only, never independent confirmation or direct comparison to base SCV-IDS random splits. All G0/G1/G2/M1 and frozen UNSW originals remain preserved.

> **Correction and next Kaggle step — 21 Oct 2026:** **Use [Notebook 30B](notebooks/30b_e7_g3_recovery.ipynb), not Notebook 30.** The first G3 fold-1 seed-42 model trained and its *private, unpublished* GitHub model ZIP upload was verified. Public metrics publication then stopped solely because the publisher treated the 40-hex Git blob protocol identifier as if it were a 64-hex SHA-256. [Publisher v2](scripts/e7_g3_archive_v2.py) corrects both seed metrics and the final summary, keeping SHA-256 checks for model assets intact; it uses a new pinned code snapshot **but the same private model and result directories**, with local/remote checkpoint recovery before retraining. [Automated regression run](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39540073582) passed **63 synthetic tests**, including this exact mismatch, and compiled Notebook 30B. **No G3 accuracy result is claimed yet; the user must execute 30B.**

> **Current research run — 20 Oct 2026:** G0, G1 and G2 read-only CIC-IDS2017 source audits are complete. **Notebook 30 is the next Kaggle experiment:** [G3 frozen day-disjoint LightGBM B0/B1/B2 (4 test days × 5 seeds)](notebooks/30_e7_g3_openworld_dayholdout_b012_lightgbm.ipynb). [Execution guide](docs/E7_G3_NOTEBOOK30_EXECUTION_GUIDE.md) and [frozen hyperparameters](configs/G3_B012_HYPERPARAMETER_LOCK_001.json) are committed. [GitHub Actions](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39407757082) **passed 54 synthetic tests**, including a real tiny LightGBM train/restore and notebook syntax. **Actual Kaggle training and metric results are NOT yet run or claimed.** Private fitted models go only to an unpublished draft release after authenticated upload/download preflight; per-seed safe aggregate results go to `results/G3_OPENWORLD_DAYHOLDOUT/` only after the private model backup verifies. Original G0–G2, M1 and frozen UNSW-NB15 test results are unchanged. This original-label exploratory day holdout is not directly comparable to SCV-IDS CIC random split and provides no verified chronology.

> **Verified research status — 20 Oct 2026:** G2 Notebook 29 **completed** four forensic reports. Thursday WebAttacks contains **288,602 fully empty trailing CSV records**, leaving **2,830,743 populated source rows** across the eight original files. Monday/Tuesday/Wednesday CSV row ordering is not verified chronological (523,827 non-simple-12→1 reversals). **Next: G3 exploratory day-disjoint open-world baseline**, with the [four-fold protocol locked before model fitting](configs/G3_OPENWORLD_DAYHOLDOUT_PROTOCOL_001.json) and [full methodological preregistration](docs/E7_G3_OPENWORLD_DAY_DISJOINT_BENCHMARK_PREREG_2026_10_20.md). Notebook 30 is **planned, not yet implemented or executed**. Original labels and flow chronology remain unverified; never claim direct numerical superiority over SCV-IDS's different CIC protocol. Earlier notebooks remain preserved.

> **Latest validated workflow — 20 Oct 2026:** G0 and G1 are complete across all eight original CIC-IDS2017 CSV files. **Next and only required Kaggle run:** [Notebook 29: G2 source-integrity and clock forensics](notebooks/29_e7_g2_source_integrity_clock_forensics.ipynb). The notebook's syntactic validity and synthetic safety checks passed [GitHub Actions](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39403758615). **Actual G2 source execution has not occurred; no E7 IDS model fitting or external model accuracy is claimed.** The source has a 288,602-row contiguous WebAttacks missing-label tail and major clock-order reversals requiring investigation. See [G2 preregistration](docs/E7_G2_SOURCE_FORENSICS_PREREG_2026_10_20.md). Earlier notebooks are preserved as historical steps and not the new run target.

# E7 — CIC-IDS2017 external validation & methodological novelty

**Project:** Rare-Attack Spatiotemporal IDS · isolated E7 workstream  
**Status:** External CIC-IDS2017 archive attached in Kaggle; read-only G0 file/schema/timestamp validation **pending**; **no E7 models trained or scores reported**.  
**Research rule:** This folder is the canonical home for **all NEW public E7 research artifacts**. Nothing from E3/E4/E5/E6/M1 is relocated, overwritten, or re-evaluated. Original UNSW-NB15 official test may **not** be reused for model selection.

## Frozen earlier results are preserved separately

- [Original frozen official UNSW result](../../results/WINDOW-OFFICIAL-001-S42/official_metrics.json): 86.321786% accuracy and 58.139685% 10-class macro-F1. This test was already used once. **No more test evaluation.**
- [E6 contextual-order audit](../../docs/E6_ORDER_ROBUSTNESS_AUDIT.md): row neighborhoods do **not** verify chronology.
- [M1 seed-42 authentic six-control report](../../results/M1-CONTEXT-RISK-001/seed_42/final_public_summary.json) and [independent audit](../../results/M1-CONTEXT-RISK-001/seed_42/INDEPENDENT_AUDIT.md): global risk cap infeasible; C4 alpha=0, not a successful gain.
- The verified original 50.5 MB M1 seed42 artifact remains in an **unpublished private GitHub draft release**; original archive SHA-256 `00c440f79a3f88e71dc831281fd10776a06af4e57569f62d0b25a8c27910ab88`. Do not make draft releases public.

## Folder layout (E7 only)

| Folder/file | Role |
|---|---|
| `docs/` | Preregistered protocol, novelty differentiation, quickstart and resume handoff |
| `scripts/` | Reproducible E7-only Kaggle audit / future model code |
| `tests/` | Synthetic no-leakage and source-integrity tests |
| `notebooks/` | Kaggle notebook to audit actual attached source before training |
| `data_provenance/` | Publisher URLs, original file manifests, checksums, safe aggregate dataset audits (never raw records) |
| `results/` | Public run summaries, confusion matrices, metric tables, plots, and signed artifact manifests (no private predictions) |
| `EXPERIMENT_LOG.md` | Append-only decisions, run IDs, GitHub commits and evidence checkpoints |
| `ASSET_POLICY.md` | What belongs in Git vs private draft releases |

## Current user-confirmed external files in Kaggle

The source lists **eight** day/attack-segment CSVs from the CIC-IDS2017 `TrafficLabelling` archive, including Wednesday, Monday, Tuesday, Thursday, and Friday. The notebook's directory listing displays a **space immediately before the filename slash**:

`/kaggle/input/datasets/dsshayan/dscicids/TrafficLabelling /Wednesday-workingHours.pcap_ISCX.csv`

The folder may literally end in a trailing space. **Never guess or strip directory names.** Discover the real CSV parent using `Path('/kaggle/input/datasets/dsshayan/dscicids').rglob('*.csv')` and derive it from `files[0].parent`. Kaggle listing is not a complete G0 provenance audit. The original user attachment is called a ZIP and Kaggle automatically extracted it.

## Current step (read-only; no fitting)

1. Open [Notebook 25](notebooks/25_e7_external_g0_provenance_audit.ipynb) in Kaggle and list exact E7 CSV paths and just three raw timestamp **strings** (not entire flow rows or IP addresses).
2. Verify actual source metadata and timestamp format. Do not infer source-order causality or flow-end semantics from parsing alone.
3. Run synthetic audit tests, then G0 source-only smoke and full versioned G0 audit. Only after G0 is reviewed, freeze the external chronology-and-label protocol.
4. All expensive future training must automatically store each completed model and private probabilities in **checksum-verified unpublished GitHub draft releases**, with public aggregate progress synced after every stage. A Kaggle local-only success does not satisfy backup completion.

## Publication rule

We may begin manuscript drafting from existing frozen results. E7 novelty and external generalization remain **unproven until independently audited real experiments**. Do not conflate within-CIC training/testing with strict zero-shot UNSW→CIC transfer, or compare incompatible labels/splits as directly matched.

## Current executable Kaggle notebook — #26

**Notebook 26 (GitHub versioned):** [Eight-source CIC-IDS2017 full G0 audit with per-CSV verified public GitHub sync](notebooks/26_e7_cicids2017_eight_csv_g0_github_sync.ipynb).

The user has verified that the eight CSVs are mounted in the actual directory whose last component is `TrafficLabelling ` (**trailing space**) and that the Timestamp header is ` Timestamp` (**leading space**), with example `7/7/2017 3:30`. Notebook 26 uses exactly this source-root discovery, verifies the assumed minute-resolution timestamp format on **each** CSV before a full scan, and runs the version-controlled synthetic audit + uploader security tests. It checks the existing Kaggle `pushRare` secret **before** long operations and uploads **each independently completed original source aggregate report** to `results/G0/`, verifying remote bytes before advancing. No model training, no UNSW official test access, no automatic threshold tuning. A failed G0 data-quality report is retained as a flagged failure, not a PASS.

**Status:** prepared, code committed, notebook JSON and old results presence verified. **Kaggle real-data scan and remote live uploads have not yet been executed or confirmed.** After user runs Notebook 26, review source/day/label/time granularity and flow completion with a new Notebook 27, without changing the protocol retrospectively.

## Kaggle execution correction — Notebook 26C

**Use [Notebook 26C — safe separate checkout and verified per-CSV GitHub archive](notebooks/26c_e7_g0_safe_checkout_verified_sync.ipynb), not 26B.** After Notebook 26B failed on `git fetch --depth 1` then `git merge --ff-only` (unrelated shallow histories), 26C uses a new independent source folder `/kaggle/working/rare-attack-e7-cicids-26c-source`. It never resets or touches `/kaggle/working/rare-attack-e7-cicids`. Its Git snapshot is frozen and reused on rerun rather than synchronized with a moving branch. Existing local data and historical research results remain intact. The corrected checkout path passed an offline Git regression test (old workspace unchanged, new HEAD current), and the notebook JSON plus references were checked against GitHub. **Live Kaggle scan and GitHub uploads are not yet verified**; the user must execute Notebook 26C next.

## Current recommended notebook: overnight E7 G0 source audit (27)

**Run [Notebook 27 — overnight full-source validity/quality audit with verified GitHub autosync](notebooks/27_e7_overnight_full_audit_verified_github_sync.ipynb)**. This **replaces** Notebook 26C for continuation; 26C remains unchanged for historical provenance.

The original GitHub `PUT` for Friday PortScan succeeded; its post-PUT `GET` temporarily returned 404. Both Friday DDoS (225,745 rows) and Friday PortScan (286,467 rows) original source G0 reports have now been independently retrieved from GitHub. The new immutable [publisher v2](scripts/e7_g0_publish_v2.py) stores a receipt after PUT and verifies Git blobs or commit snapshots with retries; it does not overwrite existing evidence. Source is downloaded from a **pinned GitHub commit and checked against exact Git blob hashes**, with no Git fetch/merge/reset. An additive [GitHub Actions E7 synthetic test run](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39304349081) passed.

Notebook 27 runs G0 remaining files and per-file extended data-quality scans over all eight files, with safe aggregate-only uploads after each file, then a consolidated summary. Expect **up to 8 G0 + 8 G0_EXTENDED + 1 aggregate** GitHub reports if Kaggle completes. Raw CSVs, flow identifiers, IPs and fitted models are never committed. Dataset clock ambiguity/flow duration and labeling issues remain explicitly unresolved, and **no IDS training is attempted** before chronologically valid experimental design. Kaggle session duration is not guaranteed, but completed files are independently archived.

## Canonical overnight recovery — Notebook 27B

**Use [Notebook 27B](notebooks/27b_e7_overnight_resume_strict_encoding_and_github_sync.ipynb)** in place of Notebook 27, which stopped after five completed source pairs due to a strictly invalid UTF-8 byte `0x96` in `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`.

**Verified already preserved:** `results/G0/` and `results/G0_EXTENDED/` each contain the same original five files (3 Friday, Monday, Thursday Afternoon). **They are not to be overwritten or treated as failed.** Notebook 27B recovers them by file/source SHA256, then audits the remaining Thursday Morning, Tuesday and Wednesday sources. Strict original-byte encoding detection accepts valid UTF-8; if invalid, it tests Windows-1252 across the **entire file** and records `G0_NOT_CLEARED` for non-UTF8 text until labels are separately verified. Never use replacement/ignore errors or alter source CSVs. The 404-after-GitHub-upload recovery publisher is unchanged.

The independent [E7 GitHub CI execution](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39307651991) **passed 29 synthetic tests**, including a literal `0x96` fixture with byte-for-byte source preservation; all four Notebook 27B code cells compiled. **This is test validation, not proof of actual Kaggle completion.** As before, no model training, official UNSW test access, or causality claims are authorized until the separate provenance/clock audit is completed.
