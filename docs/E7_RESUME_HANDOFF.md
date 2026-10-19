# E7 resumption handoff — external validation before journal targeting (2026-10-19)

**Approved by researcher:** run external-data validation and review 2024–2026 methodological novelty before choosing journal. **Remain in the same project/repo**, preserve earlier E3/E4/E5/E6/M1 evidence and one-time official UNSW test exactly unchanged.

## Completed and committed — no results fabricated

1. [2024–2026 publisher-verified original-literature audit](E7_NOVELTY_PRIOR_ART_AUDIT_2026.md): identifies existing class-imbalance hierarchies (HICID, HybridGuard, HIDIM), mixtures (EfficientNet-MoE, XGB-AGMoE, MHFED), uncertainty, concept drift and data-quality works. **MoE/gating/imbalance cannot be claimed novel alone.** The original SCV-IDS paper already evaluated the **CICIDS2017 Wednesday subset** with 80:20 supervised partition.
2. [New external-validation preregistration](E7_EXTERNAL_VALIDATION_PREREGISTRATION.md): original UNB `GeneratedLabelledFlows.zip` as first source, separate corrected-flow sensitivity (Engelen et al.), later CIC IoT-DIAD 2024, explicit chronological and causal flow context; distinct external retrain vs zero-shot transfer; strong controls and five seeds; no use of earlier official UNSW test.
3. [E7 G0 read-only source gate](../scripts/e7_g0_external_source_audit.py) and [six synthetic tests](../tests/test_e7_g0_external_source_audit.py), tested locally on entirely synthetic data: **6 passed**. Uses explicit timestamp format; public artifacts never contain IP addresses or original flow values; fails missing required metadata; **G0 checks alone never certify causal time**.
4. [Notebook 25](../notebooks/25_e7_external_g0_provenance_audit.ipynb): Kaggle CPU clone+synthetic tests; list candidate mounted external sources; timestamp-only preview; explicit source/time-format gate; 1000-row G0 smoke. [Quickstart](E7_G0_QUICKSTART.md).
5. Decision recorded in [DECISION_LOG](DECISION_LOG.md).

## First pending user action — single step

**Obtain original `GeneratedLabelledFlows.zip` from the [UNB publisher](https://www.unb.ca/cic/datasets/ids-2017.html) and attach the archive *unchanged* to Kaggle.** If Kaggle auto-unzips it, show the directory tree. Then run Notebook 25 only through the safe dataset filename and timestamp preview, share output. Do **not** run any training or claim external score yet.

## What next ChatGPT continuation must do

- Read this handoff + GitHub source; do not start from scratch or propose unrelated CIC 80/20 training.
- Accept user's actual Kaggle source path / original time strings, determine confirmed timestamp format and source version.
- Run G0 smoke then full scan, commit aggregate-only SHA manifest, audit flows’ *completion* and related session semantics, version the actual split/test.
- Only after G0 review, implement G1 CPU LightGBM controls with autosync after every expensive stage to **private GitHub draft releases** and public progress commits, as established in E6. Never leave an hours-long Kaggle job without crash-safe checkpoint+GitHub progress.
- Preserve negative results and classwise rare attack recalls. Return no journal ranking until external results and genuine novelty established.

**Status:** novelty threats mapped; external experimental validation **NOT YET EXECUTED** because no publisher external source has been attached/audited. **Do not** claim M1 C4 superiority: calibration was infeasible and cap alpha=0; C3's improvement is exploratory.
