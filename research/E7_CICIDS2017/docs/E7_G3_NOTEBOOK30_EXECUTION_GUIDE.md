# Notebook 30 — Day-Disjoint B0/B1/B2 Baseline Execution

**Current phase:** `E7-G3-OPENWORLD-DAYHOLDOUT-001` — original CIC-IDS2017, exploratory, 4 frozen day folds × 5 seeds = 20 LightGBM model fits. Notebook published on 20 October 2026. **No real G3 model scores or private-model artifacts have been produced by CI.**

## Single Kaggle entrypoint

[Notebook 30](../notebooks/30_e7_g3_openworld_dayholdout_b012_lightgbm.ipynb)

Attach existing `dsshayan/dscicids` dataset with the exact `TrafficLabelling ` directory (trailing space). Enable Internet and the existing Kaggle secret named `pushRare`. CPU is enough for LightGBM; large source ingestion and 20 fits may take substantial time. Kaggle runtime and Internet limits are not controlled by the notebook.

1. Run **cell 1** to verify all four pinned scripts/configs, locked four folds, 5 seeds, loss costs and installed libraries.
2. Run **cell 2** before sleeping. It creates/reuses one **unpublished GitHub draft release**, uploads a harmless test ZIP and downloads it to verify authenticated private asset integrity. **If this cell fails, do not start training**; model checkpoints must never be uploaded as public repository files.
3. Run the remaining cells. The 8 original source CSV SHA256, source-derived empty-record removal, class counts, and no-identifier feature policy are checked before any fit. The notebook executes B0 (train prior), B1 (static-flow LightGBM at 0.5), and B2 (same model at train-independent validation threshold).
4. After each fitted seed, a model ZIP is saved under `/kaggle/working/E7_G3_DAYHOLDOUT_001/private_models/` and backed up only as an unpublished draft release asset. **The asset is independently downloaded and checked before the test day is evaluated.** Aggregate-only metrics are then create-only saved to the public results directory. A failed private upload/verification immediately halts further training.
5. Once all 20 runs complete, a four-fold, five-seed mean/std aggregate summary is separately uploaded and byte-verified.

## Frozen protocol

- [G3 day-disjoint manifest](../configs/G3_OPENWORLD_DAYHOLDOUT_PROTOCOL_001.json)
- [G3 model hyperparameter lock](../configs/G3_B012_HYPERPARAMETER_LOCK_001.json)
- [Scientific experimental preregistration](E7_G3_OPENWORLD_DAY_DISJOINT_BENCHMARK_PREREG_2026_10_20.md)
- [Source-hash locked numeric-flow training implementation](../scripts/e7_g3_openworld_baseline.py)
- [Private draft release and public-safe metric publisher](../scripts/e7_g3_archive.py)

**Output after actual Kaggle execution:** `research/E7_CICIDS2017/results/G3_OPENWORLD_DAYHOLDOUT/`, 20 seed-level safe public metrics and one summary. Private draft release tag: `e7-g3-openworld-dayholdout-001-private-models`, which must **never** be published. The release also holds a non-sensitive upload/download preflight probe.

The user must not delete old G0/G1/G2, frozen UNSW official or M1 artifacts; they are historical evidence. No raw traffic, source IPs, flow identifiers, rowwise predictions or fitted models may be committed publicly.

### Run status and scientific caveat

GitHub Actions validates 54 synthetic tests (including an **actual tiny LightGBM fit with model-ZIP restoration**) and the notebook's four code cells. **This does not prove live Kaggle API permissions or real model scores**. All labels are provisional original CIC-IDS2017 labels; day-specific attack families and capture settings are confounded, and no chronological related-flow input is used. G3 is not directly comparable to the base SCV-IDS random-split CIC result. No UNSW-NB15 official test is accessed.
