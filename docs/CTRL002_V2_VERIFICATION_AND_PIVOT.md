# V2 verification and decision to pivot to competitive tabular baselines

**Observed Kaggle result commit:** [76cef1f998608d6d25ea383fe86157d79e90309c](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/76cef1f998608d6d25ea383fe86157d79e90309c)

**User-requested strategic pivot 2026-10-18:** Stop spending GPU budget on CNN feature collapse diagnostics and incremental activation repairs; prioritize faster, competitive models producing actual measured internal validation results. Avoid any "perfect accuracy" guarantee or claims of exceeding the 2026 published paper on different splits.

## Independently verified V2 result

- Original split lineage frozen; **35,005** internal validation windows; trained **50 epochs**, best epoch **45**.
- **28,867 / 35,005** correct; **accuracy 82.46536209%**, **macro-F1 55.24639554%**. Independently recomputed exactly from publicly synced 10×10 confusion matrix; model scores and validation labels match the submitted evidence.
- V0 reconstructed CNN–ViT: 55.8664% macro-F1; V1 trained ViT-only: 57.4409%; V2 final-CNN-LeakyReLU: 55.2464%. **V2 is worse than both on the matched internal validation targets.**
- Best CNN branch diagnostic over **512 selected internal validation** windows: 100% nonzero token values, but **zero between-example token standard deviation**, 100% constant token dimensions and pooled features, zero changed predictions under CNN masking, and mean logits L2 shift of 0.005220968. The last CNN activation changed from exactly zero to a constant nonzero value, not useful instance variation, on the observed sample.
- **54 figures (18 × PNG/SVG/PDF)** and **eight numerical CSV source tables** are publicly versioned. Checkpoint, probabilities, history and figure sources are preserved in a verified draft GitHub Release, **132,751,360 bytes (126.60 MiB)**, asset ID `653892741`, GitHub-reported SHA-256 **8f6aef8b20a922a69e0d4dd5368c7faf94b9558197ee697b4acc9ac68cdb2234**, matching user's Kaggle SHA. Sidecar checksum file 104 bytes, asset `653893015`, present.
- [Unpublished draft release](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-b86560b60bc745c6e454); review for secrets/licensing before making it public.

## Scientific interpretation boundary

We confirmed a failed *independently reconstructed model* and its minimal engineering repair. We did **not** confirm the original paper's author code has a similar defect. This control **does not support model superiority**; it supports ending the single-activation CNN repair line.

## Next fast, results-focused experiment

**[Notebook 10](../notebooks/10_fast_tabular_benchmarks.ipynb)** implements **two efficient LightGBM ten-class multiclass classifiers**, standard and square-root rare-class weighted, on **42 original network-flow features** including native categorical proto/service/state. Fit on precisely **140,328** original training *prediction-target rows*; evaluate on the exact **35,005** matched original SCV validation target labels. The official designated test set stays untouched. Seeds are fixed at 42, and early stopping is on validation macro-F1; this is model development, not unbiased generalization.

- Save actual observed aggregate scores, class-support/precision/recall/F1, confusion and all model probabilities/checkpoints. Preserve source and split SHA-256 and original label order.
- Compare V0/V1/V2 in the same internal validation target scope; provide no invented score claims.
- **Decision gate:** if a tabular model is stronger, investigate a rare-attack-confusion mechanism on tabular features and a controlled image-versus-tabular study rather than continuing to add weak CNNs. Strong network IDS performance may not require pseudotemporal traffic-image serialization.
- Novel research requires independent classes, multiple seeds, same-target strong baseline (e.g., CatBoost), leakage/duplicates awareness and post-selection external evaluation. **Do not use the official test for tuning**.

**Status:** LightGBM code/notebook committed, **Kaggle training and actual LightGBM scores pending**.
