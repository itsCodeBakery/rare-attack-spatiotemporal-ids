# Recovery of SCV-IDS-REPRO-001 after native t-SNE crash

**Incident:** The user reported successful completion of all 50 training epochs in Kaggle on 2026-10-18; the final epoch printed 0.54320 validation macro-F1 and the logged best value 0.55866. After frozen validation predictions and embeddings were saved, scikit-learn's PCA/t-SNE stage warned about invalid variance normalization and the subprocess terminated with SIGSEGV 11.

**Important:** This does **not** prove that t-SNE itself, or a specific BLAS/NumPy binary, caused the segmentation fault. It is a native failure in the postprocessing region, not evidence that epoch 50 failed.

## Recovery implemented

- [scripts/recover_scv_figures.py](../scripts/recover_scv_figures.py) reads preexisting original training outputs. It neither opens a source CSV nor starts model training/inference.
- [notebook 06](../notebooks/06_scv_postcrash_recovery.ipynb) provides the exact two recovery/sync cells to paste and execute **in the original Kaggle session**, avoiding another 50-epoch run.
- Saved evidence required: `training_history.csv`, `validation_predictions.npz`, `split_indices.npz`, `embedding_vectors.npz`, original checkpoints (best/latest) and the eight pre-crash plot-data tables.
- The same frozen probabilities are used to reconstruct ten-class accuracy, macro-/weighted-F1, balanced accuracy, Normal false-positive rate, one-vs-rest AUC/AP and confidence calibration. It validates confusion source data and window-level lineage.
- The unstable t-SNE step is **not retried**. New [variance-safe PCA](../src/rareids/safe_projection.py) explicitly marks collapsed/constant representations. We never call this PCA projection a t-SNE result.
- Regenerate all 28 charts from their **saved numeric source tables** and the frozen PCA projection coordinates. Keep `figure_manifest.json`, `projection_diagnostics.json`, `metrics_validation.json`, `run_report.md` and `artifact_manifest.json` versioned.
- The recovery report retains the original source-code SHA from `environment.txt`, separately records the recovery-code SHA, and preserves cryptographic hashes of all checkpoints and row-level artifacts.
- GitHub's figure sync is atomic and includes all PNG, SVG, PDF files and compact CSV plot sources. The saved Kaggle output must be persisted separately for binary checkpoint and raw validation prediction reuse.

## Current status

The user-supplied epoch log is **provisional** until the recovered metrics are successfully synchronized and recomputed from frozen validation predictions. No official-test inference or claim of beating SCV-IDS is authorized. The baseline still uses row-order serialization, not verified chronological traffic.

## Risks

If the original Kaggle session or `/kaggle/working` files are gone and there is no saved run output version or downloaded checkpoint archive, the full training run may have to be repeated. We cannot claim successful preservation of those files until they are independently verified or saved.
