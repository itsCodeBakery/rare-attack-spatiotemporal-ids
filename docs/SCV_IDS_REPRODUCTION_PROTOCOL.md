# SCV-IDS Reproduction Protocol — Experiment SCV-IDS-REPRO-001

**Status: implementation complete, model execution pending (2026-10-18).**

Reference: Jue Chen *et al.*, "SCV-IDS: CNN-ViT Crossattention on Serialized Traffic Images for Spatiotemporal Intrusion Detection," *IEEE Internet of Things Journal*, 13(15), 2026, DOI [10.1109/JIOT.2026.3695912](https://doi.org/10.1109/JIOT.2026.3695912). Consult the user's attached article, especially Sections III–IV and Table II. The copyrighted original PDF is **not** committed to this repository.

## 1. Fidelity matrix

| Paper detail | Paper-reported | Our baseline reconstruction | Status |
|---|---|---|---|
| Classes | UNSW-NB15 Normal + 9 attacks | All ten official classes | Matched |
| Input selection | Remove ID and encode categorical features by index | 42 input features, categorical integer indices | Matched in principle |
| Scaling | MinMax scaling [0,1] | Train-fold-only MinMax; unknown values clamped | **Leakage-safe adaptation** |
| Image serialization | 5 consecutive records → 210 values → 16×16 RGB zero-pad | Same dimensional layout; quantization uses round(normalized×(2²⁴−1)) | **Quantization reconstructed** |
| Target | Label of newest record in FIFO window | Same | Matched |
| ViT | Patch 8×8, depth 4, heads 8, 128 dim, MLP dim 256 | Identical dimensional values, CLS excluded from crossattention | Matched in principle |
| CNN | Two 3×3 convolutions/pool twice, final 3×3 stage; channels 32,64,128 | Output flattened to 16 tokens of 128 dimensions | Matched in principle |
| Fusion | Bidirectional ViT→CNN and CNN→ViT crossattention; mean pool | MultiheadAttention with 8 heads, LayerNorm, Linear 256→128, GELU | **Projection sizes reconstructed** |
| Classification head | Two FC layers, GELU and dropout | 128→128→10, dropout 0.2 | **Hidden size reconstructed** |
| Loss | FocalLoss | Multiclass focal γ=2, no α | **Gamma/alpha unspecified by paper** |
| Optimizer | AdamW | AdamW LR .001, weight decay .001 | Matched |
| Epochs | 50 | 50, all preserved in history | Matched |
| Batch | 64 | 64 | Matched |
| Validation | 0.2 | Group-disjoint 5-fold with one approximately 20% fold | **Leakage-safe adaptation** |
| Temporal semantics | Claimed sequential flow continuity | **Original CSV has no trustworthy timestamps; ordered-record serialization only** | **Not verifiable** |
| Final test | Official UNSW-NB15 test partition | **Not accessed in Step 7** | Locked |

**No authors' source code has been incorporated**, and some implementation details are absent from the published article. We therefore call this a **paper-aligned independent reconstruction**, not an exact implementation replication.

### Leakage safeguards and limitations

The official 175,341-row designated training set is partitioned into group-disjoint development training and validation, **before** index encoding, MinMax and window creation. Group identities use all 42 originally parsed features. Rows retain relative CSV order within each partition, but removal of the other split creates gaps. Thus windows are formed from **consecutive surviving rows of a partition**, which are *not* necessarily consecutive original flows; this is explicitly an artificial ordered-record convention. It is not equivalent to time-consistent segmentation and must not be marketed as such.

Validation-only checkpoint selection uses macro-F1. The untouched 82,332-row official test remains locked. The earlier data-audit findings of 1,302 shared input signatures across official train and test and 548 label-conflicted shared groups remain disclosed. **No claims of beating the published 85.77% accuracy and 46.87% macro-F1 until a fair official-test evaluation on a frozen model.**

## 2. Reproducibility and retention

Kaggle notebook: [05_scv_ids_baseline.ipynb](../notebooks/05_scv_ids_baseline.ipynb). Configuration: [step07_scv_ids_repro_001.json](../configs/step07_scv_ids_repro_001.json).

### Saved in Kaggle output (version the notebook!)

- Training/validation split indices with label IDs in `split_indices.npz`.
- Every epoch: `training_history.csv`, latest resumable checkpoint, best validation checkpoint, periodic full checkpoints every five epochs.
- Full `validation_predictions.npz` with ground-truth labels, predictions, probabilities, and validation-row lineage.
- `embedding_vectors.npz` with stratified raw/CNN/ViT/fused features for independent projection recomputation. **Post-crash correction:** use [safe_projection.py](../src/rareids/safe_projection.py) for deterministic PCA; report collapsed or constant representations explicitly. Initial t-SNE postprocessing terminated natively after 50 trained epochs, so no t-SNE figure from that run is claimed.
- `preprocessing.json`: ordered feature list, categorical vocabularies, training-only min and max parameters.
- `config_effective.json`, `environment.txt` (pip freeze), `artifact_manifest.json` with SHA-256 hashes and git commit.
- `plot_data/`: individual source tables for every chart and reproducible class-image pixels.
- `figures/`: PNG 240 dpi, editable vector SVG, vector PDF for all 28 plots.
- `figure_manifest.json`: exact plot sources and SHA-256 hashes for all exported images.

### Automatically synchronized to GitHub using `pushRare`

`results/SCV-IDS-REPRO-001/` contains configuration, manifests, freeze, compact numeric CSVs, metrics, every SVG/PDF/PNG figure, and plot code already lives under `scripts/`. Model binaries and individual row-level artifacts are **not pushed** to a public repository; they stay in the saved Kaggle version, and GitHub stores their SHA-256 hashes and paths.

**Never publish a synthetic figure as experimental evidence.** Synthetic data used in automated tests is ephemeral and exists only to check plotting functionality.

## 3. Figure inventory (28 scientific plots)

All 28 are rebuilt by `scripts/plot_scv_figures.py --run RESULTS_DIR` using the observed `plot_data/*.csv`, without retraining, with stable figure names and source-content hashes.

| # | Figure | Source(s) |
|---:|---|---|
| 01 | Train vs validation class support | split_class_counts |
| 02 | Log-scale training class imbalance | split_class_counts |
| 03 | Ten class-specific serialized RGB examples | rgb_sample_pixels |
| 04 | Per-class example luminance density | rgb_sample_pixels |
| 05 | Training and validation focal loss | training_history |
| 06 | Training accuracy | training_history |
| 07 | Validation accuracy | training_history |
| 08 | Validation macro-F1 | training_history |
| 09 | Weighted vs macro-F1 | training_history |
| 10 | Learning-rate trace | training_history |
| 11 | Duration per epoch | training_history |
| 12 | GPU allocated memory by epoch | training_history |
| 13 | Training vs validation accuracy | training_history |
| 14 | Confusion matrix: absolute counts | confusion_validation |
| 15 | Confusion matrix: row-normalized | confusion_validation |
| 16 | Precision, recall and F1 by class | per_class_validation |
| 17 | F1 ranking and weakest classes | per_class_validation |
| 18 | Analysis/Backdoor/DoS/Shellcode/Worms recall | per_class_validation |
| 19 | F1 vs class support | per_class_validation |
| 20 | True vs predicted class proportions | confusion_validation |
| 21 | Validation reliability diagram | reliability_bins |
| 22 | Confidence histogram, correct and wrong | confidence_histogram |
| 23 | OvR ROC curves for ten classes | roc_pr_curves |
| 24 | OvR precision–recall curves | roc_pr_curves |
| 25 | Stratified raw-feature PCA | embedding_coordinates |
| 26 | Stratified CNN-feature PCA | embedding_coordinates |
| 27 | Stratified ViT-feature PCA | embedding_coordinates |
| 28 | Stratified fused-feature PCA | embedding_coordinates |

PCA projections from saved embeddings are exploratory and not evidence of generalization, and the curves are downsampled to at most 301 plotted points **per class** while the exact AUC/AP scores are computed from all prediction probabilities and stored in `metrics_validation.json`.

### Figure editing later

For style, font, legend, axis or caption revisions:
1. Update versioned `scripts/plot_scv_figures.py` on GitHub.
2. In the saved Kaggle notebook/output, rerun only `python scripts/plot_scv_figures.py --run /kaggle/working/research_outputs/SCV-IDS-REPRO-001`; **no model retraining**.
3. Regeneration updates `figure_manifest.json` and `artifact_manifest.json` hashes.
4. Rerun the `sync_scv_results.py` cell; GitHub keeps the prior figure versions in Git history.

To change the *statistic*, e.g., validation confidence intervals, use `validation_predictions.npz` from the saved Kaggle version and version the updated data-aggregation code. Never manually edit an output chart to imply changed measurements.

## 4. Honest baselines and next gates

- Preliminary CatBoost group-disjoint validation produced 55.41% macro-F1, **on an internal split**. The SCV validation result in Step 7 is comparable only if the splits are matched exactly; this requires explicit split-index comparison (the same grouped rule and seed are used but different window labels).
- Run the Step 7 experiment once and inspect loss/metrics, completed training epochs, class confusion, image representations and source hashes.
- Repeat key baselines over multiple seeds and record uncertainty before freezing a final architecture; do not tune on official-test results.
- Compare the published SCV-IDS baseline on the untouched official test **only after** the training-only protocol, checkpoints, evaluations and uncertainties are agreed. Secondary overlap/nonoverlap subsets must be reported transparently.

## 2026-10-18 postprocessing SIGSEGV correction

The original Kaggle training process completed **50/50 epochs**, then terminated during scikit-learn PCA/t-SNE feature plotting. The successful training is **not repeated**. A [recovery-only script](../scripts/recover_scv_figures.py) reconstructs metrics and 28 figures from saved probabilities/embeddings. A [recovery-only notebook](../notebooks/06_scv_postcrash_recovery.ipynb) must be run in the **same existing Kaggle session**; it does not reexecute the model. Figures 25–28 are now **PCA**, never claimed to be t-SNE. A [full-artifact archiver](../scripts/archive_scv_outputs.py) bundles original checkpoints, prediction probabilities, split indices, embeddings, plot data and figure files in a single downloadable `.tar` file. The archive is not uploaded to public GitHub and must be saved/downloaded before the Kaggle working environment expires.
