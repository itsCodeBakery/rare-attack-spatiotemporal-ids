# SCV-IDS reproduction — recovered postprocessing

**Internal group-disjoint validation only; NO official test evaluation.**

- Training: 50/50 epochs completed in original Kaggle process.
- This recovery: reuses frozen predictions and embeddings; **no retraining**.
- Best-checkpoint validation macro-F1: **0.558664**
- Accuracy: **0.827396**
- Weighted-F1: **0.812089**
- Best checkpoint selected at epoch: **41**
- Validation window labels: **35,005**
- Preserved checkpoints: **12**
- Figures: **28** × 3 formats = **84** exports
- PCA diagnostics: **{'raw': 'ok', 'cnn': 'zero_variance', 'vit': 'ok', 'fused': 'ok'}**
- Original training Git SHA: `unavailable; original environment file missing`
- Original training environment.txt present: **False**
- Recovery code Git SHA: `f3dc7dd70bf0c2fa12118b89da38060af4a0febc`
- Original best model checkpoint SHA-256: `1ff947f795162b747115fee6e3131eb10fd3955445a3040902a274fa71457c65`

The figure scripts, source CSV tables, metrics and SHA-256 manifest are synchronized
separately. Original checkpoint files, raw per-row probabilities, split indices and
embeddings must be **preserved in a Kaggle saved output version**.

PCA is intentionally used instead of the native t-SNE routine that crashed.
If a representation has no measurable variation, its plot is explicitly marked
as uninformative; no artificial jitter or fake separation is introduced.
CSV row order remains unvalidated as true chronological flow order.
