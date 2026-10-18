# CTRL-VIT-001-S42 — validated matched-split standalone ViT control

- Original paper-aligned reconstructed SCV, macro-F1: 0.558664 (internal validation)
- Standalone ViT macro-F1: **0.574409** (internal validation)
- Original SCV accuracy: 0.827396; standalone ViT accuracy: **0.832367**
- Identical validation window labels: **YES**, 35,005 examples.
- Original fold and fitted preprocessing are reused unchanged.
- Best checkpoint epoch: **44 of 50**.
- Figures: **14** × PNG/SVG/PDF.
- Training seeds: **one**; no significance/generalization claim.
- Official test used: **NO**.
- This control is not a proposed novel method; it benchmarks a genuine standalone ViT.
- Preserve checkpoints/predictions in private saved Kaggle output; GitHub stores only hashes.
