# CTRL-CNNVIT-V2-S42 — minimal repaired CNN final activation

- Original paper-aligned reconstructed SCV, macro-F1: 0.558664 (internal validation)
- Corrected CNN-ViT macro-F1: **0.552464** (internal validation)
- Standalone ViT control macro-F1: 0.574409
- CNN activation-health gate passed: False
- CNN token nonzero rate: 1.000000
- CNN masking logit L2: 0.005221
- Original SCV accuracy: 0.827396; corrected CNN-ViT accuracy: **0.824654**
- Identical validation window labels: **YES**, 35,005 examples.
- Original fold and fitted preprocessing are reused unchanged.
- Best checkpoint epoch: **45 of 50**.
- Figures: **18** × PNG/SVG/PDF.
- Training seeds: **one**; no significance/generalization claim.
- Official test used: **NO**.
- A single activation repair is an engineering control, not a new scientific method.
- Preserve checkpoints/predictions in private saved Kaggle output; GitHub stores only hashes.
