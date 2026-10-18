# CARES-PAIR-001: archived six-way ablation audit and next-prototype decision

**Date:** 2026-10-18. **Actual Kaggle data verified from GitHub commit:** [c377b66a09f2bbfb40cc24ccdf748f3ebe3c0d9d](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/c377b66a09f2bbfb40cc24ccdf748f3ebe3c0d9d). Data are frozen training-derived *internal validation*, not designated official UNSW-NB15 test.

## Source preservation

- Full private archive: [GitHub draft Release](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-ffc196fae42b77bdb44e), internal tag `cares-pair-001-s42-full-artifacts`, **8,325,120 bytes (~7.94 MiB)**, SHA256 `61b65ab6faf74bd2c170b0bbdf78d508fdb14b39635aa03ee39be190f7a19f71`. The GitHub asset is uploaded and draft=true, with the checksum matching the Kaggle log. The raw archive has not been independently downloaded and rehashed.
- Exactly **66 files** synchronized publicly including **12 plot families × PNG/SVG/PDF = 36 scientific images**, **13 SHA-256-indexed numerical plot sources**, classwise metrics and full six confusion matrices. **Private** three specialist model binaries and 35,005 per-example outputs are stored in the unpublished Release, not public Git history.
- Verified all six committed confusion matrices independently: each sums to 35,005 and reproduces the reported accuracy and ten-class macro-F1; all model variants reuse the original identical validation labels.

## Independently verified results

| Model | Macro-F1 (%) | Accuracy (%) | Balanced acc. (%) | DoS TP / 2509 | DoS→Exploits | Backdoor TP / 350 | Backdoor→Exploits |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen weighted LightGBM | 63.6691 | 80.7142 | 65.5931 | **1976** | **431** | **57** | **42** |
| DoS specialist only | 63.6927 | 80.7799 | 65.4821 | 1922 | 486 | 57 | 49 |
| Backdoor specialist only | 63.5336 | 80.8970 | 65.1567 | 1841 | 574 | 50 | 70 |
| Analysis specialist only | 63.6112 | 80.9370 | 65.1767 | 1797 | 612 | 57 | 63 |
| All specialists unrestricted | 58.4622 | **82.1911** | 59.8107 | 98 | 2333 | 28 | 312 |
| All specialists selective | 63.4243 | 81.2398 | 64.4875 | 1537 | 881 | 50 | 105 |

**True conclusion:** all specialists combined **harm macro-F1**. The +0.0236 percentage-point improvement for the DoS-only variant is negligible and accompanies **54 lost DoS true positives** and reduced balanced accuracy. Do not state that the proposed method has outperformed its parent.

## Real classwise failure location

From actual frozen [baseline class metrics](../results/CARES-PAIR-001-S42/baseline/per_class.csv):

| Actual class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Analysis | 428 | 45.45% | **24.53%** | 31.87% |
| Backdoor | 350 | 42.22% | **16.29%** | 23.51% |
| DoS | 2509 | 35.44% | 78.76% | 48.88% |
| Exploits | 6684 | 81.78% | 57.20% | 67.31% |
| Worms | 21 | 52.94% | 42.86% | 47.37% |

Backdoor and Analysis are dominant classwise recall gaps; additional DoS probability inflation is not justified by the current data. With only 21 Worms examples, single-seed model selection is especially unstable.

## Why first CARES design failed

Three binary experts were calibrated on their respective training-derived pair data but applied sequentially to a common ten-class posterior; all three transfer probability to/from **Exploits**. Marginal binary conditional calibration is **not** joint ten-class calibration, and each transfer may flip argmax classifications outside that pair. A Backdoor-only correction changed DoS→Exploits count from **431 to 574**, despite not having a DoS model. The unrestricted version made 5,667 classification changes and devastated DoS recall.

## Next methodological research direction (preliminary, novelty not established)

**Cross-fitted, risk-constrained multiclass decision correction** rather than sequential binary probability transfers:

1. Create three **out-of-fold** predictions solely within development training (140,328 original target rows): fit independent LightGBM fold models using two folds, predict the third; never use the 35,005 reserved development targets to fit routing, costs, or thresholds.
2. Learn a **single multiclass residual risk decision module** on genuine out-of-fold predictions and known training labels. Its inputs can include calibrated baseline log-probabilities, top-two margins, entropy and the 42 features; it must not receive validation labels or provenance-leaking row IDs.
3. A risk-controlled decision accepts a correction only when the training-derived estimated reduction in misclassification harm exceeds a predeclared uncertainty tolerance; otherwise **retain parent argmax**. Penalize reduced Backdoor/Analysis recall and high false-positive cost for non-attacks, while constrain overall accuracy loss using *training-derived* calibration.
4. Compare identity baseline, simple classwise bias/temperature baseline, ungated residual correction, and risk-gated residual correction. Evaluate all on the same unchanged 35,005 development targets **without further tuning on them**. If there is no material macro-F1 improvement and preserved rare-class recall, stop and report null outcome.
5. Next repeat best candidates across 3–5 seeds and compare CatBoost/raw-flow baselines. Freeze all protocols *before* using the official 82,332-row test exactly once.

This is **a protocol for a second prototype**, not a proven novel theory. Related-work differentiation against 2024–2026 peer-reviewed IDS/cost-sensitive selective decision methods is required before novelty statements. With only tabular inputs, do not claim genuine temporal modeling. Do not compare present 35,005-window-target validation scores to the published SCV-IDS official test scores as if they share evaluation protocols.
