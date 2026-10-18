# CTRL-VIT-001-S42 — independent model-control audit

**Observed real-data commit:** [8b7baeada660b0203f48237b6f8ed9f793c55b77](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/commit/8b7baeada660b0203f48237b6f8ed9f793c55b77). Audited 2026-10-18. **This is internal training-derived group-disjoint validation, NOT official UNSW designated test.** The models are independent paper-aligned reconstructions, not verified author implementations.

## Genuine matched-target training control

V0: original CNN–ViT reconstruction, trained for 50 epochs, best epoch 41; sampled frozen CNN branch produces zero output. V1: separately trained ViT-only model on the **same original 140,332 training/35,009 validation rows**, **140,328/35,005 five-row windows**, **same 35,005 validation label targets in original order**, same train-fitted preprocessing, and seed 42/50 epochs/AdamW/focal gamma 2 budget. V1 uses the original ViTEncoder but mean-pools the tokens and has a different classification head without fusion. Thus, this is not a strict parameter-matched ablation.

Original designated training CSV SHA256: `bec7dd5ec88dc2a0ccc7a07879d338395ed742175fd0339e07dfe0648fa`. Preserved group-disjoint split SHA256: `01f6070e836317b09ba5f8ef662dba748ec19f9b7f6e211e4a5215d35bc4409c`. V0 best checkpoint SHA256: `1ff947f795162b747115fee6e3131eb10fd3955445a3040902a274fa71457c65`; V1 best SHA256: `79ee5c25b51f031b5f1d635b2988c331e88695fd6012909a925c9cc53604eb3b`. V1 private predictions SHA256: `ea71220a30c2895eb283fb3c3cf225ed95231cdf07d6172de58363c96cb65c31`.

## Independently recomputed from observed confusion matrices

| Metric | V0 CNN–ViT | V1 ViT-only | Difference (V1–V0) |
|---|---:|---:|---:|
| Correct / 35,005 | 28,963 | **29,137** | +174 |
| Accuracy | 82.7396% | **83.2367%** | +0.4971 pp |
| 10-class macro-F1 | 55.8664% | **57.4409%** | +1.5746 pp |
| Weighted-F1 | 81.2089% | **81.9558%** | +0.7469 pp |
| Balanced accuracy | 54.8974% | **54.9155%** | +0.0182 pp |
| Selected epoch / 50 | 41 | 44 | N/A |

All ten class-specific precision, recall, F1, the balanced/weighted aggregates, exact confusion total, and the selected epoch 44 training-history F1 agree with the submitted saved metrics to numerical precision. The **same class support** in both confusion matrices confirms target-label parity, while the saved reference `data_lineage.json` provides the matching original row/window indices and original prediction hash. Runtime execution was user-run in Kaggle, not independently observed by ChatGPT.

### Per-class F1 (actual observed validation)

| Class | Support | V0 F1 (%) | V1 F1 (%) | V1–V0 (pp) |
|---|---:|---:|---:|---:|
| Analysis | 428 | 28.78 | 21.85 | -6.93 |
| Backdoor | 350 | 9.03 | 14.95 | 5.92 |
| DoS | 2509 | 18.54 | 26.57 | 8.02 |
| Exploits | 6684 | 73.29 | 73.64 | 0.35 |
| Fuzzers | 3679 | 77.21 | 77.22 | 0.00 |
| Generic | 7894 | 98.68 | 98.62 | -0.05 |
| Normal | 11103 | 94.37 | 94.61 | 0.25 |
| Reconnaissance | 2116 | 78.87 | 79.93 | 1.06 |
| Shellcode | 221 | 57.69 | 56.72 | -0.96 |
| Worms | 21 | 22.22 | 30.30 | 8.08 |

**Clinically important for IDS and rare-class focus:** DoS true positives increased 299→492 of 2,509 and F1 improved 18.54→26.57%, but **1,911/2,509 DoS are still misclassified as Exploits** (previously 2,130/2,509). Backdoor F1 improved 9.03→14.95%, but correct Backdoor remains just 29/350. Worms improves 3→5 correct out of **21** and carries high uncertainty. **Analysis F1 deteriorates 28.78→21.85%**, correct 79→58 out of 428; Shellcode F1 is slightly worse. Despite higher macro-F1, balanced accuracy increased only ~0.018 pp, so this is not uniform minority-class recall improvement.

## Durable reproducibility

- Synced observed GitHub commit includes **14 figure recipes × PNG/SVG/PDF = 42 figure files**, seven aggregate plot-data CSVs, full history, confusion matrix, best-checkpoint metrics and class supports.
- The figure manifest links all seven numeric source CSVs and all 42 export SHA256s to editable plotted output.
- The **83.90 MiB full run archive** was uploaded as an unpublished **draft GitHub Release**: [release link](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-7be70698aa7f3dd7a5de), asset ID `653721436`, **87,971,840 bytes**, state `uploaded`, SHA256 `3a51a9f5301c215c7388f15c84379872d7cc02f9ab664349482cc65811a58dd9` matching user-reported Kaggle checksum. Separate 102-byte `.sha256` asset ID `653721603`. GitHub release metadata independently checked; raw tar not independently downloaded and rehashed.
- Best probabilities/weights/checkpoint snapshots remain in archived draft Release, **not ordinary public Git history**.
- Comparison source table: [classwise_deltas.csv](../results/comparisons/V0_CNNViT_vs_V1_ViTOnly/classwise_deltas.csv) is derived solely from both committed 10×10 confusion matrices.

## Evidence boundary and next decision

**Supported claim:** V1 exceeds *our own* V0 internal-validation macro-F1 by +1.5746 pp on identical targets under nominally matching optimization budgets. It has strong DoS/Backdoor improvements but Analysis/Shellcode regressions.

**Not supported yet:** significance across 3–5 seeds, actual official held-out test superiority to Chen et al. (2026), causal contribution of CNN encoder, proof that 5-row CSV adjacency is temporal, or a novel methodological contribution. Prior original-source duplication/label-conflict audit and unsigned time sequence remain important limitations.

**Next CTRL-001 V2:** implement a corrected CNN branch with a measurable nonzero/variable activation-health gate and matched internal validation. Freeze correction before training. Compare against V0 and V1; preserve private predictions and figure sources. Follow with window=1/row-permutation controls, 3–5 seeds and paired uncertainty testing before proposing the rare-attack-aware new method.
