# E6 order-robustness: independent public-evidence audit (2026-10-19)

**ID:** `E6-ORDER-ROBUSTNESS-001`. **Status:** completed; public aggregate scores independently reconciled; **causal temporal claim not established**.  
**Source:** [registered design](../configs/e6_order_robustness_001.json), [execution](../scripts/run_e6_order_robustness.py), [archival checks](../scripts/archive_e6_order.py), and [public numerical artifacts](../results/E6-ORDER-ROBUSTNESS-001/).

## What was checked
Recomputed ten-class accuracy, macro-F1 and rare-class recall from all five original public 10x10 confusion CSVs using TP/FP/FN, comparing with each model's stored metrics. **All five accuracies and all five macro-F1 values match within 1e-12**; Analysis, Backdoor and DoS recalls match. No retraining, test access or new results were involved. The archive script *can* replay SHA-256/predictions/nonoverlap checks against privately archived models, but private releases were **not** downloaded/replayed during this review. The reported nonoverlap-subset metrics are therefore committed evidence rather than independently recalculated here.

The code/config restrict input to the hashed original UNSW training CSV, select an alternate grouped train-derived split (seed 2026), fix LightGBM 58 boosting rounds and model seed 42, and prohibit validation early-stopping/test access. No E6 official-test evaluation is reported; the project **already** contains an earlier frozen official evaluation (`WINDOW-OFFICIAL-001-S42`).

## Same 35,151 internal-validation target windows
All numbers here are proportions, not percentages.

| Model | Training windows | Features | Accuracy | 10-class macro-F1 | Analysis recall | Backdoor recall | DoS recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_full | 140182 | 42 | 0.814315 | **0.658146** | 0.225888 | 0.161932 | **0.752365** |
| intact_full | 140182 | 444 | **0.848141** | 0.645505 | **0.258883** | 0.136364 | 0.549979 |
| shuffled_full | 140182 | 444 | 0.809394 | 0.627942 | 0.223350 | 0.113636 | 0.628137 |
| target_nonoverlap | 28037 | 42 | 0.800944 | 0.616967 | 0.246193 | **0.170455** | 0.540518 |
| intact_nonoverlap | 28037 | 444 | 0.838611 | 0.621757 | 0.210660 | 0.088068 | 0.413410 |

Full windowing and every-fifth-window **training** are different fitting sample sizes. The 7,031-row every-fifth-window validation subset is a secondary evaluation on the same target indices for all models, not an external test. Its recorded macro-F1 is respectively 0.678381, 0.662451, 0.630402, 0.637552, and 0.623297 for the table's five models (subset values not independently replayed here).

- `intact_full - target_full`: +3.3825 percentage points accuracy, **−1.2642 pp macro-F1**.
- `intact_full - shuffled_full`: +3.8747 pp accuracy, +1.7563 pp macro-F1.
- `intact_nonoverlap - target_nonoverlap` on 7,031 windows: +3.9113 pp accuracy, −1.4255 pp macro-F1.
- There is **no unambiguous winner**: target-only leads macro-F1 and DoS recall; intact context leads accuracy.

## Threats to validity and decision
1. **CSV row order is not verified event time.** E5 [diagnostics](../results/E5-CONTEXT-DIAGNOSTIC-001/summary.json) show 0.568764 adjacent-same-class rate versus 0.209812 class-marginal expectation, consistent with strong label/order clustering.
2. `shuffled_full` jointly shuffles the four previous-record columns **across examples**, then recomputes statistics. That alters context distributions, not merely within-history event order. Its deficit is **not** proof that chronology caused the gain.
3. Group-disjoint *identical feature tuples* are not disjoint sessions/hosts/campaigns. Nonoverlap reduces shared source rows among selected windows, not traffic-domain dependence.
4. One seed, one alternate train-derived development split; no uncertainty/repeated-seed robustness claim. **Do not compare these internal results directly to SCV-IDS's official UNSW test.** Do not retune against or reopen the previously evaluated UNSW official test.

**Decision: NO-GO for a validated temporal novelty or paper outperformance claim. GO for a distinct E7 provenance-first timestamped-flow study.** See [E7 protocol](E7_TEMPORAL_VALIDITY_PROTOCOL.md). All E6 source artifacts remain unchanged.
