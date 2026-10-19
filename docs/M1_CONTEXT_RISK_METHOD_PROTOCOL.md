# M1 — Context-Reliability and Rare-Attack Risk-Controlled Intrusion Detection

**Protocol:** `M1-CONTEXT-RISK-001` | **Date:** 2026-10-19 | **Status:** design registered, **NOT trained**.

This is the next proposed-method experiment in the **existing** rare-attack/spatiotemporal intrusion-detection paper, not a new project. It follows [E5 source-order diagnosis](../results/E5-CONTEXT-DIAGNOSTIC-001/summary.json), [E6 order-robustness audit](E6_ORDER_ROBUSTNESS_AUDIT.md) and the one-time [frozen original official UNSW evaluation](../results/WINDOW-OFFICIAL-001-S42/official_metrics.json).

The separately drafted [E7 timestamp-verification protocol](E7_TEMPORAL_VALIDITY_PROTOCOL.md) remains an **external temporal-validity study**; it is not a prerequisite for the train-only M1 method proof-of-concept. No second UNSW official-test access is authorized.

## 1. Exact problem exposed by E6

On the **same 35,151 internal validation targets**, the 42-feature `target_full` model achieved accuracy **0.814315** and ten-class macro-F1 **0.658146**, while 444-feature `intact_full` achieved accuracy **0.848141** and macro-F1 **0.645505**. The context model's DoS recall **0.549979** was lower than target-only **0.752365**; Backdoor recall also declined, while Analysis recall improved. These are **development** results, not new official-test results. E6 order-shuffling changes model behavior but does **not** establish genuine chronology: the source CSV order may encode arbitrary class clustering.

**Research question:** Can a **context-utility gate** use context only on samples where it adds discriminative value, while constraining degradation for high-risk rare attacks and benign false alarms, rather than choosing target-only or context-only for all samples?

**Terminology:** call original UNSW features *record-neighborhood context*, **not verified temporal context**. A later timestamp-authenticated independent experiment is required before temporal/causal claims.

## 2. Formal candidate (testable hypothesis, not established novelty)

Let `K=10`, `x_i ∈ R^42` be current record features, `h_i=(x_{i-4},...,x_{i-1})` the preceding four split-local source records, and `z_i=phi(h_i,x_i) ∈ R^444` the frozen E3 feature construction, using only training-fitted transformations. Neither history labels nor any attack-derived class-purity flag are allowed inputs.

Two matched-budget multiclass LightGBM learners produce
```text
p_T,i = P_T(y | x_i)                      # 42-feature target expert
p_C,i = P_C(y | z_i)                      # 444-feature contextual expert
```
Both are K-dimensional probability vectors in the simplex; do **not** compare independently tuned expert ensembles to a single fixed control.

For each sample, define a **label-free**, train-fitted gate feature vector `u_i`, containing:
- target and context predictive entropy, winning-class confidence and top-2 margins;
- `L1(p_T,i-p_C,i)`, whether top-1 predictions disagree, and model predictive margin difference;
- train-scaled mean/maximum numerical distance of each historical record from the target, and categorical-history mismatch counts;
- optional missingness/context-availability indicators only if they exist in both comparison conditions.

No `attack_cat`, `label`, ground-truth minority membership, source row number, IP, time, data-shard/day or validation-correctness flag is a gate feature. Source-index gaps may be used for **audit only**, not prediction. Scaling/vocabulary is fitted inside the permitted training folds.

**Learned gate:** `g_theta(u)=sigmoid(theta^T u + b)`. Use a single L2-regularized logistic gate initially (a deeper gate is a separately labeled future ablation). Final posterior for a fixed context cap `alpha`:
```text
g_i       = alpha * g_theta(u_i), 0 <= alpha <= 1
q_i,c     = (1-g_i)*p_T,i,c + g_i*p_C,i,c
yhat_i    = argmax_c q_i,c
```
`alpha=0` is an explicit target-only fallback. Probability mixing, logistic gating and mixture-of-experts are **known ideas**; neither is itself a novel contribution.

Train `theta` on genuine held-out **out-of-fold expert predictions** with a class-balanced differentiable loss:
```text
L_gate(theta) = (1/N) sum_i w_{y_i}[-log(max(q_i,y_i,eps))] + lambda*||theta||_2^2
```
Class weight `w_c = min(12,(N/(K*N_c))^0.5)`, normalised to sample-weight mean one; `lambda` must be fixed before fitting the gate or tuned *inside* gate-fit only, never on outer evaluation. All expert probabilities used to fit the gate must be from models that did not train on their own target source group.

### Risk control is a separate calibration step

On an inner **calibration-only** subset (not used to fit the gate), choose `alpha` from the fixed grid `{0, 0.25, 0.50, 0.75, 1.00}`. Evaluate hard multiclass predictions and maximize calibration macro-F1 **subject to all**:
```text
accuracy(q_alpha) >= accuracy(p_C) - 0.010
FPR_benign(q_alpha) <= FPR_benign(p_T) + 0.005
recall_c(q_alpha) >= recall_c(p_T) - 0.020, for c in
  {Analysis, Backdoor, DoS}
```
Rates above are in fractional units (0.010 = one percentage point). `FPR_benign` is the proportion of truly Normal records predicted as any attack. These are **predeclared design tolerances, not measured guarantees**. Ties: choose smaller `alpha`. If no grid point is feasible, record `INFEASIBLE` **without modifying tolerances** and report the target-only fallback `alpha=0` as a negative outcome. Rare-class sample size and uncertainty must be shown alongside feasibility; constraints on calibration do **not** imply population guarantees.

This explicitly tests whether *soft context selection plus observed risk guardrails* can improve on indiscriminate context mixing. Accuracy, macro-F1 and minority recall cannot be jointly guaranteed in advance.

## 3. Leakage-aware design: frozen before M1 outcome observation

Dataset: **original 175,341-record UNSW-NB15 TRAIN CSV only**, SHA-256:
`bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.
Load frozen 42 raw feature columns in [E3 manifest](../results/WINDOW-E3-001-S42/feature_manifest.json). Keep original 10-class order and exact raw train/categorical schema. `UNSW_NB15_testing-set.csv` and all `WINDOW-OFFICIAL-001-S42` predictions are forbidden as M1 training/tuning/evaluation inputs. Earlier one-time official results remain unchanged and are **not** an independent evaluation of M1.

**Outer sealed M1 evaluation:** use a **new** frozen `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2031)` with groups the exact canonical 42-feature-row tuple hashes. Use predeclared fold index `0` as M1 held-out evaluation, leaving four folds for M1 development. If an essential class is absent, **stop** and document infeasibility; do not select a more flattering fold. Fit all preprocessing on each allowed training partition only. Sort each split's rows back into original source CSV order only to preserve comparability with the earlier pseudo-context representation; that order is **not certified chronological**. No history may cross group/split boundaries.

**Expert out-of-fold fitting:** inside four outer-development folds, use a deterministic **4-fold stratified, group-disjoint** split with seed `2032`. Each inner fold: build 5-record windows **separately** inside its training and held-out raw partitions, fit both experts on training windows, and predict both on held-out windows. Gather OOF pairs `(p_T,p_C)`, target labels and gate features only from their held-out groups. Historical rows from other folds must never be borrowed. Use `K=10`, fixed E6 LightGBM parameters, `58` rounds, class power `0.5` and weight cap `12`; no early stopping. Log the excluded first four rows of every split, all ten-class supports and duplicate-group checks.

Split OOF **source groups** into gate fitting `75%` and cap-calibration `25%` using a newly frozen group-disjoint stratified selection seed `2033`; gate calibration labels must never enter `theta` fitting. Do not use a source group in both gate-fit and calibration. Calibration predictions are OOF with respect to their corresponding experts, although the experts underlying other OOF samples may have trained on those groups; this limitation is reported transparently.

After gate and cap are locked, refit the two experts on **all four outer-development folds** (same 58-round budget and class weights), generate identical-target held-out outer predictions, and evaluate exactly **once per predeclared seed** on the sealed outer fold. Do not choose/modify the method based on outer results. A distinct future external dataset/chronological protocol is needed for independent evidence.

Run primary seed `42` and, if the protocol passes all gates, repeat `{43,44,45,46}`. Preserve seeds/hashes/configs before execution. If CatBoost is later compared, state that it is an additional learner baseline and maintain matched splits.

## 4. Essential control experiments (all same outer target IDs)

| Variant | Description | Required interpretation |
|---|---|---|
| `C0_TARGET` | `p_T` alone | Protects best E6 development macro-F1 concept |
| `C1_CONTEXT` | `p_C` alone | Protects E6 development accuracy concept |
| `C2_BLEND` | `0.5p_T+0.5p_C` | Tests whether just blending suffices |
| `C3_GATE` | Learned `g_theta`, fixed `alpha=1`, **no** calibration constraint | Tests whether reliability gate by itself helps |
| `C4_RISK` | Same `g_theta` with preregistered cap guardrails | Proposed risk-limited mixture |
| `C5_NOCONTEXT` | Same gate inputs except remove explicit context-distance/mismatch attributes | Distinguishes learned context reliability from raw confidence mixing |

All candidates use the same expert fits and exact held-out labels. `C5` requires fitting its own gate on the **same OOF partitions**, with identical optimization and cap selection restrictions. Also run fixed-feature ablation: swap the **same four** previous records within each window in a fixed seed permutation, recompute all 444 features and evaluate distribution-shift sensitivity; do not call this a chronology proof. A separate matched *retrained* permutation control is needed for causal attribution of order.

## 5. Primary endpoint, claims and stopping rules

**Primary metric:** held-out **10-class macro-F1** for `C4_RISK` versus **both** `C0_TARGET` and `C1_CONTEXT` on identical outer labels. Secondary: overall accuracy, weighted F1, per-class precision/recall/F1 (with support), rare-class recalls/F1 (Analysis, Backdoor, DoS and Worms), `Normal` false-alarm rate, Brier score/ECE and training/inference CPU time. Report confusion matrices, paired prediction-disagreement tables, coverage distributions of `g`, feature-group ablations, drift subgroup results without label-derived input leakage. Include all negative controls.

**Go condition for a promising candidate (not publication proof):** C4 improves held-out macro-F1 over both C0 and C1 without breaking **its preregistered guardrail targets**, and rare class gains persist over repeated seeds; report paired uncertainty (group-aware bootstrap where valid) and rare-class support. If the candidate fails, retain all results, do not cherry-pick among C2/C3/C4, and redesign in a *new* registered experiment.

**No unsupported claims:** Even a positive M1 result on this pseudo-ordered dataset establishes predictive utility **under source-row context**, not genuine network time, universal robustness, unknown-attack detection, or confirmed architecture novelty. Do not retest new models on the already-used UNSW official test. The separate [E7 external timestamp study](E7_TEMPORAL_VALIDITY_PROTOCOL.md) remains the temporal-validity requirement.

## 6. Literature differentiation checkpoint — required before manuscript novelty claims

A 2026 *Scientific Reports* intrusion detection work already uses an **EfficientNet hybrid MoE** (Li et al., [DOI 10.1038/s41598-026-62727-5](https://doi.org/10.1038/s41598-026-62727-5)). Another 2026 *Sensors* study uses **out-of-fold calibration, LightGBM/CatBoost/XGBoost experts, a logistic meta-learner, and validation-adaptive blending** ([XGB-AGMoE](https://www.mdpi.com/1424-8220/26/19/6096)). A 2026 *Scientific Reports* work explicitly evaluates routing and learned gates ([MHFED](https://www.nature.com/articles/s41598-026-65467-8)). Thus **MoE, logistic stacking, confidence routing, cross-fitting and class-aware training cannot be claimed new in isolation**.

The possible (currently **unverified**) contribution is the **explicit empirical guardrail formulation across accuracy, rare-attack recalls and benign false alarms**, motivated by observed harm from unverified CSV context and tested with strict split/order controls. It requires a dedicated 2024–2026 literature review against *risk-limited/selective classification*, *context-aware intrusion detection* and *mixture-of-experts* before asserting originality.

## 7. Execution sequence

1. **This commit:** freeze M1 formulation, split protocol, controls, limits and decision rule. No training.
2. Next: create `configs/m1_context_risk_001.json`, synthetic unit tests of gate probability simplex, strict OOF group separation, `alpha` feasibility and no-test guard.
3. Implement one runnable Kaggle CPU notebook with structured log, sha256 evidence, crash-safe checkpoint/backups and private prediction archival; synthetic smoke test first.
4. Run actual training-only M1, independently audit its full confusion matrices and private model/probability checksums; publish figures only from measured numerical sources.
5. Advance to separately registered temporal-validation/external generalization research after M1 evidence, with no re-use of the original official benchmark for tuning.

**All numbers in Section 1 are previously measured E6 development statistics. Everything from Section 2 onward is a hypothesis/protocol, not an achieved score.**
