# Independent novelty audit — context, rare attacks and risk control (2026-10-19)

**Status:** targeted literature search and publisher-page verification completed; **not** an exhaustive systematic review, patent search or proof of global novelty. Peer-reviewed **original research** was preferred (2024–2026); 2021 corrected-data paper and 2023 DUEN serve as historical controls. The parent SCV-IDS article (IEEE IoT Journal 2026) and all our original E3/E4/E5/E6/M1 findings are preserved. Sources should be checked at final citation submission for exact author lists and updated issue metadata.

## Executive determination

**STOP** describing learned mixtures, entropy/margin reliability gates, class weighting, class-sensitive hierarchy, uncertainty quantification, cross-fitting, calibration or simple conformal prediction as uniquely novel. All have close conceptual prior art. **NO claim of a new fully validated risk-controlled architecture is supported by M1 seed 42:** the preregistered C4 alpha grid was infeasible and its reported held-out posterior equaled the C0 target-only model.

**Potential, narrow hypothesis for an M2 paper:** context can be treated as a *provenance-verified, selectively usable information source*; route or abstain conditioned on **actual prior completed related flows**, avoid pseudo-order attacks on sparse classes, and measure the **feasible Pareto frontier** of attack recalls, benign FPR, macro-F1 and overall accuracy under strictly separated calibration and chronological test episodes. Novelty is currently a *hypothesis*, not a finding. A genuine contribution would need a technically distinct solution, correct baselines and out-of-domain evidence.

## Direct prior-art differentiation matrix — journal original experimental studies

| Year | Publisher-verified original study / DOI | Prior capabilities already established | What remains unproven for us |
|---|---|---|---|
| 2026 | **SCV-IDS**, IEEE Internet of Things Journal (2026), author-uploaded base paper | 5-record serialized traffic images, CNN–ViT dual cross-attention, multi-class UNSW, **CIC-IDS2017 Wednesday subset** (within-dataset 80:20), rare-class limitation discussed | Replacing CNN–ViT with LightGBM is not architecture novelty; genuine causal-order validity and rare-signal preservation are not established by its reported metrics |
| 2026 | **HICID**, *Computer Networks*, [10.1016/j.comnet.2026.112532](https://doi.org/10.1016/j.comnet.2026.112532) | Two-stage minority-group hierarchy, boundary-contraction loss and adaptive kNN oversampling on UNSW-NB15 & CICIDS2017 | Must compare against a strong hierarchy, not call minority-aware routing intrinsically novel |
| 2026 | **EfficientNet MoE**, *Scientific Reports*, [10.1038/s41598-026-62727-5](https://doi.org/10.1038/s41598-026-62727-5) | Deep network intrusion detection with optimized mixture-of-experts routing | Entropy/confidence gating, MoE blend alone cannot be a novelty claim |
| 2026 | **XGB-AGMoE**, *Sensors*, [10.3390/s26196096](https://doi.org/10.3390/s26196096) | Validation-adaptive heterogeneous expert blending for multiclass intrusion detection | Class-aware empirical guardrail calibration is different, but M1 failed and requires independent positive demonstration |
| 2026 | **MHFED**, *Scientific Reports*, [10.1038/s41598-026-65467-8](https://doi.org/10.1038/s41598-026-65467-8) | Multi-head feature experts, routing under IoMT traffic heterogeneity | Need controlled context-validity/causality evidence, not simply multi-head expert fusion |
| 2025 | **Hybrid Bayesian SDN intrusion detection**, *Computer Networks*, [10.1016/j.comnet.2025.111436](https://doi.org/10.1016/j.comnet.2025.111436) | Explicit Bayesian uncertainty quantification and practical false-positive/false-negative analysis | Uncertainty itself is not novel; operational risk tradeoff needs measured calibration and accuracy |
| 2025 | **CAEAID**, *Computer Networks*, [10.1016/j.comnet.2025.111161](https://doi.org/10.1016/j.comnet.2025.111161) | Incremental contrastive representation, adaptive anomaly detection and concept-drift updating | Do not claim continual learning without real sequential update/forgetting experiments |
| 2025 | **HybridGuard**, *Computer Networks*, [10.1016/j.comnet.2025.111966](https://doi.org/10.1016/j.comnet.2025.111966) | Minority-class two-phase learning, feature selection and WCGAN augmentation evaluated across three intrusion data sets | Must match minority detection controls on aligned splits/labels; not enough to outperform SCV-IDS |
| 2025 | **Multiscale intrusion detection with EQL v2**, *Computers & Security*, [10.1016/j.cose.2024.104173](https://doi.org/10.1016/j.cose.2024.104173) | Class-imbalance-aware multiscale representation/fusion | Multiscale context and imbalance-aware optimization both preexist |
| 2025 | **HIDIM**, *Computers & Security*, [10.1016/j.cose.2024.104155](https://doi.org/10.1016/j.cose.2024.104155) | Hierarchical protocol-attribute modeling and minority boundary-sensitive augmentation | Protocol hierarchy and rare-class boundary models are preexisting |
| 2025 | **MICEM reliable IDS data evaluation**, *PLOS ONE*, [10.1371/journal.pone.0334157](https://doi.org/10.1371/journal.pone.0334157) | Explicit data-quality reliability checking for CICIDS2017 | Data auditing is required good practice, not a free-standing novel invention |
| 2025 | **Roquain selective conformal prediction**, *JRSS B*, [10.1093/jrsssb/qkae120](https://doi.org/10.1093/jrsssb/qkae120) | Selection-aware calibration and false-coverage-rate control; **statistics general**, not IDS-specific | Never promise risk/coverage guarantees from M1's five-grid empirical feasibility check |
| 2023 | **DUEN**, *Expert Systems with Applications*, [10.1016/j.eswa.2023.120420](https://doi.org/10.1016/j.eswa.2023.120420) | Dynamic ensemble/classification hardness with rare-class improvement | General minority-risk-aware ensembles existed before our method |
| 2021 | **Engelen et al.**, IEEE S&P Workshops, [fixed CICFlowMeter + corrected labels](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html) | Demonstrated flawed CICFlowMeter flow construction and mislabeled flows, with corrected versions | Cannot use original CICIDS2017 as unquestioned ground truth; sensitivity to corrected versions is essential |

**No blanket “highest impact factor/top 1%” assertion:** journal rank/impact depends on indexing year/category and requires a fresh Clarivate JCR check before targeting. We select on methodological fit and study rigor first.

## Specific research gap that survives the initial prior-art audit

Existing works individually address: (a) class imbalance/hierarchies; (b) mixture routing; (c) model uncertainty; (d) streams/concept drift; and (e) leakage-aware data quality. Our initial targeted search **did not establish** a published, directly equivalent *joint method* in which (1) context is admitted only after verified flow origin/time/session linkage and causality, (2) context utility is estimated out-of-fold without source/target leakage, (3) minority-class detection and benign false alarms are assessed as **simultaneous feasible/infeasible** operational objectives, (4) both original and corrected labels are tested, and (5) evaluation separates chronological same-dataset and external-domain transfer.

This **may be a meaningful research question**, but cannot be claimed globally unprecedented until comprehensive related work/patent verification and a proven distinct algorithm. In particular:
- Avoid “new attention”, “new gating”, “first MoE”, “new class weighting”, “conformal guarantee”, “zero-day”, “continual”, and “causal” unless independently demonstrated.
- Existing M1 global cap **failed**: it does not justify saying constraints were satisfied or that minority recall was preserved.
- C3 (ungoverned gate) is an **exploratory** result, not the preregistered winning primary method.
- E5 and E6 showed source order is not verified time; label clustering creates a major shortcut hypothesis.
- The full original official UNSW test has **already been used once**; do not optimize a new method against it.

## Five falsifiable research hypotheses — not publication claims

**H1: source provenance.** Timestamp-grounded, prior-**completed**, endpoint-related history improves 10-class or binary macro-F1 versus record-only on a **new external data source**. *Null:* no paired improvement or leakage checks fail. Negative results count.

**H2: context robustness.** Replacing authentic related history with a matched distributionally fair unrelated- or permuted-history control reduces performance. Specify matched history sizes and retrain the control on the same split to separate model distribution shift from causal ordering.

**H3: minority protection.** A **newly registered** per-sample or class-sensitive fallback policy yields a measurable feasible Pareto improvement in minority recalls and benign FPR at a documented accuracy cost. Compare fixed blends, target-only, contextual expert, naive confidence routing, hierarchy, and M1 gate. If no feasible policy exists, report impossibility, don't relax after seeing test.

**H4: correction robustness.** Relative conclusions survive dataset-ground-truth/corrected-flow sensitivity experiments without merging incompatible original and corrected rows.

**H5: external-domain transfer.** With a single harmonized feature space and a frozen source-trained detector, report truly zero-shot source->target binary accuracy, attack AUPRC and false alarms with **no target-domain fitting**. Adapted-from-scratch training is **independent dataset replication**, NOT transfer.

## Required study design before any novelty claim

1. Run E7 G0 source/timestamp/flow-column/label/duplicate inspection before choosing windows or split. Store exact file hashes, parse formats and raw-count manifests. No raw IP/flow records in Git.
2. Confirm the base-paper's already-done **within-CIC Wednesday** evaluation is different from our new **chronological / scenario-heldout** protocol. Do not claim CICIDS2017 itself was novel external coverage.
3. For every permitted true-time experiment, compare target-only, verified causal four-flow context, context-permutation, fixed mixture, unrestricted gate, frozen M1-style risk calibration, and a separately **preregistered** new method if needed. Fixed dates/seed choices, same target rows, different train-only fits.
4. Whole-day holdout can contain unseen attack families and is thus **open-world binary detection**, not an ordinary closed-set multiclass task; predeclare label intersection and evaluation semantics.
5. Report rare-class supports; absent class => NA/unsupported (not a fabricated 0-F1 as if a class is a genuine expected deployment class); report full confusion tables and both seen/unseen class macro statistics when appropriate.
6. Group/host leakage, aggregate timestamps, time feature shortcuts, attack schedule and label aliases are tested; future events may never be used in features.
7. Track five seeds and group-aware 95% uncertainty, paired comparisons on identical target instances, training/inference CPU time, RAM and model bytes. Keep model/prediction files only in checksum-verified private draft GitHub releases, aggregate files in repository.

## Publication-go / no-go rule

**GO to strong original-method journal submission only if:** (i) a technically distinct prespecified algorithm solves a real problem rather than renaming prior MoE; (ii) new external primary endpoint supports it over strong controls; (iii) leakage/provenance/chronological alternatives checked; (iv) rare-class tradeoffs and calibration violations reported transparently; and (v) all raw evidence + source/hash manifests are immutable and recoverable. **Otherwise:** paper may become a rigorous **benchmarking/reliability** study rather than an overclaimed new network architecture. Journal choice remains deferred.
