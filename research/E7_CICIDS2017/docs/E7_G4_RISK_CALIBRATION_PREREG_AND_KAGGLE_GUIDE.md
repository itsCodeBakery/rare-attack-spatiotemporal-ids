# G4 — Risk-Calibrated Decision-Making and Abstention under Source-Day Shift

**Status:** Protocol, code, CI and Kaggle Notebook 31 committed; **actual G4 execution not yet performed**.
**Parent:** G3 `E7-G3-OPENWORLD-DAYHOLDOUT-001`. Read-only reuse of all 20 already-trained private LightGBM checkpoints.
**Frozen protocol:** [G4_RISK_CALIBRATION_PROTOCOL_001.json](../configs/G4_RISK_CALIBRATION_PROTOCOL_001.json).
**Notebook:** [31_e7_g4_validation_risk_and_abstention.ipynb](../notebooks/31_e7_g4_validation_risk_and_abstention.ipynb).

## 1. Motivation supported by G3 source reports

G3 completed 20 model runs under four whole-day train/validation/test folds. The original validation-derived 5-FN:1-FP cost threshold hit extreme operating points: on Tuesday and Thursday held-out days, B2 flagged every record (test FPR 100%); on Wednesday and Friday it flagged none (test attack recall 0%). These outcomes are already public in the frozen G3 20-run summary, as are baseline B1 scores and test-family recalls.

G4 evaluates whether *precommitted risk-constrained decision policies* can avoid these degeneracies on the **existing G3 model score distributions**. These G3 test outcomes were **already examined before G4 was designed**. Therefore no G4 comparison on the same held-out days can be described as independent, confirmatory, or proof of improvement. G4 is exploratory methodology development; publish all outcomes, including failures.

## 2. Fully frozen policies

All calibrators use only original G3 **validation-day score/label pairs**. The original classifier, features, frozen capture-day partition, seed, model SHA-256 and source file fingerprints remain identical.

**M0: Original baseline.** G3 LightGBM `B1`, with fixed binary decision threshold `0.5`. This is not retrained. The G4 restored model's binary Macro-F1 at 0.5 must reproduce the corresponding G3 aggregate within tight numerical tolerance before publishing a G4 report.

**M1: Conservative alarm policy.** Candidate alarm thresholds are `{0,0.005,...,1}` and `1.000001` as no-alert sentinel. Among candidates with

\[
\widehat{\mathrm{FPR}}_{\mathrm{val}}(t)+\sqrt{\frac{\log(2/0.05)}{2 n_{\mathrm{val,benign}}}}\le0.01,
\]

choose the candidate maximizing validation attack recall, ties broken toward the largest threshold. If no candidate passes the bound, use the never-alert sentinel and report failure of the requested bound.

**M2: Abstention band.** Reuse M1 high threshold `t_high`. Define a lower auto-benign threshold `t_low < t_high` as the largest candidate from `{-0.000001,0,...,1}` satisfying

\[
\widehat{\mathrm{FNR}}_{\mathrm{auto-benign,val}}(t_{low})+
\sqrt{\frac{\log(2/0.05)}{2 n_{\mathrm{val,attack}}}}\le0.05.
\]

If infeasible, choose the never-auto-benign sentinel. Predict **benign** when `p<=t_low`, **attack** when `p>=t_high`, and **defer** otherwise. Counts of deferred traffic are always reported; never silently remove abstained instances. Measure test-day coverage, selective error on decided traffic, test benign FPR (denominator all benign traffic), missed-attack fraction, attack alert recall, classwise support and defer fraction. Disclose threshold infeasibility and coverage under 50%.

The simultaneous empirical CDF bound is derived from the Dvoretzky–Kiefer–Wolfowitz inequality for iid calibration scores. **Temporal dependence and distribution shift prevent any unconditional across-day coverage or risk guarantee.** These are diagnostics on the calibration day only.

## 3. No-leakage/irreproducibility controls

- Original source SHA256 checks for all 8 CSVs and exactly 2,830,743 derived nonempty rows; raw original CSVs are unmodified.
- The original G3 source-trained feature matrix is restored exactly: no IP address, port, flow ID, raw timestamps, day indicator or nearby CSV row context as classifier features.
- The 20 G3 public metrics files and private unpublished GitHub model ZIP assets must exist. The draft release is **read-only** for G4. Each ZIP is privately downloaded, checked against published model SHA-256, unpacked with exact frozen metadata, and checked for validation calibration identity.
- G4 threshold selection is completed **before** test probabilities are computed for that restored model.
- G4 does not create a new fitted model, retrain an existing one, or use frozen UNSW-NB15 official test data.
- GitHub outputs are *only* aggregated G4 scores and source hashes, written in a new `results/G4_RISK_ABSTENTION/` directory with create-only verification. Original G0/G1/G2/M1/G3 evidence stays unchanged. Recovery from a restarted Kaggle notebook skips any verified previous G4 report.

## 4. Reporting and follow-on research gate

Analyze calibration feasibility per day and cross-day failures; avoid selecting a preferred policy based on the reused G3 test-day results. Emphasize operating-point failure, test FPR, test miss rate, abstention coverage and class support, not only Macro-F1. Low-support Heartbleed (11), Infiltration (36), and SQL Injection (21) prohibit strong subgroup performance claims.

**Confirmatory future study required:** independently sourced/corrected CIC-IDS2017 or a new external IDS corpus; split/label audit; locked data source and configuration; independent model/method comparison; leakage and domain-shift controls. Do not compare G4 open-world day-holdout numerically to the SCV-IDS base paper's random train/test accuracy.

## 5. Execution

Import **Notebook 31** into Kaggle, attach the same `dsshayan/dscicids` dataset and enable Internet and `pushRare` secret. The notebook validates 20 private models before loading original source data, runs inference/calibration only, and saves 20 safe G4 reports + one summary individually to GitHub. The code passed 73 synthetic tests plus pinned notebook-cell compilation in [GitHub Actions](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/actions/runs/39721921712); this does **not** mean the original data have been evaluated under G4 yet.
