# M1 seed 42 — recovered outcome, pending independent audit

**Experiment:** `M1-CONTEXT-RISK-001`  
**Evaluation:** new *UNSW-NB15 TRAIN-derived* group-disjoint outer fold, **not** the original official UNSW test.  
**Status:** `COMPLETED_PENDING_INDEPENDENT_RESULT_AUDIT` based on the researcher's completed Kaggle recovery output (2026-10-19).

## Immutable private result evidence

The researcher's Kaggle recovery verified the `sealed_outer_predictions_PRIVATE.npz` hash against `final_public_summary.json` and confirmed all six controls with `official_test_accessed: false`. The complete original checkpoint/prediction archive was uploaded to an unpublished GitHub draft release:

- Release: [M1 private original seed-42 archive](https://github.com/itsCodeBakery/rare-attack-spatiotemporal-ids/releases/tag/untagged-c333b2cb4a9e1c45a796)
- Archive: `M1-CONTEXT-RISK-001_seed42_PRIVATE.tar.gz` (50,500,644 bytes)
- Archive SHA-256 (confirmed against the remote GitHub asset): `00c440f79a3f88e71dc831281fd10776a06af4e57569f62d0b25a8c27910ab88`
- The remote draft release also has the SHA-256 sidecar. This is private; **never publish** the release or place raw model/probability files in Git.

## Results from the user's Kaggle console (rounded display only)

| Control | Accuracy (%) | 10-class macro-F1 (%) |
|---|---:|---:|
| C0_TARGET | 81.0094 | 64.1491 |
| C1_CONTEXT | 84.4766 | 65.0154 |
| C2_BLEND | 84.1529 | 65.6184 |
| C3_GATE | 84.2801 | 66.0669 |
| C4_RISK | 81.0094 | 64.1491 |
| C5_NOCONTEXT | 81.0094 | 64.1491 |

**These values are transcribed from the saved Kaggle console report and are not yet independently reconstructed from the private probabilities or public original confusion matrices.** Preserve all precision in the forthcoming `final_public_summary.json`; this note is not a substitute.

### Preregistered scientific interpretation

- The unrestricted gate C3 had the highest observed held-out macro-F1, 66.0669%, and modestly higher F1 than the fixed blend. This is **exploratory** because the preregistered primary proposed candidate was C4, not C3.
- **C4 did not improve macro-F1** relative to target-only and fell short of context-only. Its exact equality with C0 is consistent with a zero-cap fallback, but the real `selected_alpha` and `gate_calibration_status` must be read from the saved calibration files before confirming the mechanism.
- C5 matched C0 as well; similarly verify its calibration status and cap.
- **The primary M1 superiority criterion failed in this seed.** Do not substitute C3 for C4 as the primary endpoint after observing the held-out results, and do not relax the originally registered constraints retrospectively.
- Five registered seeds remain a robustness plan, not five independent test sets when the underlying outer holdout is shared. Do not use the registered seeds to shop for a better method on the same holdout.
- The original single official test evaluation from the preceding project phase remains unchanged and is not a test of M1. Source-neighborhood CSV ordering remains unverified chronology.

## Publication and audit still required

1. Commit the **exact original seven implementation files** used in Kaggle to the research GitHub repository. Do not replace them with subsequently patched files when attributing seed-42 provenance. The original trained runner SHA-256 is `a1c730afbd43028db3e1125f87f27480a018dad5b520b54184620dd1e7b6b67c`.
2. Commit original `final_public_summary.json`, `calibration_public.json`, `outer_split_manifest.json`, six genuine `C[0-5]_..._confusion.csv`, six genuine `C[0-5]_..._per_class.csv` and `progress_public.json`. Do not synthesize full-precision results from rounded console output.
3. Verify each public artifact matches the identical relative file and SHA-256 recorded inside the immutable private archive's manifest. Recompute metrics from authentic saved private predictions when they can be accessed by the independent audit process.
4. Record actual `selected_alpha`, calibration feasibility and minority-class recalls/F1, including supports. Do not call an empirically enforced calibration constraint a population guarantee.
5. Synchronize future full experiment checkpoints during compute rather than only at final completion, in the established E6 style; the separate M1 autosync repair is currently not the original trained code.

**Research disposition:** preserve C4 negative result; investigate its predeclared calibration constraints in a *new* protocol before proposing method changes.
