# M1 implementation package — 2026-10-19

This source bundle implements the **preregistered M1 context-utility and rare-attack risk-control method** in the existing `itsCodeBakery/rare-attack-spatiotemporal-ids` research project. It **does not** include raw datasets, official-test predictions, or observed M1 research results.

## Included repository-relative files

- `src/rareids/m1_risk_gate.py`: pure numerical gate, OOF mixture, metrics, and risk calibration.
- `scripts/run_m1_context_risk.py`: group-disjoint train-only OOF pipeline, six controls, SHA-256 evidence and sealed outer-fold evaluation.
- `scripts/archive_m1_private.py`: archive checksum checks and optional unpublished GitHub draft-release upload.
- `tests/test_m1_synthetic.py`: mathematical synthetic tests.
- `tests/test_m1_security.py`: frozen train-only path, split and missing-class assertions.
- `notebooks/24_m1_context_risk_oof_cpu.ipynb`: user-run Kaggle CPU workflow.

## Verified here

- Nine synthetic tests passed locally, including no-test path, probability-simplex, deterministic gate, calibration infeasible fallback, label/ID invariance, group collision, per-class scoring.
- One 800-row *artificial* benchmark generated solely for software debugging completed all four OOF folds, five pairs of expert fits, gate-fit and cap-calibration, all six outer-fold controls and private archive generation. Its accuracy and F1 values **are not research outcomes**.
- Static compilation passed for all three Python modules; Kaggle notebook JSON validation passed.

## How to sync with existing GitHub repo (Windows)

1. Extract this ZIP to a local directory. Clone the original repository or use a fresh clean local checkout.
2. Copy the six repository-relative files (maintaining their paths) into that existing checkout. **Do not copy any mock data, local smoke artifacts or previously evaluated official test evidence.**
3. Confirm the checked-out branch is `main` and inspect changes: `git status` and `git diff --stat`.
4. Commit only those files: `git add src/rareids/m1_risk_gate.py scripts/run_m1_context_risk.py scripts/archive_m1_private.py tests/test_m1_synthetic.py tests/test_m1_security.py notebooks/24_m1_context_risk_oof_cpu.ipynb`; then `git commit -m "experiment(m1): add OOF gated context and guarded risk controls"`; `git push origin main`.
5. Launch the newly committed Kaggle notebook. Attach **only** the already-approved `dsshayan/unsw-nb15` dataset. Internet should be on to clone the versioned repository. First run **seed 42** only.
6. Archive includes private predictions; if the `pushRare` Kaggle secret is available the notebook attempts to back up in a **draft unpublished** GitHub release. Never publish draft assets, source rows, attack labels/prediction arrays to the public repository.

## Scientific interpretation

Do not claim temporal causality for contiguous UNSW source CSV rows. A different, timestamp-authenticated dataset is necessary for genuine causal-sequence validation; see `docs/E7_TEMPORAL_VALIDITY_PROTOCOL.md` in the existing GitHub repo. Preserve the historical one-time official-test result without reopening it.

## Notes

- The new development split is not a fresh official test. E6 previous development results on a different split must not be compared as a matched run against these M1 numbers.
- The new M1 outer fold and OOF gate/calibration split are locked in `configs/m1_context_risk_001.json` already committed to GitHub.
- The archive upload is optional; local archive and checksum are generated first. If private-upload authentication fails, keep the tarball and GitHub draft-status claims pending until verified.
- Class support must exist in every required training, validation and calibration fold; a failed support check stops the experiment rather than silently relaxing the protocol.
