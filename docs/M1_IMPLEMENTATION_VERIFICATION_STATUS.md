# M1 implementation verification and deployment gate (2026-10-19)

**Experiment:** M1-CONTEXT-RISK-001. The preregistered method and config remain in [method protocol](M1_CONTEXT_RISK_METHOD_PROTOCOL.md) and [configuration](../configs/m1_context_risk_001.json).

**Status:** source implementation package has been built and locally verified outside Kaggle, but the six executable source/test/notebook files are **not yet synchronized into this GitHub branch**. No original UNSW training CSV was accessed for M1, no official test was reopened, no experimental model results exist. Do not infer that the existing GitHub notebook contains M1 yet.

**Local verification:** nine synthetic tests passed. An 800-row fully artificial data pipeline completed four OOF folds, trained ten (five target/context pairs) synthetic smoke-test expert models, fitted two gates, calibrated the risk cap, scored six controls on a held-out synthetic fold and created a private checksum-verified archive. These scores are not scientific evidence and must not be put in results/ as real UNSW scores.

**Repository-relative deliverables in the generated ZIP provided in the active ChatGPT conversation:**

- `src/rareids/m1_risk_gate.py`
- `scripts/run_m1_context_risk.py`
- `scripts/archive_m1_private.py`
- `tests/test_m1_synthetic.py`
- `tests/test_m1_security.py`
- `notebooks/24_m1_context_risk_oof_cpu.ipynb`
- `docs/M1_IMPLEMENTATION_RUNBOOK.md`

**Next gate:** sync the six executable files and the runbook from the verified ZIP into this same repository. Only then run Kaggle Notebook 24 with the byte-verified training source, initially seed 42. The notebook must assert all synthetic tests PASS before fitting real models. Save archive checksum, original private predictions, all six control confusion matrices and aggregate outcome report; independently audit and publish only non-sensitive numerical summaries.

**Scientific boundaries:** CSV source record order != verified chronological traffic; M1 remains training-derived experimental development, independent of the earlier sealed official test evaluation. The timestamp-validity E7 study remains a separate, future external-validation gate. No novel-method or paper outperformance claim for M1 has yet been established.
