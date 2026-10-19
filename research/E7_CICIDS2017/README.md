# E7 — CIC-IDS2017 external validation & methodological novelty

**Project:** Rare-Attack Spatiotemporal IDS · isolated E7 workstream  
**Status:** External CIC-IDS2017 archive attached in Kaggle; read-only G0 file/schema/timestamp validation **pending**; **no E7 models trained or scores reported**.  
**Research rule:** This folder is the canonical home for **all NEW public E7 research artifacts**. Nothing from E3/E4/E5/E6/M1 is relocated, overwritten, or re-evaluated. Original UNSW-NB15 official test may **not** be reused for model selection.

## Frozen earlier results are preserved separately

- [Original frozen official UNSW result](../../results/WINDOW-OFFICIAL-001-S42/official_metrics.json): 86.321786% accuracy and 58.139685% 10-class macro-F1. This test was already used once. **No more test evaluation.**
- [E6 contextual-order audit](../../docs/E6_ORDER_ROBUSTNESS_AUDIT.md): row neighborhoods do **not** verify chronology.
- [M1 seed-42 authentic six-control report](../../results/M1-CONTEXT-RISK-001/seed_42/final_public_summary.json) and [independent audit](../../results/M1-CONTEXT-RISK-001/seed_42/INDEPENDENT_AUDIT.md): global risk cap infeasible; C4 alpha=0, not a successful gain.
- The verified original 50.5 MB M1 seed42 artifact remains in an **unpublished private GitHub draft release**; original archive SHA-256 `00c440f79a3f88e71dc831281fd10776a06af4e57569f62d0b25a8c27910ab88`. Do not make draft releases public.

## Folder layout (E7 only)

| Folder/file | Role |
|---|---|
| `docs/` | Preregistered protocol, novelty differentiation, quickstart and resume handoff |
| `scripts/` | Reproducible E7-only Kaggle audit / future model code |
| `tests/` | Synthetic no-leakage and source-integrity tests |
| `notebooks/` | Kaggle notebook to audit actual attached source before training |
| `data_provenance/` | Publisher URLs, original file manifests, checksums, safe aggregate dataset audits (never raw records) |
| `results/` | Public run summaries, confusion matrices, metric tables, plots, and signed artifact manifests (no private predictions) |
| `EXPERIMENT_LOG.md` | Append-only decisions, run IDs, GitHub commits and evidence checkpoints |
| `ASSET_POLICY.md` | What belongs in Git vs private draft releases |

## Current user-confirmed external files in Kaggle

The source lists **eight** day/attack-segment CSVs from the CIC-IDS2017 `TrafficLabelling` archive, including Wednesday, Monday, Tuesday, Thursday, and Friday. The notebook's directory listing displays a **space immediately before the filename slash**:

`/kaggle/input/datasets/dsshayan/dscicids/TrafficLabelling /Wednesday-workingHours.pcap_ISCX.csv`

The folder may literally end in a trailing space. **Never guess or strip directory names.** Discover the real CSV parent using `Path('/kaggle/input/datasets/dsshayan/dscicids').rglob('*.csv')` and derive it from `files[0].parent`. Kaggle listing is not a complete G0 provenance audit. The original user attachment is called a ZIP and Kaggle automatically extracted it.

## Current step (read-only; no fitting)

1. Open [Notebook 25](notebooks/25_e7_external_g0_provenance_audit.ipynb) in Kaggle and list exact E7 CSV paths and just three raw timestamp **strings** (not entire flow rows or IP addresses).
2. Verify actual source metadata and timestamp format. Do not infer source-order causality or flow-end semantics from parsing alone.
3. Run synthetic audit tests, then G0 source-only smoke and full versioned G0 audit. Only after G0 is reviewed, freeze the external chronology-and-label protocol.
4. All expensive future training must automatically store each completed model and private probabilities in **checksum-verified unpublished GitHub draft releases**, with public aggregate progress synced after every stage. A Kaggle local-only success does not satisfy backup completion.

## Publication rule

We may begin manuscript drafting from existing frozen results. E7 novelty and external generalization remain **unproven until independently audited real experiments**. Do not conflate within-CIC training/testing with strict zero-shot UNSW→CIC transfer, or compare incompatible labels/splits as directly matched.
