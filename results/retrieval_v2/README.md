# Snapshot of the v2 results

Small copies (json + per-query MRR arrays) of the v2 experiments, taken read-only from the training machine by `report/pull_results.sh`
(`runs/retrieval/v2/results/` there; the large rankings, models and training data stay on that machine). They feed `report/make_tables.py`.

| files | content |
|---|---|
| `<system>.test.json`, `.test.perq.npz`, `.test.mixed.json` | TEST split, per slice and direction (MRR@10, Hit@k, nDCG@10), per-query MRR@10, mixed-index diagnostic |
| `<system>.dev.json`, `<system>.devM.json` | DEV split (selection) and the held-out MIRACL-ar training slice (selection only) |
| `data/manifest_*.json`, `data/exclude_summary.json` | training-data manifests and the summary of the contamination filter |
| `data/audit_ood*.json` | overlap of the audited evaluation sets with the training data, before and after the filter |
| `data/protocol.json` | protocol constants that exist only in the lab notebook, each with its source |
| `guard_split/` | per-query scores of MIRACL-HN and Mr.TyDi re-ranking split into contaminated / clean queries |
| `models/*/lab_retrieval.json`, `models/wm_a75/merge.json` | the training arguments of the final models and the WiSE blend |

System names: `base_*` off-the-shelf, `v1_s13` the v1 recipe (seed 13), `f4_*` students without distillation, `f5_*` students distilled from the fine-tuned bge-m3,
`m3_b_s13` / `m3_b_s14` the raw fine-tuned bge-m3 (v2-m3 before blending), `wm_a75` = v2-m3 (WiSE, alpha 0.75).
Some of these are candidates that may be replaced by later runs; refresh with `report/pull_results.sh` and rebuild the report.
