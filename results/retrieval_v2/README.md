# Snapshot of the v2 results

Small copies (json + per-query MRR arrays) of the final v2 experiments, imported read-only from a checkout of the `v2-embedder` branch by
`report/import_snapshot.sh` (the training machine keeps the large rankings, models and training data). They feed `report/make_tables.py`,
which generates every number in `report/main.tex`. Nothing here is edited by hand except the report-only files marked below.

| files | content |
|---|---|
| `<system>.test.json`, `.test.perq.npz`, `.test.mixed.json` | TEST split, per slice and direction (MRR@10, Hit@k, nDCG@10), per-query MRR@10, mixed-index diagnostic |
| `<system>.dev.json`, `<system>.devM.json`, `<system>.train.json` | DEV split (selection), the held-out MIRACL-ar training slice (selection only), and the training log of the run |
| `<system>>>rr_*.test.json` | first stage `<system>` followed by the unchanged v1 cross-encoder reranker |
| `alpha_curve.json` | post-hoc WiSE trade-off curve (figure in the report, marked post hoc) |
| `confirmatory.json`, `mteb_paired.json`, `mteb_subsets.json`, `mteb_board.json`, `mteb_eval_time.json`, `suite_pick.json`, `ood_dev_ar.json` | confirmatory seeds, paired MTEB statistics, per-subset MTEB means, MTEB scoring time, the validation-suite pick, the mMARCO proxy scores |
| `exact_overlap.json`, `leak_exact.json`, `guard_split/` | exact-copy counts of MIRACL-HN ar dev against the TyDi-ar training questions (893 of 1,000; flagged 895 by the fixed detector, 814 / 816 by the old one), the leak check of kept training rows, and the corrected clean / flagged split of the guard tasks (per-query scores) |
| `audit_ood*.json`, `diag_xquad_style.json` | overlap audits of evaluation sets with the training data, XQuAD-style diagnostic |
| `PROGRESS.md`, `RESULTS_v2.md` | lab notebook (source of the pre-registration trail) and the results write-up the report reproduces |
| `data/prereg_trail.json` | pre-registration entries extracted verbatim from `PROGRESS.md` by `report/extract_prereg.py` (report-only) |
| `data/protocol.json`, `data/licences.json` | protocol constants that exist only in the notebook, each with its source; dataset and model licences verified on the Hub (report-only) |
| `models/*/` | training arguments of the final models and the WiSE blend (report-only, pulled by `report/pull_spark_extras.sh`) |

System names: `base_*` off-the-shelf; `v1_s13` the v1 recipe; `m3_b_s13` / `m3_b_s16` / `m3_b_s17` the raw fine-tuned bge-m3 per seed;
`final_m3` = v2-m3 (WiSE alpha 0.4, seed 13) and `w_m3b_a40_s14` .. `s17` the same recipe on the other seeds; `st_b_soup` / `st_s_soup` = v2-base / v2-small
(KD-only students, 3-seed soups). `p1_*`, `p2_*`, `p3_*`, `f4_*`, `f5_*`, `m_v1_*` and `wm_a75` are earlier ablations and candidates kept for the
ablation and negative-results tables. Refresh with `bash report/import_snapshot.sh <checkout>` and rebuild with `make -C report pdf`.
