# v2.1 round - pre-registered attempt at the three unmet S-tier items

Written after the round; every number comes from stored files. The full pre-registration (timestamped 2026-11-05 ~08:50, before any v2.1 run) is in `PROGRESS.md` under "PRE-REGISTRATION v2.1". v2 (`w_m3b_a40`) stayed frozen as the incumbent; v2.1 replaces it only under the rules below.

## 1. What was pre-registered

* **Targets:** 2a NamaaMrTydi within 1.0 of bge-m3 (v2: -3.4); 2b MTEB(ara) Mean(Task) statistically above bge-m3 (v2: +0.31, CI includes 0); 3b ar-en cross-lingual subsets above multilingual-e5-large (75.3 vs v2's 75.0).
* **Levers (two):** (a) the 23,418 data-v3 translations (slice `train_xl3_ar2en`; not the 23,416 English originals), (b) Arabic-monolingual hard negatives re-mined with v2-m3 for the TyDi-ar / ARCD rows.
* **Grid:** three trainings (R_A, R_B, R_AB; seed 13; the m3_b recipe otherwise unchanged) x WiSE alpha {0.3, 0.4, 0.5, 0.6} = 12 selectable models; the incumbent family are reference rows and the incumbent `w_m3b_a40` is the "stay" option.
* **Selection:** the v2 six-component validation suite (no TEST, no MTEB(ara) score of any candidate), plus one documented extension (component 7, a Namaa-like Arabic reranking proxy) that is only used if it passes a validity gate. The best eligible candidate goes to confirmation only if it beats the incumbent's score by at least 0.02.
* **Adoption (judged once, mean of fresh seeds 18 and 19):** TEST ar->en >= v2 mean - 1 sd and macro >= v2 mean - 1 sd; MTEB NamaaMrTydi >= bge-m3 - 1.0; MTEB Mean(Task) >= 72.29. Otherwise v2 stays and v2.1 is a negative result.

## 2. Contamination re-run before any training

The v2.1 list drops 3,276 of 90,266 training queries (v2 list: 3,223; 53 newly dropped, none un-dropped): exact normalised copies are now flagged and the NamaaMrTydi test queries and positives are part of the evaluation sets. **Exact-match leaks among the KEPT rows after the fix: MIRACL-HN.v2 ar dev 0 (998 normalised queries), NamaaMrTydi 0 (916); v2's list had 9 for MIRACL-HN.** The Jaccard detector's blind spot for near-duplicates of short common-word questions remains (only exact copies were added).

## 3. The Namaa-like proxy and its validity gate (fixed before rho was computed)

Spearman rho = **0.490** over 14 existing models (threshold 0.7, minimum 12 models) -> **FAILED: component 7 dropped, the six-component v2 suite is used unchanged**.

| model | ar_rerank AP (350 queries) | NamaaMrTydi AP (MTEB) |
|---|---:|---:|
| v1 | 76.43 | 65.91 |
| final_base | 77.65 | 68.19 |
| final_small | 76.43 | 68.87 |
| m3_b_s13 | 83.22 | 69.34 |
| wm_a75 | 84.12 | 72.14 |
| w_m3b_a60 | 84.14 | 73.70 |
| wm_a50 | 84.19 | 74.71 |
| w_m3b_a40_s17 | 84.23 | 75.54 |
| w_m3b_a40_s16 | 84.23 | 75.71 |
| w_m3b_a40 | 84.22 | 76.05 |
| w_m3b_a30 | 84.15 | 77.03 |
| e5-base | 77.34 | 77.13 |
| e5-small | 76.83 | 79.23 |
| bge-m3 | 84.46 | 79.44 |

## 4. The three trainings (DEV, seed 13)

| run | what | examples | train s | DEV ar->en | DEV macro | devM |
|---|---|---:|---:|---:|---:|---:|
| `m21_A_s13` | R_A: v2 data + data-v3 translations | 62,408 | 3585 | 81.8 | 85.6 | 78.5 |
| `m21_B_s13` | R_B: v2 data + Arabic hard negatives re-mined with v2-m3 | 39,332 | 2152 | 81.0 | 85.2 | 77.4 |
| `m21_AB_s13` | R_AB: both levers | 62,408 | 3326 | 82.0 | 85.4 | 76.8 |
| `m3_b_s13` | incumbent raw fine-tune (v2) | 39,377 | 2893 | 81.0 | 85.3 | 79.1 |

## 5. Selection on the validation suite

| model | recipe | alpha | dev | ar->en | devM | proxy | XNLI(tr) | cls | STS | ar_rerank | score (6) | score (7) | **used** | eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| w_m3b_a40 | stay | 0.40 | 83.5 | 79.0 | 80.3 | 81.9 | 79.1 | 56.9 | 81.9 | 84.2 | 0.6706 | 0.6796 | **0.6706** | yes |
| w_m3b_a30 | ref | 0.30 | 82.7 | 78.3 | 80.3 | 81.8 | 79.3 | 56.9 | 81.8 | 84.1 | 0.6432 | 0.6489 | **0.6432** | ref |
| wm_a50 | ref | 0.50 | 84.2 | 79.8 | 79.8 | 81.8 | 78.8 | 57.0 | 82.0 | 84.2 | 0.6248 | 0.6375 | **0.6248** | ref |
| w_m21_A_a30 | A | 0.30 | 82.6 | 78.6 | 80.0 | 82.2 | 79.2 | 56.9 | 81.7 | 84.2 | 0.6201 | 0.6347 | **0.6201** | yes |
| w_m21_B_a30 | B | 0.30 | 82.2 | 78.2 | 80.3 | 81.8 | 79.3 | 56.9 | 81.8 | 84.6 | 0.6057 | 0.6620 | **0.6057** | no |
| w_m21_A_a40 | A | 0.40 | 83.6 | 79.4 | 79.7 | 82.0 | 78.9 | 56.8 | 81.8 | 84.0 | 0.5759 | 0.5718 | **0.5759** | yes |
| w_m21_A_a50 | A | 0.50 | 84.5 | 80.6 | 80.0 | 81.9 | 78.6 | 56.7 | 81.8 | 84.0 | 0.5755 | 0.5768 | **0.5755** | yes |
| w_m21_B_a40 | B | 0.40 | 82.9 | 78.9 | 80.2 | 81.5 | 79.1 | 56.9 | 81.8 | 84.4 | 0.5743 | 0.6203 | **0.5743** | yes |
| w_m3b_a60 | ref | 0.60 | 84.5 | 80.1 | 79.6 | 81.6 | 78.5 | 56.9 | 82.0 | 84.1 | 0.5653 | 0.5816 | **0.5653** | ref |
| bge-m3 | ref | 0.00 | 78.2 | 73.5 | 80.5 | 81.9 | 80.0 | 56.8 | 81.5 | 84.5 | 0.5442 | 0.5967 | **0.5442** | ref |
| w_m21_A_a60 | A | 0.60 | 84.9 | 80.7 | 79.9 | 81.7 | 78.3 | 56.6 | 81.8 | 83.9 | 0.5201 | 0.5222 | **0.5201** | yes |
| w_m21_AB_a40 | AB | 0.40 | 83.1 | 79.4 | 79.7 | 81.5 | 78.9 | 56.6 | 81.8 | 84.0 | 0.4757 | 0.4872 | **0.4757** | yes |
| w_m21_AB_a30 | AB | 0.30 | 82.3 | 78.4 | 79.5 | 81.5 | 79.2 | 56.8 | 81.8 | 83.9 | 0.4687 | 0.4732 | **0.4687** | no |
| w_m21_B_a50 | B | 0.50 | 83.4 | 79.6 | 79.5 | 81.3 | 78.8 | 56.7 | 81.9 | 84.1 | 0.4566 | 0.4846 | **0.4566** | yes |
| w_m21_AB_a50 | AB | 0.50 | 83.7 | 80.3 | 79.4 | 81.4 | 78.6 | 56.5 | 81.9 | 83.8 | 0.4154 | 0.4126 | **0.4154** | yes |
| w_m21_B_a60 | B | 0.60 | 84.1 | 80.4 | 79.4 | 81.0 | 78.6 | 56.6 | 81.9 | 83.9 | 0.3940 | 0.4059 | **0.3940** | yes |
| m3_b_s13 | ref | 1.00 | 85.3 | 81.0 | 79.1 | 80.8 | 77.5 | 56.7 | 82.2 | 83.2 | 0.3478 | 0.2981 | **0.3478** | ref |
| w_m21_AB_a60 | AB | 0.60 | 84.2 | 80.9 | 78.9 | 81.4 | 78.3 | 56.4 | 81.9 | 83.6 | 0.3389 | 0.3269 | **0.3389** | yes |

**Decision (pre-registered rule): STAY** - pick `w_m3b_a40`. best eligible score w_m3b_a40 0.6706; tie band < 0.02 contains ['w_m3b_a40'] -> ingredients clause picks w_m3b_a40; incumbent w_m3b_a40 0.6706; chosen minus incumbent = +0.0000 -> NOT >= incumbent + 0.02: v2 stays, nothing further is spent.

Ineligible by the pre-registered rule (DEV ar->en < bge-m3 + 5.0 = 78.5): `w_m21_B_a30` (78.2), `w_m21_AB_a30` (78.4). Best eligible v2.1 candidate: `w_m21_A_a30` with score 0.6201 vs the incumbent `w_m3b_a40` 0.6706 (-0.0506; the confirmation threshold was +0.02). At equal alpha the differences between a v2.1 candidate and the incumbent family are small on every component (mostly <= 0.8 points): recipe A (data v3) gains 0.3-0.8 DEV ar->en at every alpha but is not consistently better on devM or the proxy; recipe B (re-mined Arabic negatives) is level with the incumbent on DEV (-0.2 to +0.3 ar->en) and on devM (-0.3 to 0); recipe AB is 0.4-0.8 below the incumbent on devM at every alpha. The incumbent at alpha 0.4 has the best combination under the min-max scoring.

## 6. Confirmation and adoption decision

No confirmation run was made (see the selection decision above), so there is no adoption evidence and **v2 stays**.

## 7. GPU time

Selection queue (drive18): 3.3 h of gated step wall-clock between 08:19:44 and 11:46:55 (the first step also waited ~10 min for the gate's quiet window, not counted), of which 2.5 h are the three trainings (A 60 min, B 36 min, AB 55 min); the rest is mining (~2 min), the proxy runs, 12 blends x (merge + DEV + devM + proxy + suite + ar_rerank) and the selection. Budget was ~8 GPU-hours for selection; the confirmation budget (~4 h: seeds 18 and 19, TEST, two full MTEB(ara) scorings) was **not spent** because the pre-registered rule said stay.

## 8. Scorecard per S-tier item after the round

| S-tier item | v2 (incumbent) | after the v2.1 round |
|---|---|---|
| 1. ar->en >= bge-m3, macro > bge-m3 (TEST) | met by v2 (71.2 vs 63.3) | unchanged |
| 2a. NamaaMrTydi within 1.0 of bge-m3 | v2: 76.0 vs 79.4 (-3.4) | v2 (kept): 76.0 -> NOT MET **[UNMET]** |
| 2b. MTEB(ara) mean statistically above bge-m3 | v2: 72.29 vs 71.98, CI includes 0 | v2 (kept): 72.29 -> NOT MET (unchanged) **[UNMET]** |
| 3a. #1 among <=300M models | met by v2-base (68.91) | unchanged (students are redone only if v2.1 is adopted) |
| 3b. #1 on ar-en cross-lingual subsets | v2: 75.0 vs e5-large 75.3 | v2 (kept): 75.0 -> NOT MET **[UNMET]** |
| 4. contamination audit + licences | met with a disclosed exact-copy leak (9 rows) | v2.1 training uses the exact-match-fixed list: 0 exact leaks for MIRACL-HN and Namaa |
| 5. full-pipeline numbers with the reranker | reported for v2 (section 9) | unchanged |

Items still unmet after the v2.1 round: **2a, 2b, 3b**.

## 9. Conclusion

**v2 stays; v2.1 is a negative result.** Under the pre-registered rule no v2.1 candidate beat the incumbent's suite score by the required margin, so nothing was sent to confirmation and the adoption criteria (TEST, MTEB Namaa >= bge-m3 - 1.0, Mean(Task) >= 72.29) were never evaluated for v2.1; the three unmet items (2a Namaa, 2b statistically-above-bge-m3 mean, 3b ar-en cross-lingual subsets vs e5-large) are exactly as unmet as before. What the round does show: (1) the exact-copy leak is fixed in the training list (0 exact leaks for MIRACL-HN and Namaa); (2) at one seed the two levers change the DEV macro of the raw fine-tune by less than 0.5 points and do not improve devM (training table), and no blended candidate improves on the incumbent's suite score; (3) the Namaa-like proxy built from MIRACL-train judged candidates does not track Namaa across the existing models (rho = 0.49), so the suite still cannot see the Namaa loss; the loss would have to be judged on Namaa itself, which the protocol forbids for selection. Whether a larger change (for example a lower fine-tuned share than 0.3, which fails the ar->en eligibility floor, or a different training objective on Arabic-only data) can close 2a without giving up ar->en is untested.
