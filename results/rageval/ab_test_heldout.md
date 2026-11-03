### Config comparison (correctness judge: gemma:correct_v1; grounding judge: qwen:cite_v1); 95% bootstrap CI over questions

| config | acceptable (all) | success (all) | success (answerable) | success_str | grounded | gold_hit | over-refusal | unans. abstain | right-language (of answered) |
|---|---|---|---|---|---|---|---|---|---|
| heldout_A_baseline | 58.3% [43.8, 72.9] (n=48) | 79.2% [66.7, 89.6] (n=48) | 79.2% [66.7, 89.6] (n=48) | 52.1% [37.5, 64.6] (n=48) | 100.0% [100.0, 100.0] (n=48) | 79.2% [66.7, 89.6] (n=48) | 18.8% [8.3, 31.2] (n=48) | nan% [nan, nan] (n=0) | 74.4% [61.5, 87.2] (n=39) |
| heldout_I_rerank_lang | 85.4% [75.0, 93.8] (n=48) | 85.4% [75.0, 93.8] (n=48) | 85.4% [75.0, 93.8] (n=48) | 64.6% [50.0, 77.1] (n=48) | 100.0% [100.0, 100.0] (n=48) | 87.5% [77.1, 95.8] (n=48) | 10.4% [2.1, 18.8] (n=48) | nan% [nan, nan] (n=0) | 100.0% [100.0, 100.0] (n=43) |

### Paired differences vs heldout_A_baseline (positive = better; over_refusal: negative = better)

| config | metric | diff [95% CI] | bootstrap p | McNemar p (b vs c) | n paired |
|---|---|---|---|---|---|
| heldout_I_rerank_lang | acceptable | +27.1 pts [+14.6, +39.6] | 0.0000 | 0.0002 (13 vs 0) | 48 |
| heldout_I_rerank_lang | success | +6.2 pts [+0.0, +14.6] | 0.0902 | 0.2500 (3 vs 0) | 48 |
| heldout_I_rerank_lang | success_str | +12.5 pts [+4.2, +22.9] | 0.0030 | 0.0312 (6 vs 0) | 48 |
| heldout_I_rerank_lang | gold_hit | +8.3 pts [+2.1, +16.7] | 0.0320 | 0.1250 (4 vs 0) | 48 |
| heldout_I_rerank_lang | grounded | +0.0 pts [+0.0, +0.0] | 1.0000 | 1.0000 (0 vs 0) | 48 |
| heldout_I_rerank_lang | lang_ok | +25.6 pts [+12.8, +38.5] | 0.0000 | 0.0020 (10 vs 0) | 39 |
| heldout_I_rerank_lang | over_refusal | -8.3 pts [-16.7, -2.1] | 0.0344 | 0.1250 (0 vs 4) | 48 |

Rough minimum detectable paired difference at n=48, 80% power, base rate 0.7, within-question correlation 0.5: 18.5 points.