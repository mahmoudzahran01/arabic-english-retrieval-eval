### Config comparison (correctness judge: qwen:correct_v1; grounding judge: qwen:cite_v1); 95% bootstrap CI over questions

| config | acceptable (all) | success (all) | success (answerable) | success_str | grounded | gold_hit | over-refusal | unans. abstain | right-language (of answered) |
|---|---|---|---|---|---|---|---|---|---|
| A_baseline | 55.0% [47.8, 62.2] (n=180) | 72.2% [65.6, 78.9] (n=180) | 68.8% [61.3, 75.6] (n=160) | 56.1% [48.9, 63.3] (n=180) | 100.0% [100.0, 100.0] (n=180) | 81.2% [75.0, 87.5] (n=160) | 21.2% [15.0, 28.1] (n=160) | 100.0% [100.0, 100.0] (n=20) | 73.0% [65.1, 80.2] (n=126) |
| I_rerank_lang | 78.9% [72.8, 85.0] (n=180) | 78.9% [72.8, 85.0] (n=180) | 76.2% [69.4, 82.5] (n=160) | 70.0% [63.3, 76.7] (n=180) | 100.0% [100.0, 100.0] (n=180) | 91.9% [87.5, 95.6] (n=160) | 11.2% [6.9, 16.2] (n=160) | 100.0% [100.0, 100.0] (n=20) | 100.0% [100.0, 100.0] (n=142) |

### Paired differences vs A_baseline (positive = better; over_refusal: negative = better)

| config | metric | diff [95% CI] | bootstrap p | McNemar p (b vs c) | n paired |
|---|---|---|---|---|---|
| I_rerank_lang | acceptable | +23.9 pts [+16.7, +31.1] | 0.0000 | 0.0000 (49 vs 6) | 180 |
| I_rerank_lang | success | +6.7 pts [+0.6, +12.8] | 0.0356 | 0.0428 (21 vs 9) | 180 |
| I_rerank_lang | success_str | +13.9 pts [+7.8, +20.0] | 0.0000 | 0.0000 (30 vs 5) | 180 |
| I_rerank_lang | gold_hit | +10.6 pts [+5.0, +16.2] | 0.0004 | 0.0005 (20 vs 3) | 160 |
| I_rerank_lang | grounded | +0.0 pts [+0.0, +0.0] | 1.0000 | 1.0000 (0 vs 0) | 180 |
| I_rerank_lang | lang_ok | +26.8 pts [+19.5, +35.0] | 0.0000 | 0.0000 (33 vs 0) | 123 |
| I_rerank_lang | over_refusal | -10.0 pts [-15.6, -5.0] | 0.0004 | 0.0009 (3 vs 19) | 160 |

Rough minimum detectable paired difference at n=180, 80% power, base rate 0.7, within-question correlation 0.5: 9.6 points.