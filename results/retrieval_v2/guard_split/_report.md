
**MIRACL-HN.v2 ar (nDCG@10 over the full corpus)** - 1000 queries: both 18, clean 184, miracl 2, tydi 796

| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |
|---|---:|---:|---:|---:|
| bge-m3 | 75.7 | 73.1 (n=184) | 76.2 (n=816) | -3.1 |
| e5-base | 70.4 | 67.3 (n=184) | 71.1 (n=816) | -3.9 |
| m3_b_s13 | 66.8 | 64.9 (n=184) | 67.3 (n=816) | -2.4 |
| v1 | 64.0 | 61.8 (n=184) | 64.5 (n=816) | -2.7 |
| wm_a75 | 70.2 | 68.8 (n=184) | 70.6 (n=816) | -1.8 |

**NamaaMrTydi (average precision over the candidate lists)** - 918 queries: both 6, clean 894, miracl 1, tydi 17

| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |
|---|---:|---:|---:|---:|
| bge-m3 | 79.4 | 79.1 (n=894) | 89.2 (n=24) | -10.0 |
| e5-base | 77.4 | 77.2 (n=894) | 82.6 (n=24) | -5.4 |
| m3_b_s13 | 69.4 | 69.3 (n=894) | 73.8 (n=24) | -4.5 |
| v1 | 65.8 | 65.6 (n=894) | 72.5 (n=24) | -6.9 |
| wm_a75 | 72.1 | 71.9 (n=894) | 79.3 (n=24) | -7.4 |