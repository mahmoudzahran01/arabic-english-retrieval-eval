
**MIRACL-HN.v2 ar (nDCG@10 over the full corpus)** - 1000 queries: both 20, clean 105, tydi 875

| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |
|---|---:|---:|---:|---:|
| bge-m3 | 79.0 | 71.9 (n=105) | 79.9 (n=895) | -8.0 |
| e5-base | 73.0 | 67.4 (n=105) | 73.6 (n=895) | -6.2 |
| e5-small | 72.4 | 64.7 (n=105) | 73.3 (n=895) | -8.5 |
| final_base | 70.7 | 61.6 (n=105) | 71.8 (n=895) | -10.2 |
| final_m3 | 77.9 | 70.5 (n=105) | 78.8 (n=895) | -8.3 |
| final_small | 68.3 | 57.1 (n=105) | 69.6 (n=895) | -12.5 |
| m3_b_s13 | 72.9 | 63.3 (n=105) | 74.0 (n=895) | -10.7 |
| v1 | 65.8 | 54.0 (n=105) | 67.2 (n=895) | -13.2 |
| wm_a75 | 75.7 | 67.7 (n=105) | 76.7 (n=895) | -8.9 |

**NamaaMrTydi (average precision over the candidate lists)** - 918 queries: both 6, clean 891, miracl 1, tydi 20

| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |
|---|---:|---:|---:|---:|
| bge-m3 | 79.4 | 79.1 (n=891) | 88.5 (n=27) | -9.4 |
| e5-base | 77.4 | 77.2 (n=891) | 82.7 (n=27) | -5.5 |
| e5-small | 79.3 | 79.2 (n=891) | 84.1 (n=27) | -4.9 |
| final_base | 68.1 | 68.2 (n=891) | 67.4 (n=27) | +0.8 |
| final_m3 | 76.0 | 75.9 (n=891) | 80.7 (n=27) | -4.8 |
| final_small | 68.9 | 68.7 (n=891) | 74.2 (n=27) | -5.5 |
| m3_b_s13 | 69.4 | 69.4 (n=891) | 69.6 (n=27) | -0.2 |
| v1 | 65.8 | 65.6 (n=891) | 70.7 (n=27) | -5.1 |
| wm_a75 | 72.1 | 72.0 (n=891) | 74.5 (n=27) | -2.5 |