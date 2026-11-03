| suite | what is compared | reading | gemma: P(x chosen) | gemma: picks slot A | gemma: order-consistent | qwen: P(x chosen) | qwen: picks slot A | qwen: order-consistent |
|---|---|---|---|---|---|---|---|---|
| quality | correct vs wrong-number answer | x = correct; should be 100% | 100% [81, 100] (n=16) | 50% | 100% | 98% [89, 100] (n=48) | 48% | 96% |
| equal | ASCII-digit answer vs same answer in Arabic-Indic digits (both correct) | x = ASCII version; 50% if script-invariant | 100% [81, 100] (n=16) | 50% | 100% | 90% [78, 95] (n=48) | 56% | 79% |
| pad_equal | short correct vs same + content-free fluff (both correct) | x = short; 50% if no verbosity effect, <50% = prefers long | 100% [81, 100] (n=16) | 50% | 100% | 46% [33, 60] (n=48) | 92% | 17% |
| pad_wrong | short correct vs wrong-number answer + fluff | x = short correct; <100% = padding rescued a wrong answer | 100% [81, 100] (n=16) | 50% | 100% | 92% [80, 97] (n=48) | 50% | 83% |
| self | Gemma-written vs Qwen-written correct answers | x = Gemma-written; <50% = prefers Qwen-written | 39% [26, 55] (n=38) | 42% | 74% | 35% [25, 46] (n=80) | 48% | 70% |

gemma: self-preference suite used 19 pairs; Qwen-written / Gemma-written length ratio = 0.94.

qwen: self-preference suite used 40 pairs; Qwen-written / Gemma-written length ratio = 0.94.
