| model | ST 6.2 loads original | ST 5.3 loads original | ST 5.3 loads compat | ST 6.2 loads compat | max_seq_length (6 orig / 5 compat) | pooling (6 orig / 5 compat) | max abs diff (6 orig vs 5 compat) | min cosine | check |
|---|---|---|---|---|---|---|---|---|---|
| v2-m3 | yes | no | yes | yes | 256 / 256 | cls / cls | 4.25e-07 | 0.99999982 | PASS |
| v2-base | yes | no | yes | yes | 256 / 256 | mean / mean | 7.60e-07 | 0.99999988 | PASS |
| v2-small | yes | no | yes | yes | 256 / 256 | mean / mean | 3.07e-07 | 0.99999988 | PASS |
