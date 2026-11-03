### Judge calibration vs planted ground truth - split=dev (n_items=89)

| judge | axis | n | defects | precision | recall | F1 | kappa [95% CI] | acc | unparsable |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| gemma:overall_pipeline_v3 | acceptable | 0 | 0 | nan | nan | nan | nan [nan, nan] | nan | 0 |
| gemma:overall_v0 | acceptable | 89 | 70 | 1.00 | 0.81 | 0.90 | 0.65 [0.52, 0.79] | 0.85 | 0 |
| qwen:overall_pipeline | acceptable | 89 | 70 | 1.00 | 0.99 | 0.99 | 0.97 [0.88, 1.00] | 0.99 | 0 |
| qwen:overall_pipeline_v3 | acceptable | 89 | 70 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| qwen:overall_v0 | acceptable | 89 | 70 | 0.97 | 0.41 | 0.58 | 0.20 [0.13, 0.33] | 0.53 | 0 |
| gemma:cite_v1 | attributed | 0 | 0 | nan | nan | nan | nan [nan, nan] | nan | 0 |
| qwen:cite_v1 | attributed | 89 | 44 | 1.00 | 0.86 | 0.93 | 0.86 [0.72, 0.96] | 0.93 | 1 |
| gemma:correct_v0 | correct | 89 | 45 | 1.00 | 0.93 | 0.97 | 0.93 [0.84, 1.00] | 0.97 | 0 |
| gemma:correct_v1 | correct | 0 | 0 | nan | nan | nan | nan [nan, nan] | nan | 0 |
| qwen:correct_v0 | correct | 89 | 45 | 0.90 | 0.78 | 0.83 | 0.69 [0.52, 0.83] | 0.84 | 0 |
| qwen:correct_v1 | correct | 89 | 45 | 0.94 | 1.00 | 0.97 | 0.93 [0.85, 1.00] | 0.97 | 0 |
| ragas[qwen]:FC-f1>=0.10 | correct | 50 | 26 | 0.83 | 0.77 | 0.80 | 0.60 [0.36, 0.82] | 0.80 | 0 |
| ragas[qwen]:FC-precision>=0.10 | correct | 50 | 26 | 0.87 | 0.77 | 0.82 | 0.64 [0.40, 0.86] | 0.82 | 0 |
| ragas[qwen]:FC-recall>=0.10 | correct | 50 | 26 | 0.87 | 0.77 | 0.82 | 0.64 [0.40, 0.86] | 0.82 | 0 |
| string_match:contains_ref | correct | 89 | 45 | 0.90 | 1.00 | 0.95 | 0.89 [0.81, 0.97] | 0.94 | 0 |
| gemma:faith_v0 | faithful | 79 | 33 | 0.69 | 1.00 | 0.81 | 0.63 [0.53, 0.74] | 0.81 | 0 |
| gemma:faith_v3 | faithful | 0 | 0 | nan | nan | nan | nan [nan, nan] | nan | 0 |
| qwen:faith_v0 | faithful | 79 | 33 | 0.64 | 0.85 | 0.73 | 0.48 [0.28, 0.68] | 0.73 | 0 |
| qwen:faith_v1 | faithful | 79 | 33 | 0.89 | 0.97 | 0.93 | 0.87 [0.77, 0.97] | 0.94 | 0 |
| qwen:faith_v2 | faithful | 79 | 33 | 0.97 | 0.85 | 0.90 | 0.84 [0.70, 0.95] | 0.92 | 0 |
| qwen:faith_v3 | faithful | 79 | 33 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| ragas[qwen]:faithfulness==1 | faithful | 45 | 21 | 0.70 | 1.00 | 0.82 | 0.61 [0.40, 0.81] | 0.80 | 0 |

### Flag rate by planted type (fraction of items the judge marks as DEFECT) - split=dev

| judge | clean (n=7) | numeral-var (n=2) | wrong# (n=7) | wrong#-AR-digits (n=4) | wrong-ent (n=8) | halluc+ (n=4) | omitted (n=6) | wrong-lang (n=10) | wrong-cite (n=11) | ok-abstain (n=10) | bad-abstain (n=10) | halluc-unans (n=10) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma:overall_pipeline_v3 [acceptable] | - | - | - | - | - | - | - | - | - | - | - | - |
| gemma:overall_v0 [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 0.88 | 0.50 | 1.00 | 0.50 | 0.55 | 0.00 | 1.00 | 1.00 |
| qwen:overall_pipeline [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| qwen:overall_pipeline_v3 [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| qwen:overall_v0 [acceptable] | 0.00 | 0.00 | 0.86 | 0.75 | 1.00 | 0.25 | 0.00 | 0.00 | 0.09 | 0.10 | 0.00 | 1.00 |
| gemma:cite_v1 [attributed] | - | - | - | - | - | - | - | - | - | - | - | - |
| qwen:cite_v1 [attributed] | 0.00 | 0.00 | 0.86 | 1.00 | 0.88 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| gemma:correct_v0 [correct] | 0.00 | 0.00 | 0.86 | 1.00 | 0.88 | 0.00 | 0.83 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| gemma:correct_v1 [correct] | - | - | - | - | - | - | - | - | - | - | - | - |
| qwen:correct_v0 [correct] | 0.14 | 0.00 | 0.86 | 1.00 | 1.00 | 0.25 | 0.33 | 0.10 | 0.09 | 0.00 | 0.60 | 0.90 |
| qwen:correct_v1 [correct] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.10 | 0.00 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-f1>=0.10 [correct] | 0.00 | 0.50 | 0.67 | 0.50 | 1.00 | 0.00 | 0.00 | 0.33 | 0.20 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-precision>=0.10 [correct] | 0.00 | 0.50 | 0.67 | 0.50 | 1.00 | 0.00 | 0.00 | 0.17 | 0.20 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-recall>=0.10 [correct] | 0.00 | 0.50 | 0.67 | 0.50 | 1.00 | 0.00 | 0.00 | 0.17 | 0.20 | 0.00 | 1.00 | 1.00 |
| string_match:contains_ref [correct] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.00 | 0.00 | 1.00 | 1.00 |
| gemma:faith_v0 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 | 1.00 | 0.00 | 0.10 | 1.00 |
| gemma:faith_v3 [faithful] | - | - | - | - | - | - | - | - | - | - | - | - |
| qwen:faith_v0 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.17 | 0.10 | 0.36 | 1.00 | 0.60 | 0.80 |
| qwen:faith_v1 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.00 | 0.10 | 0.27 | 0.00 | 0.50 | 1.00 |
| qwen:faith_v2 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 0.62 | 0.50 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 | 1.00 |
| qwen:faith_v3 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| ragas[qwen]:faithfulness==1 [faithful] | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.17 | 0.00 | 1.00 | 1.00 | 1.00 |

Defect types the judge is *meant* to catch should be near 1.00; clean/ok types should be near 0.00.

#### Ablation of `gemma:overall_pipeline_v3` (axis acceptable, split=dev)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.00 | nan | 0.00 | 0.00 |
| + language check | 0.00 | nan | 0.00 | 0.00 |
| + abstention check | 0.00 | nan | 0.00 | 0.00 |
| full pipeline (+both deterministic checks) | 0.00 | nan | 0.00 | 0.00 |
| deterministic checks only (language+abstention) | 0.00 | nan | 0.00 | 0.00 |

#### Ablation of `qwen:overall_pipeline` (axis acceptable, split=dev)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.86 | 1.00 | 0.92 | 0.72 |
| + language check | 0.99 | 1.00 | 0.99 | 0.97 |
| + abstention check | 0.86 | 1.00 | 0.92 | 0.72 |
| full pipeline (+both deterministic checks) | 0.99 | 1.00 | 0.99 | 0.97 |
| deterministic checks only (language+abstention) | 0.43 | 1.00 | 0.60 | 0.24 |

#### Ablation of `qwen:overall_pipeline_v3` (axis acceptable, split=dev)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.87 | 1.00 | 0.93 | 0.74 |
| + language check | 1.00 | 1.00 | 1.00 | 1.00 |
| + abstention check | 0.87 | 1.00 | 0.93 | 0.74 |
| full pipeline (+both deterministic checks) | 1.00 | 1.00 | 1.00 | 1.00 |
| deterministic checks only (language+abstention) | 0.43 | 1.00 | 0.60 | 0.24 |

#### Per question-language (split=dev): kappa / defect recall / false-positive rate on clean-type items

| judge | en: n | en kappa | en recall | en FPR | ar: n | ar kappa | ar recall | ar FPR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma:cite_v1 | 0 | nan | nan | nan | 0 | nan | nan | nan |
| gemma:correct_v0 | 51 | 0.92 | 0.92 | 0.00 | 38 | 0.95 | 0.95 | 0.00 |
| gemma:correct_v1 | 0 | nan | nan | nan | 0 | nan | nan | nan |
| gemma:faith_v0 | 45 | 0.58 | 1.00 | 0.00 | 34 | 0.71 | 1.00 | 0.00 |
| gemma:faith_v3 | 0 | nan | nan | nan | 0 | nan | nan | nan |
| gemma:overall_pipeline_v3 | 0 | nan | nan | nan | 0 | nan | nan | nan |
| gemma:overall_v0 | 51 | 0.67 | 0.82 | 0.00 | 38 | 0.63 | 0.80 | 0.00 |
| qwen:cite_v1 | 51 | 0.88 | 0.88 | 0.00 | 38 | 0.84 | 0.85 | 0.00 |
| qwen:correct_v0 | 51 | 0.65 | 0.72 | 0.09 | 38 | 0.74 | 0.85 | 0.00 |
| qwen:correct_v1 | 51 | 0.96 | 1.00 | 0.00 | 38 | 0.89 | 1.00 | 0.00 |
| qwen:faith_v0 | 45 | 0.51 | 0.82 | 0.55 | 34 | 0.42 | 0.88 | 0.50 |
| qwen:faith_v1 | 45 | 0.91 | 0.94 | 0.00 | 34 | 0.82 | 1.00 | 0.00 |
| qwen:faith_v2 | 45 | 0.80 | 0.76 | 0.00 | 34 | 0.88 | 0.94 | 0.00 |
| qwen:faith_v3 | 45 | 1.00 | 1.00 | 0.00 | 34 | 1.00 | 1.00 | 0.00 |
| qwen:overall_pipeline | 51 | 1.00 | 1.00 | 0.00 | 38 | 0.92 | 0.97 | 0.00 |
| qwen:overall_pipeline_v3 | 51 | 1.00 | 1.00 | 0.00 | 38 | 1.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 51 | 0.14 | 0.35 | 0.09 | 38 | 0.30 | 0.50 | 0.00 |

#### Number errors: defect recall by subtype and digit script, and false-positive rate on correct numeral variants (split=dev; n shown for judges that scored every item)

| judge | near (+-small) (n=3) | scale (x10) (n=0) | far (n=8) | ASCII digits (n=7) | Arabic-Indic digits (n=4) | FPR on correct numeral variants (n=2) |
|---|--:|--:|--:|--:|--:|--:|
| gemma:correct_v0 | 1.00 | - | 0.88 | 0.86 | 1.00 | 0.00 |
| gemma:correct_v1 | - | - | - | - | - | - |
| gemma:faith_v0 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v3 | - | - | - | - | - | - |
| gemma:overall_pipeline_v3 | - | - | - | - | - | - |
| gemma:overall_v0 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:correct_v0 | 0.67 | - | 1.00 | 0.86 | 1.00 | 0.00 |
| qwen:correct_v1 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v0 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v1 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v2 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v3 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_pipeline | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_pipeline_v3 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 1.00 | - | 0.75 | 0.86 | 0.75 | 0.00 |
| ragas[qwen]:FC-f1>=0.10 | 0.00 | - | 0.67 | 0.67 | 0.50 | 0.50 |
| ragas[qwen]:FC-precision>=0.10 | 0.00 | - | 0.67 | 0.67 | 0.50 | 0.50 |
| ragas[qwen]:FC-recall>=0.10 | 0.00 | - | 0.67 | 0.67 | 0.50 | 0.50 |
| ragas[qwen]:faithfulness==1 | 1.00 | - | 1.00 | 1.00 | 1.00 | 1.00 |
| string_match:contains_ref | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.00 |

#### Question language != evidence language (cross-lingual) vs same language: defect recall / clean FPR (split=dev)

| judge | cross: recall | cross: FPR | mono: recall | mono: FPR |
|---|--:|--:|--:|--:|
| gemma:cite_v1 | nan | nan | nan | nan |
| gemma:correct_v0 | 0.92 | 0.00 | 0.95 | 0.00 |
| gemma:correct_v1 | nan | nan | nan | nan |
| gemma:faith_v0 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:faith_v3 | nan | nan | nan | nan |
| gemma:overall_pipeline_v3 | nan | nan | nan | nan |
| gemma:overall_v0 | 0.81 | 0.00 | 0.82 | 0.00 |
| qwen:cite_v1 | 0.87 | 0.00 | 0.86 | 0.00 |
| qwen:correct_v0 | 0.83 | 0.08 | 0.71 | 0.00 |
| qwen:correct_v1 | 1.00 | 0.00 | 1.00 | 0.00 |
| qwen:faith_v0 | 0.76 | 0.46 | 0.94 | 0.67 |
| qwen:faith_v1 | 1.00 | 0.00 | 0.94 | 0.00 |
| qwen:faith_v2 | 0.82 | 0.00 | 0.88 | 0.00 |
| qwen:faith_v3 | 1.00 | 0.00 | 1.00 | 0.00 |
| qwen:overall_pipeline | 0.97 | 0.00 | 1.00 | 0.00 |
| qwen:overall_pipeline_v3 | 1.00 | 0.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 0.41 | 0.08 | 0.42 | 0.00 |
| ragas[qwen]:FC-f1>=0.10 | 0.81 | 0.11 | 0.70 | 0.00 |
| ragas[qwen]:FC-precision>=0.10 | 0.81 | 0.11 | 0.70 | 0.00 |
| ragas[qwen]:FC-recall>=0.10 | 0.81 | 0.11 | 0.70 | 0.00 |
| ragas[qwen]:faithfulness==1 | 1.00 | 0.67 | 1.00 | 1.00 |

#### Judge-vs-judge agreement (Cohen's kappa between the verdicts themselves, on the items both judges scored, split=dev)

| judge | n | gemma vs qwen: kappa | raw agreement |
|---|--:|--:|--:|
| correct_v0 | 89 | 0.66 | 0.83 |
| faith_v0 | 89 | 0.25 | 0.63 |
| overall_v0 | 89 | 0.32 | 0.63 |

#### Judge improvements: kappa before -> after, with paired cluster-bootstrap 95% CI of the difference (split=dev)

| axis | before | after | n | kappa before | kappa after | diff [95% CI] |
|---|---|---|--:|--:|--:|---|
| acceptable | qwen:overall_v0 | qwen:overall_pipeline_v3 | 89 | 0.20 | 1.00 | +0.80 [+0.67, +0.88] |
| correct | qwen:correct_v0 | qwen:correct_v1 | 89 | 0.69 | 0.93 | +0.25 [+0.11, +0.41] |
| faithful | qwen:faith_v0 | qwen:faith_v1 | 79 | 0.48 | 0.87 | +0.39 [+0.23, +0.57] |
| faithful | qwen:faith_v0 | qwen:faith_v3 | 79 | 0.48 | 1.00 | +0.52 [+0.32, +0.72] |
| faithful | qwen:faith_v2 | qwen:faith_v3 | 79 | 0.84 | 1.00 | +0.16 [+0.05, +0.31] |
| acceptable | qwen:overall_pipeline | qwen:overall_pipeline_v3 | 89 | 0.97 | 1.00 | +0.03 [+0.00, +0.12] |

Ragas notes: {"ragas[qwen]": {"items_scored": 96, "dev-tuned thresholds": {"FC-f1": 0.1, "FC-precision": 0.1, "FC-recall": 0.1}, "none_scores": {"faithfulness": 0, "factual_correctness": 0}}}