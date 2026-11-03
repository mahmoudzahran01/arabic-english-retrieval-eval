### Judge calibration vs planted ground truth - split=all (n_items=180)

| judge | axis | n | defects | precision | recall | F1 | kappa [95% CI] | acc | unparsable |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| gemma:overall_pipeline_v3 | acceptable | 91 | 64 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| gemma:overall_v0 | acceptable | 180 | 134 | 1.00 | 0.89 | 0.94 | 0.80 [0.71, 0.88] | 0.92 | 0 |
| qwen:overall_pipeline | acceptable | 180 | 134 | 0.98 | 0.96 | 0.97 | 0.90 [0.82, 0.97] | 0.96 | 2 |
| qwen:overall_pipeline_v3 | acceptable | 180 | 134 | 0.99 | 1.00 | 1.00 | 0.99 [0.95, 1.00] | 0.99 | 0 |
| qwen:overall_v0 | acceptable | 180 | 134 | 0.98 | 0.48 | 0.64 | 0.31 [0.22, 0.41] | 0.61 | 0 |
| gemma:cite_v1 | attributed | 91 | 52 | 1.00 | 0.96 | 0.98 | 0.96 [0.88, 1.00] | 0.98 | 0 |
| qwen:cite_v1 | attributed | 180 | 96 | 1.00 | 0.83 | 0.91 | 0.82 [0.73, 0.91] | 0.91 | 1 |
| gemma:correct_v0 | correct | 180 | 86 | 1.00 | 0.90 | 0.94 | 0.90 [0.83, 0.96] | 0.95 | 0 |
| gemma:correct_v1 | correct | 91 | 41 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| qwen:correct_v0 | correct | 180 | 86 | 0.82 | 0.81 | 0.82 | 0.65 [0.53, 0.76] | 0.83 | 0 |
| qwen:correct_v1 | correct | 180 | 86 | 0.90 | 1.00 | 0.95 | 0.89 [0.82, 0.95] | 0.94 | 0 |
| ragas[qwen]:FC-f1>=0.10 | correct | 93 | 46 | 0.83 | 0.65 | 0.73 | 0.53 [0.32, 0.71] | 0.76 | 0 |
| ragas[qwen]:FC-precision>=0.10 | correct | 93 | 46 | 0.86 | 0.65 | 0.74 | 0.55 [0.35, 0.73] | 0.77 | 0 |
| ragas[qwen]:FC-recall>=0.10 | correct | 93 | 46 | 0.86 | 0.65 | 0.74 | 0.55 [0.35, 0.73] | 0.77 | 0 |
| string_match:contains_ref | correct | 180 | 86 | 0.92 | 1.00 | 0.96 | 0.92 [0.87, 0.97] | 0.96 | 0 |
| gemma:faith_v0 | faithful | 166 | 80 | 0.78 | 1.00 | 0.87 | 0.73 [0.65, 0.80] | 0.86 | 0 |
| gemma:faith_v3 | faithful | 87 | 47 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| qwen:faith_v0 | faithful | 166 | 80 | 0.70 | 0.80 | 0.75 | 0.48 [0.35, 0.61] | 0.74 | 0 |
| qwen:faith_v1 | faithful | 166 | 80 | 0.93 | 0.99 | 0.96 | 0.92 [0.85, 0.96] | 0.96 | 0 |
| qwen:faith_v2 | faithful | 166 | 80 | 0.95 | 0.79 | 0.86 | 0.76 [0.66, 0.85] | 0.88 | 3 |
| qwen:faith_v3 | faithful | 166 | 80 | 0.99 | 0.97 | 0.98 | 0.96 [0.90, 1.00] | 0.98 | 0 |
| ragas[qwen]:faithfulness==1 | faithful | 85 | 40 | 0.71 | 1.00 | 0.83 | 0.63 [0.48, 0.78] | 0.81 | 0 |

### Flag rate by planted type (fraction of items the judge marks as DEFECT) - split=all

| judge | clean (n=22) | numeral-var (n=10) | wrong# (n=18) | wrong#-AR-digits (n=12) | wrong-ent (n=18) | halluc+ (n=18) | omitted (n=10) | wrong-lang (n=14) | wrong-cite (n=16) | ok-abstain (n=14) | bad-abstain (n=14) | halluc-unans (n=14) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma:overall_pipeline_v3 [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| gemma:overall_v0 [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 0.94 | 0.83 | 1.00 | 0.64 | 0.62 | 0.00 | 1.00 | 1.00 |
| qwen:overall_pipeline [acceptable] | 0.09 | 0.00 | 1.00 | 1.00 | 1.00 | 0.72 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| qwen:overall_pipeline_v3 [acceptable] | 0.05 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| qwen:overall_v0 [acceptable] | 0.00 | 0.00 | 0.89 | 0.92 | 1.00 | 0.11 | 0.00 | 0.00 | 0.19 | 0.07 | 0.00 | 1.00 |
| gemma:cite_v1 [attributed] | 0.00 | 0.00 | 1.00 | 1.00 | 0.90 | 0.93 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| qwen:cite_v1 [attributed] | 0.00 | 0.00 | 0.94 | 1.00 | 0.94 | 0.22 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| gemma:correct_v0 [correct] | 0.00 | 0.00 | 0.83 | 0.92 | 0.94 | 0.00 | 0.60 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| gemma:correct_v1 [correct] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| qwen:correct_v0 [correct] | 0.18 | 0.10 | 0.83 | 0.92 | 1.00 | 0.39 | 0.40 | 0.07 | 0.12 | 0.00 | 0.64 | 0.93 |
| qwen:correct_v1 [correct] | 0.05 | 0.00 | 1.00 | 1.00 | 1.00 | 0.44 | 1.00 | 0.07 | 0.00 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-f1>=0.10 [correct] | 0.14 | 0.12 | 0.38 | 0.38 | 0.88 | 0.12 | 0.17 | 0.25 | 0.12 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-precision>=0.10 [correct] | 0.14 | 0.12 | 0.38 | 0.38 | 0.88 | 0.12 | 0.17 | 0.12 | 0.12 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-recall>=0.10 [correct] | 0.14 | 0.12 | 0.38 | 0.38 | 0.88 | 0.12 | 0.17 | 0.12 | 0.12 | 0.00 | 1.00 | 1.00 |
| string_match:contains_ref [correct] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.00 | 0.00 | 1.00 | 1.00 |
| gemma:faith_v0 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 | 0.00 | 1.00 | 0.00 | 0.07 | 1.00 |
| gemma:faith_v3 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| qwen:faith_v0 [faithful] | 0.09 | 0.10 | 1.00 | 1.00 | 1.00 | 0.22 | 0.10 | 0.07 | 0.50 | 1.00 | 0.64 | 0.86 |
| qwen:faith_v1 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.10 | 0.07 | 0.25 | 0.00 | 0.64 | 1.00 |
| qwen:faith_v2 [faithful] | 0.05 | 0.00 | 1.00 | 1.00 | 0.72 | 0.33 | 0.00 | 0.14 | 0.00 | 0.00 | 0.00 | 1.00 |
| qwen:faith_v3 [faithful] | 0.00 | 0.00 | 0.94 | 0.92 | 1.00 | 1.00 | 0.00 | 0.07 | 0.00 | 0.00 | 0.00 | 1.00 |
| ragas[qwen]:faithfulness==1 [faithful] | 0.14 | 0.38 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.12 | 0.12 | 1.00 | 1.00 | 1.00 |

Defect types the judge is *meant* to catch should be near 1.00; clean/ok types should be near 0.00.

#### Ablation of `gemma:overall_pipeline_v3` (axis acceptable, split=all)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.45 | 1.00 | 0.62 | 0.29 |
| + language check | 0.48 | 1.00 | 0.65 | 0.32 |
| + abstention check | 0.45 | 1.00 | 0.62 | 0.29 |
| full pipeline (+both deterministic checks) | 0.48 | 1.00 | 0.65 | 0.32 |
| deterministic checks only (language+abstention) | 0.09 | 1.00 | 0.16 | 0.05 |

#### Ablation of `qwen:overall_pipeline` (axis acceptable, split=all)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.87 | 0.98 | 0.92 | 0.75 |
| + language check | 0.96 | 0.98 | 0.97 | 0.90 |
| + abstention check | 0.87 | 0.98 | 0.92 | 0.75 |
| full pipeline (+both deterministic checks) | 0.96 | 0.98 | 0.97 | 0.90 |
| deterministic checks only (language+abstention) | 0.31 | 1.00 | 0.48 | 0.19 |

#### Ablation of `qwen:overall_pipeline_v3` (axis acceptable, split=all)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.91 | 0.99 | 0.95 | 0.82 |
| + language check | 1.00 | 0.99 | 1.00 | 0.99 |
| + abstention check | 0.91 | 0.99 | 0.95 | 0.82 |
| full pipeline (+both deterministic checks) | 1.00 | 0.99 | 1.00 | 0.99 |
| deterministic checks only (language+abstention) | 0.31 | 1.00 | 0.48 | 0.19 |

#### Per question-language (split=all): kappa / defect recall / false-positive rate on clean-type items

| judge | en: n | en kappa | en recall | en FPR | ar: n | ar kappa | ar recall | ar FPR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma:cite_v1 | 40 | 1.00 | 1.00 | 0.00 | 51 | 0.92 | 0.94 | 0.00 |
| gemma:correct_v0 | 91 | 0.89 | 0.88 | 0.00 | 89 | 0.91 | 0.91 | 0.00 |
| gemma:correct_v1 | 40 | 1.00 | 1.00 | 0.00 | 51 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v0 | 84 | 0.72 | 1.00 | 0.00 | 82 | 0.73 | 1.00 | 0.00 |
| gemma:faith_v3 | 39 | 1.00 | 1.00 | 0.00 | 48 | 1.00 | 1.00 | 0.00 |
| gemma:overall_pipeline_v3 | 40 | 1.00 | 1.00 | 0.00 | 51 | 1.00 | 1.00 | 0.00 |
| gemma:overall_v0 | 91 | 0.83 | 0.89 | 0.00 | 89 | 0.77 | 0.88 | 0.00 |
| qwen:cite_v1 | 91 | 0.82 | 0.82 | 0.00 | 89 | 0.82 | 0.84 | 0.00 |
| qwen:correct_v0 | 91 | 0.60 | 0.74 | 0.15 | 89 | 0.71 | 0.89 | 0.05 |
| qwen:correct_v1 | 91 | 0.89 | 1.00 | 0.04 | 89 | 0.89 | 1.00 | 0.00 |
| qwen:faith_v0 | 84 | 0.57 | 0.81 | 0.31 | 82 | 0.38 | 0.79 | 0.45 |
| qwen:faith_v1 | 84 | 0.95 | 0.97 | 0.00 | 82 | 0.88 | 1.00 | 0.00 |
| qwen:faith_v2 | 84 | 0.73 | 0.73 | 0.04 | 82 | 0.78 | 0.84 | 0.00 |
| qwen:faith_v3 | 84 | 1.00 | 1.00 | 0.00 | 82 | 0.93 | 0.95 | 0.00 |
| qwen:overall_pipeline | 91 | 0.89 | 0.97 | 0.08 | 89 | 0.91 | 0.96 | 0.00 |
| qwen:overall_pipeline_v3 | 91 | 0.97 | 1.00 | 0.04 | 89 | 1.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 91 | 0.28 | 0.43 | 0.04 | 89 | 0.33 | 0.52 | 0.00 |

#### Number errors: defect recall by subtype and digit script, and false-positive rate on correct numeral variants (split=all; n shown for judges that scored every item)

| judge | near (+-small) (n=14) | scale (x10) (n=2) | far (n=14) | ASCII digits (n=18) | Arabic-Indic digits (n=12) | FPR on correct numeral variants (n=10) |
|---|--:|--:|--:|--:|--:|--:|
| gemma:correct_v0 | 0.79 | 1.00 | 0.93 | 0.83 | 0.92 | 0.00 |
| gemma:correct_v1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:overall_pipeline_v3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:overall_v0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:correct_v0 | 0.71 | 1.00 | 1.00 | 0.83 | 0.92 | 0.10 |
| qwen:correct_v1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.10 |
| qwen:faith_v1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v3 | 1.00 | 1.00 | 0.86 | 0.94 | 0.92 | 0.00 |
| qwen:overall_pipeline | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_pipeline_v3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 0.93 | 1.00 | 0.86 | 0.89 | 0.92 | 0.00 |
| ragas[qwen]:FC-f1>=0.10 | 0.14 | 0.00 | 0.62 | 0.38 | 0.38 | 0.12 |
| ragas[qwen]:FC-precision>=0.10 | 0.14 | 0.00 | 0.62 | 0.38 | 0.38 | 0.12 |
| ragas[qwen]:FC-recall>=0.10 | 0.14 | 0.00 | 0.62 | 0.38 | 0.38 | 0.12 |
| ragas[qwen]:faithfulness==1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 |
| string_match:contains_ref | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

#### Question language != evidence language (cross-lingual) vs same language: defect recall / clean FPR (split=all)

| judge | cross: recall | cross: FPR | mono: recall | mono: FPR |
|---|--:|--:|--:|--:|
| gemma:cite_v1 | 1.00 | 0.00 | 0.94 | 0.00 |
| gemma:correct_v0 | 0.88 | 0.00 | 0.91 | 0.00 |
| gemma:correct_v1 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:faith_v0 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:faith_v3 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:overall_pipeline_v3 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:overall_v0 | 0.87 | 0.00 | 0.90 | 0.00 |
| qwen:cite_v1 | 0.79 | 0.00 | 0.86 | 0.00 |
| qwen:correct_v0 | 0.88 | 0.14 | 0.78 | 0.08 |
| qwen:correct_v1 | 1.00 | 0.00 | 1.00 | 0.04 |
| qwen:faith_v0 | 0.68 | 0.38 | 0.88 | 0.36 |
| qwen:faith_v1 | 1.00 | 0.00 | 0.98 | 0.00 |
| qwen:faith_v2 | 0.74 | 0.05 | 0.82 | 0.00 |
| qwen:faith_v3 | 1.00 | 0.00 | 0.96 | 0.00 |
| qwen:overall_pipeline | 0.95 | 0.05 | 0.97 | 0.04 |
| qwen:overall_pipeline_v3 | 1.00 | 0.00 | 1.00 | 0.04 |
| qwen:overall_v0 | 0.42 | 0.05 | 0.52 | 0.00 |
| ragas[qwen]:FC-f1>=0.10 | 0.71 | 0.08 | 0.60 | 0.09 |
| ragas[qwen]:FC-precision>=0.10 | 0.71 | 0.08 | 0.60 | 0.09 |
| ragas[qwen]:FC-recall>=0.10 | 0.71 | 0.08 | 0.60 | 0.09 |
| ragas[qwen]:faithfulness==1 | 1.00 | 0.58 | 1.00 | 0.45 |

#### Judge-vs-judge agreement (Cohen's kappa between the verdicts themselves, on the items both judges scored, split=all)

| judge | n | gemma vs qwen: kappa | raw agreement |
|---|--:|--:|--:|
| cite_v1 | 91 | 0.78 | 0.89 |
| correct_v0 | 180 | 0.64 | 0.82 |
| correct_v1 | 91 | 0.85 | 0.92 |
| faith_v0 | 180 | 0.34 | 0.68 |
| faith_v3 | 91 | 0.93 | 0.97 |
| overall_pipeline_v3 | 91 | 0.97 | 0.99 |
| overall_v0 | 180 | 0.39 | 0.67 |

#### Judge improvements: kappa before -> after, with paired cluster-bootstrap 95% CI of the difference (split=all)

| axis | before | after | n | kappa before | kappa after | diff [95% CI] |
|---|---|---|--:|--:|--:|---|
| acceptable | gemma:overall_v0 | gemma:overall_pipeline_v3 | 91 | 0.95 | 1.00 | +0.05 [+0.00, +0.12] |
| correct | gemma:correct_v0 | gemma:correct_v1 | 91 | 0.87 | 1.00 | +0.13 [+0.04, +0.27] |
| faithful | gemma:faith_v0 | gemma:faith_v3 | 87 | 0.81 | 1.00 | +0.19 [+0.08, +0.30] |
| acceptable | qwen:overall_v0 | qwen:overall_pipeline_v3 | 180 | 0.31 | 0.99 | +0.68 [+0.57, +0.77] |
| correct | qwen:correct_v0 | qwen:correct_v1 | 180 | 0.65 | 0.89 | +0.23 [+0.13, +0.35] |
| faithful | qwen:faith_v0 | qwen:faith_v1 | 166 | 0.48 | 0.92 | +0.43 [+0.31, +0.56] |
| faithful | qwen:faith_v0 | qwen:faith_v3 | 166 | 0.48 | 0.96 | +0.48 [+0.34, +0.63] |
| faithful | qwen:faith_v2 | qwen:faith_v3 | 166 | 0.76 | 0.96 | +0.21 [+0.10, +0.33] |
| acceptable | qwen:overall_pipeline | qwen:overall_pipeline_v3 | 180 | 0.90 | 0.99 | +0.09 [+0.03, +0.16] |

Ragas notes: {"ragas[qwen]": {"items_scored": 96, "dev-tuned thresholds": {"FC-f1": 0.1, "FC-precision": 0.1, "FC-recall": 0.1}, "none_scores": {"faithfulness": 0, "factual_correctness": 0}}}