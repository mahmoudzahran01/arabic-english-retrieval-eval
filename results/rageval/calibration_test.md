### Judge calibration vs planted ground truth - split=test (n_items=91)

| judge | axis | n | defects | precision | recall | F1 | kappa [95% CI] | acc | unparsable |
|---|---|--:|--:|--:|--:|--:|---|--:|--:|
| gemma:overall_pipeline_v3 | acceptable | 91 | 64 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| gemma:overall_v0 | acceptable | 91 | 64 | 1.00 | 0.97 | 0.98 | 0.95 [0.88, 1.00] | 0.98 | 0 |
| qwen:overall_pipeline | acceptable | 91 | 64 | 0.97 | 0.94 | 0.95 | 0.85 [0.71, 0.95] | 0.93 | 2 |
| qwen:overall_pipeline_v3 | acceptable | 91 | 64 | 0.98 | 1.00 | 0.99 | 0.97 [0.91, 1.00] | 0.99 | 0 |
| qwen:overall_v0 | acceptable | 91 | 64 | 1.00 | 0.55 | 0.71 | 0.42 [0.28, 0.58] | 0.68 | 0 |
| gemma:cite_v1 | attributed | 91 | 52 | 1.00 | 0.96 | 0.98 | 0.96 [0.88, 1.00] | 0.98 | 0 |
| qwen:cite_v1 | attributed | 91 | 52 | 1.00 | 0.81 | 0.89 | 0.78 [0.63, 0.90] | 0.89 | 0 |
| gemma:correct_v0 | correct | 91 | 41 | 1.00 | 0.85 | 0.92 | 0.87 [0.72, 0.97] | 0.93 | 0 |
| gemma:correct_v1 | correct | 91 | 41 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| qwen:correct_v0 | correct | 91 | 41 | 0.76 | 0.85 | 0.80 | 0.63 [0.43, 0.79] | 0.81 | 0 |
| qwen:correct_v1 | correct | 91 | 41 | 0.85 | 1.00 | 0.92 | 0.85 [0.72, 0.95] | 0.92 | 0 |
| ragas[qwen]:FC-f1>=0.10 | correct | 43 | 20 | 0.83 | 0.50 | 0.62 | 0.42 [0.15, 0.71] | 0.72 | 0 |
| ragas[qwen]:FC-precision>=0.10 | correct | 43 | 20 | 0.83 | 0.50 | 0.62 | 0.42 [0.15, 0.71] | 0.72 | 0 |
| ragas[qwen]:FC-recall>=0.10 | correct | 43 | 20 | 0.83 | 0.50 | 0.62 | 0.42 [0.15, 0.71] | 0.72 | 0 |
| string_match:contains_ref | correct | 91 | 41 | 0.95 | 1.00 | 0.98 | 0.96 [0.90, 1.00] | 0.98 | 0 |
| gemma:faith_v0 | faithful | 87 | 47 | 0.85 | 1.00 | 0.92 | 0.81 [0.71, 0.91] | 0.91 | 0 |
| gemma:faith_v3 | faithful | 87 | 47 | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 0 |
| qwen:faith_v0 | faithful | 87 | 47 | 0.77 | 0.77 | 0.77 | 0.49 [0.29, 0.68] | 0.75 | 0 |
| qwen:faith_v1 | faithful | 87 | 47 | 0.96 | 1.00 | 0.98 | 0.95 [0.89, 1.00] | 0.98 | 0 |
| qwen:faith_v2 | faithful | 87 | 47 | 0.95 | 0.74 | 0.83 | 0.68 [0.52, 0.83] | 0.84 | 3 |
| qwen:faith_v3 | faithful | 87 | 47 | 0.98 | 0.96 | 0.97 | 0.93 [0.82, 1.00] | 0.97 | 0 |
| ragas[qwen]:faithfulness==1 | faithful | 40 | 19 | 0.73 | 1.00 | 0.84 | 0.66 [0.43, 0.88] | 0.82 | 0 |

### Flag rate by planted type (fraction of items the judge marks as DEFECT) - split=test

| judge | clean (n=15) | numeral-var (n=8) | wrong# (n=11) | wrong#-AR-digits (n=8) | wrong-ent (n=10) | halluc+ (n=14) | omitted (n=4) | wrong-lang (n=4) | wrong-cite (n=5) | ok-abstain (n=4) | bad-abstain (n=4) | halluc-unans (n=4) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma:overall_pipeline_v3 [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| gemma:overall_v0 [acceptable] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 0.80 | 0.00 | 1.00 | 1.00 |
| qwen:overall_pipeline [acceptable] | 0.13 | 0.00 | 1.00 | 1.00 | 1.00 | 0.71 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| qwen:overall_pipeline_v3 [acceptable] | 0.07 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| qwen:overall_v0 [acceptable] | 0.00 | 0.00 | 0.91 | 1.00 | 1.00 | 0.07 | 0.00 | 0.00 | 0.40 | 0.00 | 0.00 | 1.00 |
| gemma:cite_v1 [attributed] | 0.00 | 0.00 | 1.00 | 1.00 | 0.90 | 0.93 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| qwen:cite_v1 [attributed] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.29 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| gemma:correct_v0 [correct] | 0.00 | 0.00 | 0.82 | 0.88 | 1.00 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| gemma:correct_v1 [correct] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| qwen:correct_v0 [correct] | 0.20 | 0.12 | 0.82 | 0.88 | 1.00 | 0.43 | 0.50 | 0.00 | 0.20 | 0.00 | 0.75 | 1.00 |
| qwen:correct_v1 [correct] | 0.07 | 0.00 | 1.00 | 1.00 | 1.00 | 0.43 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-f1>=0.10 [correct] | 0.25 | 0.00 | 0.20 | 0.25 | 0.75 | 0.20 | 0.33 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-precision>=0.10 [correct] | 0.25 | 0.00 | 0.20 | 0.25 | 0.75 | 0.20 | 0.33 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| ragas[qwen]:FC-recall>=0.10 [correct] | 0.25 | 0.00 | 0.20 | 0.25 | 0.75 | 0.20 | 0.33 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| string_match:contains_ref [correct] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.00 | 0.00 | 1.00 | 1.00 |
| gemma:faith_v0 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| gemma:faith_v3 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| qwen:faith_v0 [faithful] | 0.13 | 0.12 | 1.00 | 1.00 | 1.00 | 0.21 | 0.00 | 0.00 | 0.80 | 1.00 | 0.75 | 1.00 |
| qwen:faith_v1 [faithful] | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.00 | 0.20 | 0.00 | 1.00 | 1.00 |
| qwen:faith_v2 [faithful] | 0.07 | 0.00 | 1.00 | 1.00 | 0.80 | 0.29 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 1.00 |
| qwen:faith_v3 [faithful] | 0.00 | 0.00 | 0.91 | 0.88 | 1.00 | 1.00 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 1.00 |
| ragas[qwen]:faithfulness==1 [faithful] | 0.25 | 0.17 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 |

Defect types the judge is *meant* to catch should be near 1.00; clean/ok types should be near 0.00.

#### Ablation of `gemma:overall_pipeline_v3` (axis acceptable, split=test)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.94 | 1.00 | 0.97 | 0.90 |
| + language check | 1.00 | 1.00 | 1.00 | 1.00 |
| + abstention check | 0.94 | 1.00 | 0.97 | 0.90 |
| full pipeline (+both deterministic checks) | 1.00 | 1.00 | 1.00 | 1.00 |
| deterministic checks only (language+abstention) | 0.19 | 1.00 | 0.32 | 0.12 |

#### Ablation of `qwen:overall_pipeline` (axis acceptable, split=test)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.89 | 0.97 | 0.93 | 0.78 |
| + language check | 0.94 | 0.97 | 0.95 | 0.85 |
| + abstention check | 0.89 | 0.97 | 0.93 | 0.78 |
| full pipeline (+both deterministic checks) | 0.94 | 0.97 | 0.95 | 0.85 |
| deterministic checks only (language+abstention) | 0.19 | 1.00 | 0.32 | 0.12 |

#### Ablation of `qwen:overall_pipeline_v3` (axis acceptable, split=test)

| components | recall of defects | precision | F1 | kappa |
|---|--:|--:|--:|--:|
| LLM parts only (faithful+correct+citation) | 0.95 | 0.98 | 0.97 | 0.90 |
| + language check | 1.00 | 0.98 | 0.99 | 0.97 |
| + abstention check | 0.95 | 0.98 | 0.97 | 0.90 |
| full pipeline (+both deterministic checks) | 1.00 | 0.98 | 0.99 | 0.97 |
| deterministic checks only (language+abstention) | 0.19 | 1.00 | 0.32 | 0.12 |

#### Per question-language (split=test): kappa / defect recall / false-positive rate on clean-type items

| judge | en: n | en kappa | en recall | en FPR | ar: n | ar kappa | ar recall | ar FPR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma:cite_v1 | 40 | 1.00 | 1.00 | 0.00 | 51 | 0.92 | 0.94 | 0.00 |
| gemma:correct_v0 | 40 | 0.84 | 0.82 | 0.00 | 51 | 0.88 | 0.88 | 0.00 |
| gemma:correct_v1 | 40 | 1.00 | 1.00 | 0.00 | 51 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v0 | 39 | 0.90 | 1.00 | 0.00 | 48 | 0.74 | 1.00 | 0.00 |
| gemma:faith_v3 | 39 | 1.00 | 1.00 | 0.00 | 48 | 1.00 | 1.00 | 0.00 |
| gemma:overall_pipeline_v3 | 40 | 1.00 | 1.00 | 0.00 | 51 | 1.00 | 1.00 | 0.00 |
| gemma:overall_v0 | 40 | 1.00 | 1.00 | 0.00 | 51 | 0.90 | 0.95 | 0.00 |
| qwen:cite_v1 | 40 | 0.75 | 0.76 | 0.00 | 51 | 0.80 | 0.84 | 0.00 |
| qwen:correct_v0 | 40 | 0.54 | 0.76 | 0.20 | 51 | 0.69 | 0.92 | 0.08 |
| qwen:correct_v1 | 40 | 0.80 | 1.00 | 0.07 | 51 | 0.88 | 1.00 | 0.00 |
| qwen:faith_v0 | 39 | 0.64 | 0.80 | 0.13 | 48 | 0.36 | 0.74 | 0.42 |
| qwen:faith_v1 | 39 | 1.00 | 1.00 | 0.00 | 48 | 0.91 | 1.00 | 0.00 |
| qwen:faith_v2 | 39 | 0.64 | 0.70 | 0.07 | 48 | 0.71 | 0.78 | 0.00 |
| qwen:faith_v3 | 39 | 1.00 | 1.00 | 0.00 | 48 | 0.87 | 0.93 | 0.00 |
| qwen:overall_pipeline | 40 | 0.79 | 0.92 | 0.14 | 51 | 0.90 | 0.95 | 0.00 |
| qwen:overall_pipeline_v3 | 40 | 0.95 | 1.00 | 0.07 | 51 | 1.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 40 | 0.49 | 0.56 | 0.00 | 51 | 0.35 | 0.54 | 0.00 |

#### Number errors: defect recall by subtype and digit script, and false-positive rate on correct numeral variants (split=test; n shown for judges that scored every item)

| judge | near (+-small) (n=11) | scale (x10) (n=2) | far (n=6) | ASCII digits (n=11) | Arabic-Indic digits (n=8) | FPR on correct numeral variants (n=8) |
|---|--:|--:|--:|--:|--:|--:|
| gemma:correct_v0 | 0.73 | 1.00 | 1.00 | 0.82 | 0.88 | 0.00 |
| gemma:correct_v1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:faith_v3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:overall_pipeline_v3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| gemma:overall_v0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:correct_v0 | 0.73 | 1.00 | 1.00 | 0.82 | 0.88 | 0.12 |
| qwen:correct_v1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.12 |
| qwen:faith_v1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:faith_v3 | 1.00 | 1.00 | 0.67 | 0.91 | 0.88 | 0.00 |
| qwen:overall_pipeline | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_pipeline_v3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| qwen:overall_v0 | 0.91 | 1.00 | 1.00 | 0.91 | 1.00 | 0.00 |
| ragas[qwen]:FC-f1>=0.10 | 0.17 | 0.00 | 0.50 | 0.20 | 0.25 | 0.00 |
| ragas[qwen]:FC-precision>=0.10 | 0.17 | 0.00 | 0.50 | 0.20 | 0.25 | 0.00 |
| ragas[qwen]:FC-recall>=0.10 | 0.17 | 0.00 | 0.50 | 0.20 | 0.25 | 0.00 |
| ragas[qwen]:faithfulness==1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 |
| string_match:contains_ref | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

#### Question language != evidence language (cross-lingual) vs same language: defect recall / clean FPR (split=test)

| judge | cross: recall | cross: FPR | mono: recall | mono: FPR |
|---|--:|--:|--:|--:|
| gemma:cite_v1 | 1.00 | 0.00 | 0.94 | 0.00 |
| gemma:correct_v0 | 0.75 | 0.00 | 0.88 | 0.00 |
| gemma:correct_v1 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:faith_v0 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:faith_v3 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:overall_pipeline_v3 | 1.00 | 0.00 | 1.00 | 0.00 |
| gemma:overall_v0 | 1.00 | 0.00 | 0.96 | 0.00 |
| qwen:cite_v1 | 0.69 | 0.00 | 0.86 | 0.00 |
| qwen:correct_v0 | 1.00 | 0.25 | 0.82 | 0.11 |
| qwen:correct_v1 | 1.00 | 0.00 | 1.00 | 0.05 |
| qwen:faith_v0 | 0.57 | 0.25 | 0.85 | 0.26 |
| qwen:faith_v1 | 1.00 | 0.00 | 1.00 | 0.00 |
| qwen:faith_v2 | 0.64 | 0.12 | 0.79 | 0.00 |
| qwen:faith_v3 | 1.00 | 0.00 | 0.94 | 0.00 |
| qwen:overall_pipeline | 0.89 | 0.12 | 0.96 | 0.06 |
| qwen:overall_pipeline_v3 | 1.00 | 0.00 | 1.00 | 0.05 |
| qwen:overall_v0 | 0.44 | 0.00 | 0.59 | 0.00 |
| ragas[qwen]:FC-f1>=0.10 | 0.40 | 0.00 | 0.53 | 0.10 |
| ragas[qwen]:FC-precision>=0.10 | 0.40 | 0.00 | 0.53 | 0.10 |
| ragas[qwen]:FC-recall>=0.10 | 0.40 | 0.00 | 0.53 | 0.10 |
| ragas[qwen]:faithfulness==1 | 1.00 | 0.33 | 1.00 | 0.40 |

#### Judge-vs-judge agreement (Cohen's kappa between the verdicts themselves, on the items both judges scored, split=test)

| judge | n | gemma vs qwen: kappa | raw agreement |
|---|--:|--:|--:|
| cite_v1 | 91 | 0.78 | 0.89 |
| correct_v0 | 91 | 0.63 | 0.81 |
| correct_v1 | 91 | 0.85 | 0.92 |
| faith_v0 | 91 | 0.44 | 0.73 |
| faith_v3 | 91 | 0.93 | 0.97 |
| overall_pipeline_v3 | 91 | 0.97 | 0.99 |
| overall_v0 | 91 | 0.45 | 0.70 |

#### Judge improvements: kappa before -> after, with paired cluster-bootstrap 95% CI of the difference (split=test)

| axis | before | after | n | kappa before | kappa after | diff [95% CI] |
|---|---|---|--:|--:|--:|---|
| acceptable | gemma:overall_v0 | gemma:overall_pipeline_v3 | 91 | 0.95 | 1.00 | +0.05 [+0.00, +0.12] |
| correct | gemma:correct_v0 | gemma:correct_v1 | 91 | 0.87 | 1.00 | +0.13 [+0.04, +0.27] |
| faithful | gemma:faith_v0 | gemma:faith_v3 | 87 | 0.81 | 1.00 | +0.19 [+0.08, +0.30] |
| acceptable | qwen:overall_v0 | qwen:overall_pipeline_v3 | 91 | 0.42 | 0.97 | +0.56 [+0.37, +0.71] |
| correct | qwen:correct_v0 | qwen:correct_v1 | 91 | 0.63 | 0.85 | +0.22 [+0.06, +0.39] |
| faithful | qwen:faith_v0 | qwen:faith_v1 | 87 | 0.49 | 0.95 | +0.46 [+0.28, +0.66] |
| faithful | qwen:faith_v0 | qwen:faith_v3 | 87 | 0.49 | 0.93 | +0.44 [+0.23, +0.65] |
| faithful | qwen:faith_v2 | qwen:faith_v3 | 87 | 0.68 | 0.93 | +0.25 [+0.07, +0.45] |
| acceptable | qwen:overall_pipeline | qwen:overall_pipeline_v3 | 91 | 0.85 | 0.97 | +0.13 [+0.03, +0.26] |

Ragas notes: {"ragas[qwen]": {"items_scored": 96, "dev-tuned thresholds": {"FC-f1": 0.1, "FC-precision": 0.1, "FC-recall": 0.1}, "none_scores": {"faithfulness": 0, "factual_correctness": 0}}}