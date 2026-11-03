"""Aggregate Ragas scores on RAG answers: per language / kind, with bootstrap CIs, plus Arabic-misbehaviour diagnostics.

python -m arretrieval.rageval.ragas_report --config A_baseline [--md out.md]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .paths import RUNS
from .stats import bootstrap_ci
from .textnorm import contains_ref

METRICS = [
    "faithfulness",
    "answer_relevancy_bge",
    "answer_relevancy_minilm",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "factual_correctness_span_ref",
    "semantic_similarity",
]


def fmt(v):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return "-"
    p, lo, hi = bootstrap_ci(np.array(v, float), n_boot=2000)
    return f"{p:.2f} [{lo:.2f},{hi:.2f}] n={len(v)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="A_baseline")
    ap.add_argument("--md", default=None)
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--n-unans", type=int, default=4)
    a = ap.parse_args()
    ans = {r["qid"]: r for r in (json.loads(l) for l in open(RUNS / f"answers_{a.config}.jsonl"))}
    rg = [json.loads(l) for l in open(RUNS / f"ragas_{a.config}.jsonl")]
    all_rows = [{**ans[r["qid"]], **r} for r in rg]
    # the stratified sample (4 per language x kind cell + unanswerable) is what the per-language table describes; the extra
    # "targeted retrieval-miss" questions (added with --add-misses) would bias those means, so they get their own rows
    from .ragas_eval import stratified_subset

    main_ids = {r["qid"] for r in stratified_subset(list(ans.values()), a.per_cell, a.n_unans)}
    rows = [r for r in all_rows if r["qid"] in main_ids]
    groups = {
        "all answerable": lambda r: r["answerable"],
        "en questions": lambda r: r["answerable"] and r["q_lang"] == "en",
        "ar questions": lambda r: r["answerable"] and r["q_lang"] == "ar",
        "en mono": lambda r: r["answerable"] and r["q_lang"] == "en" and r["kind"] == "mono",
        "en cross (ar passage)": lambda r: r["answerable"] and r["q_lang"] == "en" and r["kind"] == "cross",
        "ar mono": lambda r: r["answerable"] and r["q_lang"] == "ar" and r["kind"] == "mono",
        "ar cross (en passage)": lambda r: r["answerable"] and r["q_lang"] == "ar" and r["kind"] == "cross",
        "unanswerable (abstained)": lambda r: not r["answerable"],
        "answerable, abstained": lambda r: r["answerable"] and r["abstained"],
        "answerable, answered": lambda r: r["answerable"] and not r["abstained"],
    }
    L = [
        f"### Ragas on `{a.config}` (Gemma evaluator), mean [95% bootstrap CI]",
        "",
        "| slice | " + " | ".join(m.replace("_", " ") for m in METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for g, f in groups.items():
        sub = [r for r in rows if f(r)]
        L.append(f"| {g} (q={len(sub)}) | " + " | ".join(fmt([r.get(m) for r in sub]) for m in METRICS) + " |")
    L += [
        "",
        f"### Does Ragas notice a real retrieval failure? (all {len(all_rows)} scored rows, answerable only; includes the targeted misses)",
        "",
        "| slice | " + " | ".join(m.replace("_", " ") for m in METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    for g, f in {
        "gold passage retrieved": lambda r: r["answerable"] and r["gold_retrieved"],
        "gold passage NOT retrieved (model refused or guessed)": lambda r: r["answerable"] and not r["gold_retrieved"],
        "  of which refused": lambda r: r["answerable"] and not r["gold_retrieved"] and r["abstained"],
    }.items():
        sub = [r for r in all_rows if f(r)]
        L.append(f"| {g} (q={len(sub)}) | " + " | ".join(fmt([r.get(m) for r in sub]) for m in METRICS) + " |")
    # diagnostics
    an = [r for r in rows if not r["abstained"] and r["answerable"]]
    L += ["", "### Diagnostics", ""]
    L.append(
        f"- Correct abstentions on unanswerable questions: ragas answer_relevancy mean = "
        f"{np.mean([r['answer_relevancy_bge'] or 0 for r in rows if not r['answerable']]):.2f} (ragas forces relevancy to 0 for 'noncommittal' answers; "
        f"a correct refusal is scored as a failure), faithfulness mean = "
        f"{np.nanmean([r['faithfulness'] if r['faithfulness'] is not None else np.nan for r in rows if not r['answerable']]):.2f}."
    )
    gold = [r for r in all_rows if r["answerable"]]
    cr_hit = [r["context_recall"] for r in gold if r["gold_retrieved"] and r.get("context_recall") is not None]
    cr_miss = [r["context_recall"] for r in gold if not r["gold_retrieved"] and r.get("context_recall") is not None]
    L.append(
        f"- Context recall tracks the real retrieval outcome? mean context_recall when the gold passage WAS retrieved = "
        f"{np.mean(cr_hit):.2f} (n={len(cr_hit)}), when it was NOT = {np.mean(cr_miss) if cr_miss else float('nan'):.2f} (n={len(cr_miss)})."
    )
    sp = [
        (r["factual_correctness"], r["factual_correctness_span_ref"])
        for r in an
        if r.get("factual_correctness") is not None and r.get("factual_correctness_span_ref") is not None
    ]
    if sp:
        L.append(
            f"- FactualCorrectness with the bare XQuAD span as reference: mean {np.mean([b for _, b in sp]):.2f} vs {np.mean([x for x, _ in sp]):.2f} "
            f"with the reference expanded to its sentence (n={len(sp)} answered answerable questions; the span-reference version collapses)."
        )
    sc = [(r["factual_correctness"], contains_ref(r["answer"], r["reference"])) for r in an if r.get("factual_correctness") is not None]
    if sc:
        hit = [x for x, c in sc if c]
        miss = [x for x, c in sc if not c]
        L.append(
            f"- Ragas FC vs simple string match: mean FC when the reference string IS in the answer = {np.mean(hit):.2f} (n={len(hit)}), when it is NOT = {np.mean(miss) if miss else float('nan'):.2f} (n={len(miss)})."
        )
    wl = [r for r in an if r["answer_lang"] != r["q_lang"]]
    ok = [r for r in an if r["answer_lang"] == r["q_lang"]]
    if wl and ok:
        L.append(
            f"- Wrong-language answers (n={len(wl)}) vs right-language (n={len(ok)}): answer_relevancy(bge) {np.mean([r['answer_relevancy_bge'] or 0 for r in wl]):.2f} vs "
            f"{np.mean([r['answer_relevancy_bge'] or 0 for r in ok]):.2f}; answer_relevancy(minilm) {np.mean([r['answer_relevancy_minilm'] or 0 for r in wl]):.2f} vs "
            f"{np.mean([r['answer_relevancy_minilm'] or 0 for r in ok]):.2f}; faithfulness {np.nanmean([r['faithfulness'] if r['faithfulness'] is not None else np.nan for r in wl]):.2f} vs "
            f"{np.nanmean([r['faithfulness'] if r['faithfulness'] is not None else np.nan for r in ok]):.2f}."
        )
    errs = defaultdict(int)
    for r in rows:
        for k in r:
            if k.startswith("err_"):
                errs[k] += 1
    L.append(
        f"- Metric errors / None (NaN) per metric: errors={dict(errs)}; none-counts={ {m: sum(1 for r in rows if r.get(m) is None and (r['answerable'] or m in ('faithfulness', 'answer_relevancy_bge', 'answer_relevancy_minilm'))) for m in METRICS} }"
    )
    text = "\n".join(L)
    print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
