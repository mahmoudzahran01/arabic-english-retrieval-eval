"""Before/after comparison of RAG configurations with bootstrap CIs over questions.

  python -m arretrieval.rageval.ab_test --base A_baseline --others B_rerank,C_k5,D_strict --judge-model gemma

Per-question binary outcomes (n = 180 questions, 160 answerable + 20 unanswerable):
  acceptable   success AND answered in the question's language AND judge says grounded (abstaining correctly on unanswerable counts)
  success      answerable: not abstained AND judge `correct_v1` = pass ; unanswerable: abstained
  success_str  same but correctness by reference-string match (non-LLM cross-check)
  grounded     (non-abstained answers) judge `faith_v1`/`faith_v2` = pass ; abstentions count as grounded
  gold_hit     gold passage in the retrieved set (answerable only)
  over_refusal answerable question answered with "not in the documents"
Paired bootstrap over questions gives the CI for (X - baseline); exact McNemar gives a p-value on the discordant pairs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .paths import RUNS
from .stats import bootstrap_ci, mcnemar_exact, min_detectable_effect, paired_bootstrap
from .textnorm import contains_ref


def load_rows(cfg):
    return {r["qid"]: r for r in (json.loads(l) for l in open(RUNS / f"answers_{cfg}.jsonl"))}


def load_j(cfg, model, judge):
    p = RUNS / f"judge_rag_{cfg}_{model}" / f"{judge}.jsonl"
    return {r["id"]: r["pass"] for r in (json.loads(l) for l in open(p))} if p.exists() else None


def outcomes(cfg, model, correct_judge, faith_judge, faith_model=None):
    rows = load_rows(cfg)
    cj, fj = load_j(cfg, model, correct_judge), load_j(cfg, faith_model or model, faith_judge)
    out = {}
    for q, r in rows.items():
        ab = r["abstained"]
        if r["answerable"]:
            succ = None if (cj is None or (not ab and cj.get(q) is None)) else int((not ab) and cj[q] == 1)
            sstr = int((not ab) and contains_ref(r["answer"], r["reference"]))
            over = int(ab)
        else:
            succ, sstr, over = int(ab), int(ab), None
        grounded = None if (fj is None or (not ab and fj.get(q) is None)) else int(ab or fj[q] == 1)
        lang_ok = int(r["answer_lang"] == r["q_lang"]) if not ab else None
        acc = (
            None
            if (succ is None or grounded is None)
            else int(bool(succ) and (ab and not r["answerable"] or (lang_ok == 1 and grounded == 1)))
        )
        out[q] = {
            "acceptable": acc,
            "success": succ,
            "success_str": sstr,
            "grounded": grounded,
            "over_refusal": over,
            "gold_hit": int(bool(r["gold_retrieved"])) if r["answerable"] else None,
            "answerable": r["answerable"],
            "q_lang": r["q_lang"],
            "kind": r["kind"],
            "lang_ok": lang_ok,
        }
    return out


def ci_str(v):
    v = [x for x in v if x is not None]
    p, lo, hi = bootstrap_ci(np.array(v, float), n_boot=4000)
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={len(v)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="A_baseline")
    ap.add_argument("--others", default="B_rerank")
    ap.add_argument("--judge-model", default="gemma")
    ap.add_argument("--correct-judge", default="correct_v1")
    ap.add_argument("--faith-judge", default="faith_v1")
    ap.add_argument("--faith-model", default=None, help="model for the grounding/attribution judge if different from --judge-model")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    cfgs = [a.base] + a.others.split(",")
    O = {c: outcomes(c, a.judge_model, a.correct_judge, a.faith_judge, a.faith_model) for c in cfgs}
    qids = sorted(O[a.base])
    lines = [
        f"### Config comparison (correctness judge: {a.judge_model}:{a.correct_judge}; grounding judge: {a.faith_model or a.judge_model}:{a.faith_judge}); 95% bootstrap CI over questions",
        "",
        "| config | acceptable (all) | success (all) | success (answerable) | success_str | grounded | gold_hit | over-refusal | unans. abstain | right-language (of answered) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cfgs:
        o = O[c]
        g = lambda k, f=lambda x: True: [o[q][k] for q in qids if o[q][k] is not None and f(o[q])]
        lines.append(
            f"| {c} | {ci_str(g('acceptable'))} | {ci_str(g('success'))} | {ci_str(g('success', lambda x: x['answerable']))} | {ci_str(g('success_str'))} | "
            f"{ci_str(g('grounded'))} | {ci_str(g('gold_hit'))} | {ci_str(g('over_refusal'))} | "
            f"{ci_str(g('success', lambda x: not x['answerable']))} | {ci_str(g('lang_ok'))} |"
        )
    lines += [
        "",
        f"### Paired differences vs {a.base} (positive = better; over_refusal: negative = better)",
        "",
        "| config | metric | diff [95% CI] | bootstrap p | McNemar p (b vs c) | n paired |",
        "|---|---|---|---|---|---|",
    ]
    for c in cfgs[1:]:
        for k in ("acceptable", "success", "success_str", "gold_hit", "grounded", "lang_ok", "over_refusal"):
            ids = [q for q in qids if O[a.base][q][k] is not None and O[c][q][k] is not None]
            if not ids:
                continue
            x = np.array([O[a.base][q][k] for q in ids], float)
            y = np.array([O[c][q][k] for q in ids], float)
            pb = paired_bootstrap(x, y, n_boot=10000)
            mc = mcnemar_exact(x, y)
            lines.append(
                f"| {c} | {k} | {100 * pb['diff']:+.1f} pts [{100 * pb['lo']:+.1f}, {100 * pb['hi']:+.1f}] | {pb['p']:.4f} | "
                f"{mc['p']:.4f} ({mc['n01']} vs {mc['n10']}) | {len(ids)} |"
            )
    mde = min_detectable_effect(len(qids), p=0.7, rho=0.5)
    lines += [
        "",
        f"Rough minimum detectable paired difference at n={len(qids)}, 80% power, base rate 0.7, within-question correlation 0.5: "
        f"{100 * mde:.1f} points.",
    ]
    text = "\n".join(lines)
    print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
