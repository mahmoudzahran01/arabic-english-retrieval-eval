"""Pairwise-judge bias checks: position, verbosity, self-preference. Every pair is judged in BOTH orders.

  python -m arretrieval.rageval.bias_checks --model gemma     # and --model qwen
Writes runs/rageval/bias_<model>.json

Suites (all built from the verified-correct `bases` so ground truth is known):
  quality    correct answer vs the same answer with a wrong number          -> should always pick the correct one
  equal      correct answer vs the same answer with numerals in another script (both correct) -> pure position bias
  pad_equal  short correct vs the same correct answer + content-free fluff    -> verbosity bias (should be 50%)
  pad_wrong  short correct vs wrong-number answer + fluff                    -> does length rescue a wrong answer?
  self       Gemma-written vs Qwen-written correct answers (both verified)   -> self-preference per judge
"""

from __future__ import annotations

import argparse
import json
import random
import re

from .llm import get_llm
from .paths import DATA, RUNS
from .plant_items import plant_numeral_variant, plant_wrong_number
from .prompts import messages
from .textnorm import cited_ids, contains_ref, is_abstention, lang_of

FLUFF = {
    "en": " This information comes directly from the provided documents, which describe the topic in detail and give helpful context.",
    "ar": " هذه المعلومة مأخوذة مباشرة من المستندات المقدمة التي تصف الموضوع بالتفصيل وتوفر سياقًا مفيدًا.",
}

PAIR_P = """You are comparing two answers to the same question.
Question: {q}
Reference answer: {ref}

Answer A: {a}

Answer B: {b}

Which answer is better? Prefer the answer that agrees with the Reference answer; if both agree, prefer the clearer one. Reply with exactly one letter: A or B."""


def judge_pair(llm, q, ref, a, b) -> str | None:
    t = llm.chat(PAIR_P.format(q=q, ref=ref, a=a, b=b), max_tokens=4).strip().upper()
    return t[:1] if t[:1] in ("A", "B") else None


def both_orders(llm, base, x, y):
    """Return (picked_x_when_x_first, picked_x_when_x_second)."""
    r1 = judge_pair(llm, base["question"], base["reference"], x, y)
    r2 = judge_pair(llm, base["question"], base["reference"], y, x)
    return (None if r1 is None else r1 == "A"), (None if r2 is None else r2 == "B")


def rate(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "rate": sum(xs) / len(xs) if xs else None}


def qwen_answers(bases):
    q = get_llm("qwen")

    def one(b):
        return q.chat(messages(b["question"], b["contexts"], "v1", b["q_lang"]), max_tokens=300).strip()

    return q.pmap(one, bases, desc="qwen answers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    llm = get_llm(a.model)
    rng = random.Random(5)
    bases = [b for b in (json.loads(l) for l in open(DATA / "bases.jsonl")) if b["ok"] and b["role"] == "single"]
    nums = [b for b in bases if b["kind"] == "num"]
    rng.shuffle(nums)
    out = {"model": a.model}

    # --- quality / equal / padding suites (numeric bases) ------------------------------------------------
    pq, pe, pf, pw = [], [], [], []
    for b in nums:
        w = plant_wrong_number(b, rng)
        v = plant_numeral_variant(b, rng)
        if w and len(pq) < a.n:
            pq.append((b, b["answer"], w[0]))
            pw.append((b, b["answer"], w[0] + FLUFF[b["q_lang"]]))
        if v and len(pe) < a.n:
            pe.append((b, b["answer"], v[0]))
        if len(pf) < a.n:
            pf.append((b, b["answer"], b["answer"] + FLUFF[b["q_lang"]]))

    def suite(pairs):
        res = llm.pmap(lambda p: both_orders(llm, *p), pairs, desc=f"{a.model} pairs")
        first = [r[0] for r in res]
        second = [r[1] for r in res]
        # position bias: P(choose position A) over both orders (0.5 = none)
        pos_a = [(r[0]) for r in res] + [(None if r[1] is None else (not r[1])) for r in res]
        consistent = [None if (r[0] is None or r[1] is None) else (r[0] == r[1]) for r in res]
        return {
            "x_first": rate(first),
            "x_second": rate(second),
            "x_overall": rate(first + second),
            "picks_position_A": rate(pos_a),
            "order_consistent": rate(consistent),
        }

    out["quality"] = suite(pq)  # x = correct answer
    out["equal"] = suite(pe)  # x = original (ASCII-digit) answer; 0.5 expected
    out["pad_equal"] = suite(pf)  # x = short answer; <0.5 means the judge prefers the padded one
    out["pad_wrong"] = suite(pw)  # x = short correct; <1 means padding rescued a wrong answer

    # --- self-preference ------------------------------------------------------------------------------------
    sb = bases[: 2 * a.n + 10]
    qans = qwen_answers(sb)
    pairs, lens = [], []
    for b, qa in zip(sb, qans):
        good = (
            (not is_abstention(qa))
            and b["gold_id"] in cited_ids(qa)
            and contains_ref(qa, b["reference"])
            and lang_of(re.sub(r"\[P\d{3}\]", "", qa)) == b["q_lang"]
            and qa.strip() != b["answer"].strip()
        )
        if good:
            pairs.append((b, b["answer"], qa))  # x = Gemma-written, y = Qwen-written (both verified correct)
            lens.append(len(qa) / max(1, len(b["answer"])))
    out["self"] = suite(pairs)
    out["self"]["n_pairs"] = len(pairs)
    out["self"]["qwen_over_gemma_length_ratio"] = sum(lens) / len(lens) if lens else None
    json.dump(out, open(RUNS / f"bias_{a.model}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
