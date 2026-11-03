"""Confirm the A -> I improvement on FRESH questions that were never used to diagnose or tune anything.

The A/B on the 180 eval questions is in-sample: the wrong-language defect and the reranker were found by looking at those
very questions. Here we draw 48 new answerable XQuAD questions (24 en / 24 ar; not among the 180, not among the planted-set
base questions), run configs A and I unchanged, judge with the same judges, and compare.

  python -m arretrieval.rageval.heldout_ab --n 48          # builds questions, runs A and I
  then: calibrate --model qwen --answers heldout_A_baseline ... ; ab_test --base heldout_A_baseline --others heldout_I_rerank_lang
"""

from __future__ import annotations

import argparse
import json
import random

from . import rag
from .paths import DATA


def build(n: int) -> list[dict]:
    rng = random.Random(99)
    used = {json.loads(l)["xquad_id"] for l in open(DATA / "bases.jsonl")}
    rows = [json.loads(l) for l in open(DATA / "all_qas.jsonl")]
    pool = [
        r
        for r in rows
        if not r["in_eval"] and r["xquad_id"] not in used and r["ref_en"].strip() and r["ref_ar"].strip() and len(r["ref_en"].split()) <= 6
    ]
    rng.shuffle(pool)
    out = []
    for k, r in enumerate(pool[:n]):
        ql = "en" if k % 2 == 0 else "ar"
        out.append(
            {
                "qid": f"H{k:03d}",
                "xquad_id": r["xquad_id"],
                "para": r["para"],
                "q_lang": ql,
                "question": r["q_en"] if ql == "en" else r["q_ar"],
                "reference": r["ref_en"] if ql == "en" else r["ref_ar"],
                "ref_sentence": r["sent_en"] if ql == "en" else r["sent_ar"],
                "ref_en": r["ref_en"],
                "ref_ar": r["ref_ar"],
                "answerable": True,
                "gold_id": r["gold_id"],
                "gold_lang": r["gold_lang"],
                "kind": "mono" if r["gold_lang"] == ql else "cross",
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--configs", default="A_baseline,I_rerank_lang")
    a = ap.parse_args()
    qs = build(a.n)
    with open(DATA / "heldout_questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    for c in a.configs.split(","):
        rows = rag.run(c, questions=qs, out=f"answers_heldout_{c}.jsonl")
        print(
            c,
            "gold retrieved",
            sum(r["gold_retrieved"] for r in rows),
            "/",
            len(rows),
            "abstained",
            sum(r["abstained"] for r in rows),
            flush=True,
        )


if __name__ == "__main__":
    main()
