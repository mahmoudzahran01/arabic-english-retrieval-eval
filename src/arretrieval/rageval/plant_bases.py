"""Stage 1 of the judge-calibration set: real, verified-correct Gemma answers to NEW questions (never the 180 RAG eval
questions), generated with the gold passage forced into the context. Planted errors are then edited into these.

  python -m arretrieval.rageval.plant_bases            -> runs/rageval/data/bases.jsonl

Kept only if the answer (a) is not an abstention, (b) cites the gold passage, (c) contains the reference string
(digit/Arabic-normalised) and (d) is written in the question's language. Dropped items are counted (selection bias is
reported in RESULTS.md: surviving bases are the *easy* ones).
"""

from __future__ import annotations

import json
import random
import re

from . import retrieval as R
from .llm import get_llm
from .paths import DATA
from .prompts import messages
from .textnorm import cited_ids, contains_ref, is_abstention, lang_of, numbers

N_NUM, N_ENT, N_PAIRS = 60, 90, 20
SEED = 7


def kind_of(ref_en: str) -> str:
    w = ref_en.split()
    if re.search(r"\d", ref_en) and len(w) <= 3 and len(numbers(ref_en)) == 1:
        return "num"
    if not re.search(r"\d", ref_en) and 1 <= len(w) <= 5 and ref_en[:1].isupper():
        return "ent"
    return "other"


def pick_questions() -> list[dict]:
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(DATA / "all_qas.jsonl")]
    pool = [r for r in rows if not r["in_eval"] and r["ref_en"].strip() and r["ref_ar"].strip()]
    for r in pool:
        r["kind"] = kind_of(r["ref_en"])
    by_para: dict[int, list[dict]] = {}
    for r in pool:
        by_para.setdefault(r["para"], []).append(r)
    # (1) paired questions from the same paragraph (for the "omitted key fact" compound questions)
    cand = [ps for ps in by_para.values() if len([p for p in ps if p["kind"] in ("num", "ent")]) >= 2]
    rng.shuffle(cand)
    chosen, used = [], set()
    for k, ps in enumerate(cand[:N_PAIRS]):
        ps = [p for p in ps if p["kind"] in ("num", "ent")]
        a, b = rng.sample(ps, 2)
        lang = "en" if k % 2 == 0 else "ar"
        for role, r in (("pair_a", a), ("pair_b", b)):
            chosen.append({**r, "role": role, "pair": k, "q_lang": lang})
            used.add(r["xquad_id"])
    # (2) singles
    nums = [r for r in pool if r["kind"] == "num" and r["xquad_id"] not in used]
    ents = [r for r in pool if r["kind"] == "ent" and r["xquad_id"] not in used]
    rng.shuffle(nums)
    rng.shuffle(ents)
    for k, r in enumerate(nums[:N_NUM] + ents[:N_ENT]):
        chosen.append({**r, "role": "single", "pair": None, "q_lang": "en" if k % 2 == 0 else "ar"})
    return chosen


def main():
    llm = get_llm("gemma")
    corpus = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "corpus.jsonl")}
    qs = pick_questions()
    print(f"{len(qs)} candidate base questions", flush=True)

    def work(r):
        q = r["q_en"] if r["q_lang"] == "en" else r["q_ar"]
        ref = r["ref_en"] if r["q_lang"] == "en" else r["ref_ar"]
        hits = R.retrieve("rag_eval_xquad", q, k=10)
        others = [h for h in hits if h["id"] != r["gold_id"]][:2]
        g = corpus[r["gold_id"]]
        ctx = [{"id": g["id"], "lang": g["lang"], "text": g["text"]}] + [
            {"id": h["id"], "lang": h["lang"], "text": h["text"]} for h in others
        ]
        random.Random(f"{SEED}-{r['xquad_id']}").shuffle(ctx)

        def attempt(version: str):
            ans = llm.chat(messages(q, ctx, version, r["q_lang"]), max_tokens=300).strip()
            why = []
            if is_abstention(ans):
                why.append("abstained")
            if r["gold_id"] not in cited_ids(ans):
                why.append("no_gold_cite")
            if not contains_ref(ans, ref):
                why.append("ref_missing")
            if lang_of(re.sub(r"\[P\d{3}\]", "", ans)) != r["q_lang"]:
                why.append("wrong_lang")
            return ans, why

        ans, why = attempt("v1")
        used = "v1"
        if why == ["wrong_lang"] or "wrong_lang" in why:  # 30% of v1 answers were in the wrong language (see RESULTS):
            ans2, why2 = attempt("v3")  # retry once with the explicit-language prompt
            if not why2:
                ans, why, used = ans2, why2, "v3"
        # distractor candidates for later stages: ranked non-gold passages beyond the 2 used
        spare = [{"id": h["id"], "lang": h["lang"], "text": h["text"]} for h in hits if h["id"] != r["gold_id"]][2:6]
        return {
            "xquad_id": r["xquad_id"],
            "para": r["para"],
            "gold_id": r["gold_id"],
            "gold_lang": r["gold_lang"],
            "role": r["role"],
            "pair": r["pair"],
            "kind": r["kind"],
            "q_lang": r["q_lang"],
            "question": q,
            "reference": ref,
            "ref_sentence": r["sent_en"] if r["q_lang"] == "en" else r["sent_ar"],
            "ref_en": r["ref_en"],
            "ref_ar": r["ref_ar"],
            "contexts": ctx,
            "spare": spare,
            "answer": ans,
            "ok": not why,
            "drop_reasons": why,
            "prompt_used": used,
        }

    rows = llm.pmap(work, qs, desc="bases")
    with open(DATA / "bases.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    ok = [x for x in rows if x["ok"]]
    from collections import Counter

    print(f"kept {len(ok)}/{len(rows)}; reasons: {Counter(w for x in rows for w in x['drop_reasons'])}")


if __name__ == "__main__":
    main()
