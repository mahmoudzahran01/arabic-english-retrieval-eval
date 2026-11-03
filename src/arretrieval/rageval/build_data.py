"""Build the bilingual RAG eval set from XQuAD (ar + en, CC BY-SA 4.0).

XQuAD = the 240 SQuAD-v1.1 dev paragraphs / 1190 questions, professionally translated, so every paragraph and
question exists in BOTH languages with the same ids. We exploit that to get cross-lingual cases for free:

  * each paragraph goes into the corpus in exactly ONE language (random, 50/50) -> a question in language X can
    have its evidence in language X (mono-lingual) or in the other language (cross-lingual);
  * 20 paragraphs are HELD OUT of the corpus (both languages) and one of their questions each becomes an
    *unanswerable* question. The held-out paragraph's neighbours (same Wikipedia article) stay in the corpus,
    so retrieval returns topically close but non-answering passages (a hard, realistic abstention test).
  * 160 answerable eval questions (1 per paragraph, 40 each of ar->ar, ar->en, en->en, en->ar);
  * every other XQuAD question on a corpus paragraph goes to a separate pool used ONLY for the judge-calibration
    (planted-error) set, so judge tuning never sees the questions the RAG system is scored on.

Needs the raw files from `python -m arretrieval.rageval.download_data`.
Outputs (runs/rageval/data/): corpus.jsonl, eval_questions.jsonl, all_qas.jsonl, heldout.jsonl, data_card.json
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

from .paths import DATA, RAW
from .textnorm import norm

SEED = 42
N_ANSWERABLE, N_UNANSWERABLE = 160, 20


def sentence_at(text: str, pos: int) -> str:
    """The sentence of `text` that contains character offset `pos` (used to expand span references into sentences)."""
    bounds, start = [], 0
    for m in re.finditer(r"(?<=[.!?؟])\s+", text):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    for a, b in bounds:
        if a <= pos < b or pos == b:
            return text[a:b].strip()
    return text.strip()


def load(lang: str) -> list[dict]:
    d = json.load(open(RAW / f"xquad.{lang}.json"))
    paras, pi = [], 0
    for a_i, art in enumerate(d["data"]):
        for p in art["paragraphs"]:
            paras.append(
                {
                    "pi": pi,
                    "article": a_i,
                    "title": art["title"],
                    "context": p["context"],
                    "qas": [
                        {
                            "id": q["id"],
                            "question": q["question"],
                            "answer": q["answers"][0]["text"],
                            "sent": sentence_at(p["context"], q["answers"][0]["answer_start"]),
                        }
                        for q in p["qas"]
                    ],
                }
            )
            pi += 1
    return paras


def main():
    en, ar = load("en"), load("ar")
    assert len(en) == len(ar) == 240
    for a, b in zip(en, ar):  # alignment check: same question ids in the same paragraph
        assert [q["id"] for q in a["qas"]] == [q["id"] for q in b["qas"]], a["pi"]
    rng = random.Random(SEED)

    # --- which paragraphs are held out (unanswerable) and which language each corpus paragraph is in -----------
    by_article: dict[int, list[int]] = {}
    for p in en:
        by_article.setdefault(p["article"], []).append(p["pi"])
    arts = [a for a, ps in by_article.items() if len(ps) >= 4]
    rng.shuffle(arts)
    held = sorted(rng.choice(by_article[a]) for a in arts[:N_UNANSWERABLE])  # <=1 per article
    kept = [i for i in range(240) if i not in held]
    langs = ["en"] * (len(kept) // 2) + ["ar"] * (len(kept) - len(kept) // 2)
    rng.shuffle(langs)
    lang_of_p = dict(zip(kept, langs))
    corpus = [
        {
            "id": f"P{i:03d}",
            "pi": i,
            "lang": lang_of_p[i],
            "article": en[i]["article"],
            "title": en[i]["title"],
            "text": (en if lang_of_p[i] == "en" else ar)[i]["context"],
        }
        for i in kept
    ]
    corpus_norm = " \n ".join(norm(c["text"]) for c in corpus)

    # --- unanswerable questions: answer string must not occur anywhere in the corpus ---------------------------------
    unans, ql = [], ["en", "ar"] * (N_UNANSWERABLE // 2)
    for h, lang in zip(held, ql):
        qas = list(zip(en[h]["qas"], ar[h]["qas"]))
        rng.shuffle(qas)
        for qe, qa_ in qas:
            ans_ok = all(len(norm(x["answer"])) >= 3 and norm(x["answer"]) not in corpus_norm for x in (qe, qa_))
            if ans_ok:
                q = qe if lang == "en" else qa_
                unans.append(
                    {
                        "qid": f"U{len(unans):02d}",
                        "xquad_id": qe["id"],
                        "para": h,
                        "q_lang": lang,
                        "question": q["question"],
                        "reference": q["answer"],
                        "ref_en": qe["answer"],
                        "ref_ar": qa_["answer"],
                        "ref_sentence": q["sent"],
                        "answerable": False,
                        "gold_id": None,
                        "gold_lang": None,
                        "kind": "unanswerable",
                    }
                )
                break
        else:
            raise RuntimeError(f"no clean unanswerable question for held-out paragraph {h}")

    # --- answerable questions: 80 from en-passages, 80 from ar-passages, half of each asked in the other language -----
    ans: list[dict] = []
    used_xids: set[str] = set()
    for pl in ("en", "ar"):
        pool = [i for i in kept if lang_of_p[i] == pl]
        rng.shuffle(pool)
        for j, pi in enumerate(pool[: N_ANSWERABLE // 2]):
            q_lang = "en" if j % 2 == 0 else "ar"  # exactly 40 mono + 40 cross per passage language
            k = rng.randrange(len(en[pi]["qas"]))
            qe, qa_ = en[pi]["qas"][k], ar[pi]["qas"][k]
            q = qe if q_lang == "en" else qa_
            used_xids.add(qe["id"])
            ans.append(
                {
                    "qid": f"Q{len(ans):03d}",
                    "xquad_id": qe["id"],
                    "para": pi,
                    "q_lang": q_lang,
                    "question": q["question"],
                    "reference": q["answer"],
                    "ref_en": qe["answer"],
                    "ref_ar": qa_["answer"],
                    "ref_sentence": q["sent"],
                    "answerable": True,
                    "gold_id": f"P{pi:03d}",
                    "gold_lang": pl,
                    "kind": "mono" if q_lang == pl else "cross",
                }
            )
    rng.shuffle(ans)
    for i, q in enumerate(ans):
        q["qid"] = f"Q{i:03d}"
    evalq = ans + unans

    # --- pool for the planted-error set: every other question on a corpus paragraph ---------------------------------
    allqa = []
    for pi in kept:
        for qe, qa_ in zip(en[pi]["qas"], ar[pi]["qas"]):
            allqa.append(
                {
                    "xquad_id": qe["id"],
                    "para": pi,
                    "gold_id": f"P{pi:03d}",
                    "gold_lang": lang_of_p[pi],
                    "q_en": qe["question"],
                    "q_ar": qa_["question"],
                    "ref_en": qe["answer"],
                    "ref_ar": qa_["answer"],
                    "sent_en": qe["sent"],
                    "sent_ar": qa_["sent"],
                    "in_eval": qe["id"] in used_xids,
                }
            )

    def dump(name, rows):
        with open(DATA / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("corpus.jsonl", corpus)
    dump("eval_questions.jsonl", evalq)
    dump("all_qas.jsonl", allqa)
    dump("heldout.jsonl", [{"pi": h, "title": en[h]["title"], "en": en[h]["context"], "ar": ar[h]["context"]} for h in held])
    card = {
        "source": "XQuAD (google-deepmind/xquad), Artetxe et al. 2020, https://arxiv.org/abs/1910.11856",
        "license": "CC BY-SA 4.0 (dataset); underlying SQuAD v1.1 text is CC BY-SA 4.0 (Wikipedia)",
        "corpus_passages": len(corpus),
        "corpus_by_lang": dict(Counter(c["lang"] for c in corpus)),
        "eval_questions": len(evalq),
        "answerable": len(ans),
        "unanswerable": len(unans),
        "by_kind": dict(Counter(f"{q['q_lang']}|{q['kind']}" for q in evalq)),
        "meta_pool_questions": sum(1 for a in allqa if not a["in_eval"]),
        "seed": SEED,
    }
    json.dump(card, open(DATA / "data_card.json", "w"), indent=2)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
