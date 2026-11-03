"""The RAG system under test: retrieve (dense | dense->rerank) -> Gemma with a grounded, citing prompt.

  python -m arretrieval.rageval.rag --config A_baseline
Writes runs/rageval/answers_<config>.jsonl  (one row per eval question; resumable via the LLM cache).
"""

from __future__ import annotations

import argparse
import json

from . import retrieval as R
from .llm import get_llm
from .paths import DATA, RUNS
from .prompts import messages
from .textnorm import cited_ids, is_abstention, lang_of

CONFIGS = {
    # name: retrieval + prompt settings
    "A_baseline": dict(collection="rag_eval_xquad", k=3, use_rerank=False, prompt="v1"),
    "B_rerank": dict(collection="rag_eval_xquad", k=3, use_rerank=True, fetch=10, prompt="v1"),
    "C_k5": dict(collection="rag_eval_xquad", k=5, use_rerank=False, prompt="v1"),
    "D_strict": dict(collection="rag_eval_xquad", k=3, use_rerank=False, prompt="v2"),
    "E_rerank_strict": dict(collection="rag_eval_xquad", k=3, use_rerank=True, fetch=10, prompt="v2"),
    "F_chunks": dict(collection="rag_eval_xquad_chunks", k=5, use_rerank=False, prompt="v1"),
    "H_lang": dict(collection="rag_eval_xquad", k=3, use_rerank=False, prompt="v3"),
    "I_rerank_lang": dict(collection="rag_eval_xquad", k=3, use_rerank=True, fetch=10, prompt="v3"),
    "G_k5_strict": dict(collection="rag_eval_xquad", k=5, use_rerank=False, prompt="v2"),
}


def retrieve_for(cfg: dict, question: str) -> list[dict]:
    hits = R.retrieve(cfg["collection"], question, k=cfg["k"], use_rerank=cfg["use_rerank"], fetch=cfg.get("fetch", 10))
    return [
        {"id": h["id"], "parent": h.get("parent", h["id"]), "lang": h["lang"], "text": h["text"], "score": h["score"], "rank": h["rank"]}
        for h in hits
    ]


def answer_one(q: dict, cfg: dict, llm) -> dict:
    passages = retrieve_for(cfg, q["question"])
    text = llm.chat(messages(q["question"], passages, cfg["prompt"], q["q_lang"]), max_tokens=300, temperature=0.0)
    text = text.strip()
    return {
        "qid": q["qid"],
        "config": None,
        "question": q["question"],
        "q_lang": q["q_lang"],
        "answerable": q["answerable"],
        "kind": q["kind"],
        "reference": q["reference"],
        "gold_id": q["gold_id"],
        "retrieved": passages,
        "answer": text,
        "abstained": is_abstention(text),
        "cited": cited_ids(text),
        "answer_lang": lang_of(text),
        "gold_retrieved": q["gold_id"] in [p["parent"] for p in passages] if q["answerable"] else None,
    }


def run(config: str, questions: list[dict] | None = None, out: str | None = None) -> list[dict]:
    cfg = CONFIGS[config]
    llm = get_llm("gemma")
    qs = questions or [json.loads(l) for l in open(DATA / "eval_questions.jsonl")]
    rows = llm.pmap(lambda q: answer_one(q, cfg, llm), qs, desc=f"rag {config}")
    for r in rows:
        r["config"] = config
    path = RUNS / (out or f"answers_{config}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    a = ap.parse_args()
    rows = run(a.config)
    ab = [r for r in rows if not r["answerable"]]
    an = [r for r in rows if r["answerable"]]
    print(
        f"{a.config}: {len(rows)} answers | gold retrieved {sum(r['gold_retrieved'] for r in an)}/{len(an)} | "
        f"abstained on answerable {sum(r['abstained'] for r in an)}/{len(an)} | "
        f"abstained on unanswerable {sum(r['abstained'] for r in ab)}/{len(ab)}"
    )
