"""Ragas (0.4.x, `ragas.metrics.collections`) with an OpenAI-compatible LLM (the `gemma` or `qwen` role) as evaluator.

  python -m arretrieval.rageval.ragas_eval rag  --config A_baseline --per-cell 16     # RAG answers, stratified subset
  python -m arretrieval.rageval.ragas_eval meta                                      # planted-error set (faithfulness + factual correctness)

RAG mode metrics: faithfulness, answer_relevancy (bge-m3 AND the RAG system's own MiniLM on the same generated questions),
context_precision (with reference), context_recall, factual_correctness (F1), semantic similarity (bge-m3 cosine).
Unanswerable questions only get faithfulness + answer_relevancy (no reference answer exists to compare against).
Failures are stored per metric (`err_<metric>`), never swallowed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness

from .llm import get_llm
from .paths import DATA, RUNS
from .ragas_adapters import SCORING_KEY, CachedRagasLLM, LocalSTEmbeddings, MiniLMEmbeddings, cosine

RAGAS_VERSION = None


def build_metrics(model: str = "gemma", embeddings: bool = True):
    import ragas

    global RAGAS_VERSION
    RAGAS_VERSION = ragas.__version__
    llm = CachedRagasLLM(get_llm(model))
    llm_s = CachedRagasLLM(get_llm(model), temperature=0.4, sampling=True)  # answer relevancy samples 3 questions
    # separate instance (own call counters) for the MiniLM variant -> same (prompt, seed) pairs -> LLM cache hits, so both
    # embedders score the *same* generated questions and the difference isolates the embedding model
    llm_s2 = CachedRagasLLM(get_llm(model), temperature=0.4, sampling=True)
    bge, mini = (LocalSTEmbeddings("BAAI/bge-m3"), MiniLMEmbeddings()) if embeddings else (None, None)
    return {
        "llm": llm,
        "llm_s": llm_s,
        "llm_s2": llm_s2,
        "bge": bge,
        "mini": mini,
        "faith": Faithfulness(llm=llm),
        "rel_bge": AnswerRelevancy(llm=llm_s, embeddings=bge) if embeddings else None,
        "rel_mini": AnswerRelevancy(llm=llm_s2, embeddings=mini) if embeddings else None,
        "cp": ContextPrecision(llm=llm),
        "cr": ContextRecall(llm=llm),
        "fc": FactualCorrectness(llm=llm, mode="f1"),
        "fc_p": FactualCorrectness(llm=llm, mode="precision"),
        "fc_r": FactualCorrectness(llm=llm, mode="recall"),
    }


async def _safe(out: dict, name: str, coro):
    try:
        r = await coro
        out[name] = None if r.value != r.value else float(r.value)  # NaN -> None
    except Exception as e:  # noqa: BLE001 - recorded
        out[name] = None
        out[f"err_{name}"] = repr(e)[:300]


async def score_rag(row: dict, M: dict) -> dict:
    SCORING_KEY.set(row["qid"])
    # XQuAD references are bare spans ("308"). Ragas FactualCorrectness scores a correct answer 0.0 against a bare span
    # (see toy_ragas: fc_short_ref), so reference-based metrics use the human-translated *sentence* containing the span.
    row = {**row, "reference_span": row["reference"], "reference": row.get("ref_sentence") or row["reference"]}
    ctxs = [p["text"] for p in row["retrieved"]]
    q, a = row["question"], row["answer"]
    out = {"qid": row["qid"]}
    tasks = [_safe(out, "faithfulness", M["faith"].ascore(q, a, ctxs)), _safe(out, "answer_relevancy_bge", M["rel_bge"].ascore(q, a))]
    if row["answerable"]:
        tasks += [
            _safe(out, "context_precision", M["cp"].ascore(q, row["reference"], ctxs)),
            _safe(out, "context_recall", M["cr"].ascore(q, ctxs, row["reference"])),
            _safe(out, "factual_correctness", M["fc"].ascore(a, row["reference"])),
            _safe(out, "factual_correctness_span_ref", M["fc"].ascore(a, row["reference_span"])),
        ]
    await asyncio.gather(*tasks)
    # same generated questions (cached), different embedder -> isolates the effect of the embedding model
    await _safe(out, "answer_relevancy_minilm", M["rel_mini"].ascore(q, a))
    if row["answerable"]:
        out["semantic_similarity"] = cosine(M["bge"].embed_text(a), M["bge"].embed_text(row["reference"]))
    return out


async def score_meta(it: dict, M: dict) -> dict:
    SCORING_KEY.set(it["id"])
    ctxs = [c["text"] for c in it["contexts"]]
    out = {"id": it["id"]}
    ref = it.get("ref_sentence") or it["reference"]
    await asyncio.gather(
        _safe(out, "faithfulness", M["faith"].ascore(it["question"], it["answer"], ctxs)),
        _safe(out, "factual_correctness", M["fc"].ascore(it["answer"], ref)),
        _safe(out, "factual_correctness_precision", M["fc_p"].ascore(it["answer"], ref)),
        _safe(out, "factual_correctness_recall", M["fc_r"].ascore(it["answer"], ref)),
    )
    return out


async def run_all(rows, fn, M, key: str, out_path, conc: int = 6):
    conc = 12 if M["llm"].llm.name == "qwen" else conc
    sem = asyncio.Semaphore(conc)
    res, t0 = [], time.time()

    async def one(r):
        async with sem:
            x = await fn(r, M)
            res.append(x)
            if len(res) % 10 == 0:
                print(f"[ragas] {len(res)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
            return x

    out = await asyncio.gather(*[one(r) for r in rows])
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def stratified_subset(rows: list[dict], per_cell: int, n_unans: int, seed: int = 3, misses: int = 0) -> list[dict]:
    rng = random.Random(seed)
    sel = []
    for ql in ("en", "ar"):
        for kind in ("mono", "cross"):
            cell = [r for r in rows if r["answerable"] and r["q_lang"] == ql and r["kind"] == kind]
            rng.shuffle(cell)
            sel += cell[:per_cell]
    un = [r for r in rows if not r["answerable"]]
    rng.shuffle(un)
    # optionally add answerable questions whose gold passage was NOT retrieved (to test how Ragas context recall reacts)
    have = {r["qid"] for r in sel}
    miss = [r for r in rows if r["answerable"] and not r["gold_retrieved"] and r["qid"] not in have]
    rng.shuffle(miss)
    return sel + un[:n_unans] + miss[:misses]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rag", "meta"])
    ap.add_argument("--config", default="A_baseline")
    ap.add_argument("--per-cell", type=int, default=16)
    ap.add_argument("--n-unans", type=int, default=10)
    ap.add_argument(
        "--add-misses", type=int, default=0, help="rag mode: also score N answerable questions whose gold passage was not retrieved"
    )
    ap.add_argument("--llm", default="gemma", choices=["gemma", "qwen"], help="evaluator LLM for Ragas (meta mode supports qwen)")
    ap.add_argument("--max-per-type", type=int, default=0, help="meta mode: cap items per planted type (stratified, seeded)")
    a = ap.parse_args()
    M = build_metrics(a.llm, embeddings=(a.mode == "rag"))
    suffix = "" if a.llm == "gemma" else f"_{a.llm}"
    if a.mode == "rag":
        rows = [json.loads(l) for l in open(RUNS / f"answers_{a.config}.jsonl")]
        refs = {q["qid"]: q["ref_sentence"] for q in (json.loads(l) for l in open(DATA / "eval_questions.jsonl"))}
        rows = [{**r, "ref_sentence": refs[r["qid"]]} for r in rows]
        sub = stratified_subset(rows, a.per_cell, a.n_unans, misses=a.add_misses)
        out = asyncio.run(run_all(sub, score_rag, M, "qid", RUNS / f"ragas_{a.config}{suffix}.jsonl"))
    else:
        items = [json.loads(l) for l in open(DATA / "meta_items.jsonl")]
        if a.max_per_type:
            rng, by = random.Random(5), {}
            for it in items:
                by.setdefault(it["etype"], []).append(it)
            items = []
            for et in sorted(by):
                rng.shuffle(by[et])
                items += by[et][: a.max_per_type]
        out = asyncio.run(run_all(items, score_meta, M, "id", RUNS / f"ragas_meta{suffix}.jsonl"))
    errs = sum(1 for o in out for k in o if k.startswith("err_"))
    print(
        f"ragas {RAGAS_VERSION}: scored {len(out)} rows, metric errors={errs}, "
        f"llm calls={M['llm'].calls + M['llm_s'].calls + M['llm_s2'].calls}, parse failures={M['llm'].parse_failures + M['llm_s'].parse_failures + M['llm_s2'].parse_failures}"
    )


if __name__ == "__main__":
    main()
