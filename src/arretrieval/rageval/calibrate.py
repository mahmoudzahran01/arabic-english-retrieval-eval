"""Run LLM-as-judge variants over the planted-error set and store raw verdicts (resumable via the LLM cache).

  python -m arretrieval.rageval.calibrate --model gemma --judges faith_v0,correct_v0,overall_v0
  python -m arretrieval.rageval.calibrate --model qwen  --judges all
Output: runs/rageval/judge_<model>/<judge>.jsonl  ->  {"id", "pass", "raw", ...}
"""

from __future__ import annotations

import argparse
import json

from .judges import JUDGES
from .llm import get_llm
from .paths import DATA, RUNS


def load_items(path=None):
    return [i for i in (json.loads(l) for l in open(path or DATA / "meta_items.jsonl")) if not i.get("excluded")]


def rag_answers_as_items(config: str, skip_abstained: bool = True, ids: list[str] | None = None) -> list[dict]:
    """Turn RAG output rows into judge items (id = qid; contexts = what the generator actually saw).

    Abstentions are scored deterministically in ab_test.py (correct iff the question is unanswerable), so by default
    only real answers go to the LLM judges - this saves ~20% of the calls and loses nothing.
    """
    rows = [json.loads(l) for l in open(RUNS / f"answers_{config}.jsonl")]
    rows = [r for r in rows if not (skip_abstained and r["abstained"]) and (ids is None or r["qid"] in ids)]
    return [
        {
            "id": r["qid"],
            "question": r["question"],
            "q_lang": r["q_lang"],
            "answer": r["answer"],
            "reference": r["reference"],
            "answerable": r["answerable"],
            "cited": r["cited"],
            "kind": r["kind"],
            "contexts": [{"id": p["id"], "text": p["text"], "lang": p["lang"]} for p in r["retrieved"]],
        }
        for r in rows
    ]


def run_judge(name: str, items: list[dict], model: str, out_dir=None):
    llm = get_llm(model)
    fn = JUDGES[name]
    out = out_dir or RUNS / f"judge_{model}"
    out.mkdir(parents=True, exist_ok=True)
    res = llm.pmap(lambda it: {"id": it["id"], **fn(it, llm)}, items, desc=f"{model}:{name}", every=30)
    with open(out / f"{name}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = sum(1 for r in res if r["pass"] is None)
    print(f"{model}:{name}: {len(res)} items, unparsable={bad}, pass-rate={sum(r['pass'] or 0 for r in res) / len(res):.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--judges", default="all")
    ap.add_argument("--items", default=None)
    ap.add_argument("--answers", default=None, help="judge a RAG config's answers instead of the planted set")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated qids (RAG answers mode); also judges abstentions")
    ap.add_argument("--sample", type=int, default=None, help="judge a seeded random sample of N items (cross-checks on real answers)")
    ap.add_argument("--split", default=None, choices=["dev", "test"], help="only judge items of this split (planted set)")
    a = ap.parse_args()
    qids = a.ids.split(",") if a.ids else None
    items = (rag_answers_as_items(a.answers, skip_abstained=not qids, ids=qids) if a.answers else load_items(a.items))[: a.limit]
    if a.split:
        items = [i for i in items if i.get("split") == a.split]
    if a.sample:
        import random

        items = random.Random(11).sample(items, min(a.sample, len(items)))
    if a.answers and not a.out:
        a.out = str(RUNS / f"judge_rag_{a.answers}_{a.model}")
    names = list(JUDGES) if a.judges == "all" else a.judges.split(",")
    from pathlib import Path

    for n in names:
        run_judge(n, items, a.model, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
