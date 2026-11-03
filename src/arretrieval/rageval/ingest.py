"""Embed the corpus with the MiniLM embedder and store it in the exact-cosine collections (see retrieval.py).

python -m arretrieval.rageval.ingest                 # whole-passage index   -> rag_eval_xquad
python -m arretrieval.rageval.ingest --chunks        # sentence-window chunks -> rag_eval_xquad_chunks
"""

from __future__ import annotations

import argparse
import json
import re

from . import retrieval as R
from .paths import DATA

SENT_SPLIT = re.compile(r"(?<=[.!?؟。])\s+")


def chunk_passage(text: str, max_words: int = 60, stride_sents: int = 2, window: int = 3) -> list[str]:
    """Sentence windows (<= `window` sentences, stride `stride_sents`), capped at ~max_words words each.

    Chosen so a chunk usually fits the embedder's 128 word-piece limit (see RESULTS.md: truncation finding).
    """
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s]
    chunks, i = [], 0
    while i < len(sents):
        cur = []
        j = i
        while j < len(sents) and len(cur) < window and (sum(len(x.split()) for x in cur) + len(sents[j].split()) <= max_words or not cur):
            cur.append(sents[j])
            j += 1
        chunks.append(" ".join(cur))
        if j >= len(sents):
            break
        i += stride_sents
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", action="store_true")
    ap.add_argument("--recreate", action="store_true")
    a = ap.parse_args()
    corpus = [json.loads(l) for l in open(DATA / "corpus.jsonl")]
    if a.chunks:
        name = "rag_eval_xquad_chunks"
        items = []
        for c in corpus:
            for ci, t in enumerate(chunk_passage(c["text"])):
                items.append({"id": f"{c['id']}.c{ci}", "parent": c["id"], "lang": c["lang"], "text": t})
    else:
        name = "rag_eval_xquad"
        items = [{"id": c["id"], "parent": c["id"], "lang": c["lang"], "text": c["text"]} for c in corpus]
    R.create_collection(name, 384, recreate=a.recreate)
    vecs = R.embed([i["text"] for i in items])
    R.upsert(name, [{"id": n, "vector": v, "payload": it} for n, (it, v) in enumerate(zip(items, vecs))])
    print(f"{name}: {len(items)} points upserted")


if __name__ == "__main__":
    main()
