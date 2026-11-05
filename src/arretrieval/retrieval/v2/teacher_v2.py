"""bge-m3 teacher embeddings (dense, 1024-d, L2-normalised, fp16 storage) for every train query and train-pool passage.

Incremental: queries/passages already embedded are skipped, so it can be re-run after more translations arrive.
Output: runs/retrieval/v2/data/teacher_bgem3_{q,p}.npz  (ids + embeddings).   Run (GPU): python -m arretrieval.retrieval.v2.teacher_v2
"""

from __future__ import annotations

import numpy as np

from arretrieval.retrieval import evalkit as E

from .common_v2 import TEACHER, V2DATA, data_family, load_passages_v2, load_queries_v2


def _load(path):
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return list(z["ids"]), z["emb"]
    return [], np.zeros((0, 1), dtype=np.float16)


def _extend(path, ids_all: list[str], texts: dict[str, str], enc) -> None:
    ids, emb = _load(path)
    have = set(ids)
    todo = [i for i in ids_all if i not in have]
    print(f"{path.name}: {len(ids)} cached, {len(todo)} new", flush=True)
    if todo:
        new = enc([texts[i] for i in todo]).astype(np.float16)
        ids, emb = ids + todo, (new if len(ids) == 0 else np.concatenate([emb, new]))
        np.savez(path, ids=np.array(ids, dtype=object), emb=emb)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="teacher model dir (default: BAAI/bge-m3); e.g. the fine-tuned v2-m3")
    ap.add_argument("--tag", default="bgem3", help="file tag: teacher_{tag}_{q,p}.npz")
    ap.add_argument("--prefix", default="none", choices=["e5", "none"])
    ap.add_argument(
        "--data", default="v3", choices=["v1", "v2", "v3"], help="embed the queries of these data generations (+ the whole train pool)"
    )
    a = ap.parse_args()
    hf, qp, pp = TEACHER
    if a.model:
        hf, qp, pp = a.model, *(("query: ", "passage: ") if a.prefix == "e5" else ("", ""))
    m = E.Dense(hf, a.tag + "-teacher", qp, pp, max_len=512, bs=32, cache=False)
    queries, passages = load_queries_v2(), load_passages_v2()
    tq = queries[queries.split == "train"]
    tq = tq[data_family(tq.slice).isin({"v1": ["v1"], "v2": ["v1", "v2"], "v3": ["v1", "v2", "v3"]}[a.data] + ["replay"])]
    tp = passages[passages.pool == "train"]
    _extend(V2DATA / f"teacher_{a.tag}_p.npz", list(tp.pid), dict(zip(tp.pid, tp.text)), lambda t: m.encode(t, "p"))
    _extend(V2DATA / f"teacher_{a.tag}_q.npz", list(tq.qid), dict(zip(tq.qid, tq.text)), lambda t: m.encode(t, "q"))


if __name__ == "__main__":
    main()
