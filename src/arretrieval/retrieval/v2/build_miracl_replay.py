"""Replay data against forgetting: MIRACL-ar TRAIN (miracl/miracl, Apache-2.0) - the data bge-m3 itself was trained on.

* queries/qrels/corpus come from the raw MIRACL files (miracl/miracl topics+qrels, miracl/miracl-corpus ar shards); judged-relevant passages are the positives,
  judged-NOT-relevant passages (annotator verdicts, so no false-negative guessing) are the hard negatives.
* Filters (before anything is trained on): a train query is dropped if it is a near-duplicate (token Jaccard >= 0.8) of ANY MIRACL-ar dev query, of any query of the v1
  benchmark's test/dev pools, or if a positive overlaps (>= 50% of its word 8-grams) a judged-relevant passage of MIRACL-ar dev or any passage of the v1 TEST pool.
  contam_v2 later applies the full MTEB(ara) list on top (it sees these rows as ordinary train queries).
* HOLD-OUT: 500 of the remaining train queries become a selection-only slice `dev_miracl_ar` with its own mini-corpus (pool `devM`: their judged passages + 10,000 random
  corpus passages that are not used anywhere in training). It is evaluated separately (split `devM`), never mixed into the macro of the existing dev table.
Outputs (runs/retrieval/v2/data): passages_miracl.parquet, queries_miracl.parquet, miracl_negs.json (qid -> judged negatives), manifest_miracl.json.
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.build_miracl_replay
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval.paths import SEED

from .audit_ood import shingles, toks
from .common_v2 import V2DATA

N_HOLD, N_DISTRACT, JAC = 500, 10000, 0.8
NEG_PER_Q = 5


def read_tsv(repo_path: str) -> list[list[str]]:
    p = hf_hub_download("miracl/miracl", repo_path, repo_type="dataset")
    return [ln.rstrip("\n").split("\t") for ln in open(p, encoding="utf-8") if ln.strip()]


def main() -> None:
    topics = {sp: {r[0]: r[1] for r in read_tsv(f"miracl-v1.0-ar/topics/topics.miracl-v1.0-ar-{sp}.tsv")} for sp in ("train", "dev")}
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for sp in ("train", "dev"):
        d: dict[str, dict[str, int]] = defaultdict(dict)
        for r in read_tsv(f"miracl-v1.0-ar/qrels/qrels.miracl-v1.0-ar-{sp}.tsv"):
            d[r[0]][r[2]] = int(r[3])
        qrels[sp] = d
    need = {doc for sp in qrels for q in qrels[sp] for doc in qrels[sp][q]}
    print("topics", {k: len(v) for k, v in topics.items()}, "judged docs needed", len(need), flush=True)
    # stream the 2M-passage Arabic corpus once: keep the judged passages + a reservoir of random passages for the hold-out distractors
    rng = random.Random(SEED + 5)
    text, reservoir, seen = {}, [], 0
    for i in range(5):
        p = hf_hub_download("miracl/miracl-corpus", f"miracl-corpus-v1.0-ar/docs-{i}.jsonl.gz", repo_type="dataset")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                if j["docid"] in need:
                    text[j["docid"]] = (j.get("title", "") + " " + j["text"]).strip() if j.get("title") else j["text"]
                seen += 1
                if seen <= 4 * N_DISTRACT:
                    reservoir.append(j)
                elif rng.random() < 4 * N_DISTRACT / seen:
                    reservoir[rng.randrange(len(reservoir))] = j
        print("corpus shard", i, "scanned", seen, "judged found", len(text), flush=True)
    # ---------------- overlap filters
    Q = E.load_queries()
    P = E.load_passages()
    ref_q = [t for t in topics["dev"].values()] + list(Q[Q.split.isin(["test", "dev"])].text)
    ref_sets = [frozenset(toks(t)) for t in ref_q]
    inv: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(ref_sets):
        for w in s:
            inv[w].append(i)

    def near_dup(t: str) -> bool:
        s = frozenset(toks(t))
        c: dict[int, int] = defaultdict(int)
        for w in s:
            lst = inv.get(w, ())
            if len(lst) <= 300:
                for i in lst:
                    c[i] += 1
        return bool(s) and any(n >= 2 and len(s & ref_sets[i]) / len(s | ref_sets[i]) >= JAC for i, n in c.items())

    ref_sh: set[int] = set()
    for doc, sc in ((d, s) for q in qrels["dev"] for d, s in qrels["dev"][q].items()):
        if sc > 0 and doc in text:
            ref_sh |= shingles(text[doc])
    for t in P[(P.pool == "test") & (P.lang == "ar")].text:
        ref_sh |= shingles(t)

    def overlaps(doc: str) -> bool:
        s = shingles(text[doc])
        return bool(s) and len(s & ref_sh) / len(s) >= 0.5

    rows, dropped = [], {"no_positive": 0, "query_near_dup": 0, "positive_overlap": 0}
    for qid, qt in topics["train"].items():
        judged = {d: s for d, s in qrels["train"].get(qid, {}).items() if d in text}
        pos = [d for d, s in judged.items() if s > 0]
        neg = [d for d, s in judged.items() if s == 0]
        if not pos:
            dropped["no_positive"] += 1
        elif near_dup(qt):
            dropped["query_near_dup"] += 1
        elif any(overlaps(d) for d in pos):
            dropped["positive_overlap"] += 1
        else:
            rows.append((qid, qt, pos, neg))
    rng.shuffle(rows)
    hold, train = rows[:N_HOLD], rows[N_HOLD:]
    train_docs = {d for r in train for d in r[2] + r[3]}
    # ---------------- tables
    mk = lambda d, pool, src: {
        "pid": f"miracl-ar-{d}",
        "lang": "ar",
        "text": text[d],
        "title": "",
        "source": src,
        "pool": pool,
        "kind": "gold",
    }  # noqa: E731
    pas, qs, negs = {}, [], {}
    for qid, qt, pos, neg in train:
        for d in pos + neg:
            pas[d] = mk(d, "train", "miracl_ar_train")
        qs.append(
            {
                "qid": f"miracl_ar-train-{qid}",
                "text": qt,
                "qlang": "ar",
                "pos_pid": f"miracl-ar-{pos[0]}",
                "plang": "ar",
                "split": "train",
                "slice": "train_miracl_ar",
                "source": "miracl_ar_train",
                "title": "",
                "src_qid": "",
            }
        )
        negs[f"miracl_ar-train-{qid}"] = [f"miracl-ar-{d}" for d in neg[:NEG_PER_Q]]
    hold_docs: dict[str, dict] = {}
    for qid, qt, pos, neg in hold:
        for d in pos + neg:
            if d not in train_docs:
                hold_docs[d] = mk(d, "devM", "miracl_ar_holdout")
        pos_ok = [d for d in pos if d in hold_docs]
        if pos_ok:
            qs.append(
                {
                    "qid": f"miracl_ar-hold-{qid}",
                    "text": qt,
                    "qlang": "ar",
                    "pos_pid": f"miracl-ar-{pos_ok[0]}",
                    "plang": "ar",
                    "split": "devM",
                    "slice": "dev_miracl_ar",
                    "source": "miracl_ar_train",
                    "title": "",
                    "src_qid": "",
                }
            )
    n_dist = 0
    for j in rng.sample(reservoir, len(reservoir)):
        d = j["docid"]
        if d not in train_docs and d not in hold_docs and n_dist < N_DISTRACT:
            text[d] = (j.get("title", "") + " " + j["text"]).strip() if j.get("title") else j["text"]
            hold_docs[d] = mk(d, "devM", "miracl_ar_holdout_distractor")
            n_dist += 1
    pa = pd.DataFrame(list(pas.values()) + list(hold_docs.values()))
    pa.to_parquet(V2DATA / "passages_miracl.parquet", index=False)
    pd.DataFrame(qs).to_parquet(V2DATA / "queries_miracl.parquet", index=False)
    (V2DATA / "miracl_negs.json").write_text(json.dumps(negs))
    man = {
        "miracl_train_topics": len(topics["train"]),
        "dropped": dropped,
        "kept_queries": len(rows),
        "train_queries": len(train),
        "holdout_queries": sum(1 for q in qs if q["split"] == "devM"),
        "train_passages": len(pas),
        "holdout_pool_passages": len(hold_docs),
        "holdout_distractors": n_dist,
        "licence": "miracl/miracl and miracl/miracl-corpus: Apache-2.0",
        "mr_tydi": "not added: its train questions are the TyDi-QA-ar train questions v1/v2 already use (castorini/mr-tydi is Apache-2.0 but script-based, corpus = same Arabic Wikipedia)",
    }
    (V2DATA / "manifest_miracl.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
