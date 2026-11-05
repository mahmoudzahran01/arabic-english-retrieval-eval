"""Do the out-of-domain evaluation sets share text with the TRAINING data?  (passage 8-gram overlap and near-duplicate questions)

For every mteb task/subset/split used in the OOD tables: how many of its corpus passages have >= 50% of their word 8-grams inside the training passages
(train pool of the v1/v2 benchmark, both languages), and how many of its queries are near-duplicates (token Jaccard >= 0.8) of a training query.
Run (CPU container): NO_GPU=1 python -m arretrieval.retrieval.v2.audit_ood
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

from arretrieval.retrieval.textnorm import normalize_ar

from .common_v2 import V2RES, load_passages_v2, load_queries_v2

TASKS = [
    ("XPQARetrieval", ["ara-eng", "eng-ara", "ara-ara"], ["test"]),
    ("MLQARetrieval", ["ara-eng", "eng-ara", "ara-ara"], ["validation", "test"]),
    ("BelebeleRetrieval", ["arb_Arab-eng_Latn", "eng_Latn-arb_Arab", "arb_Arab-arb_Arab"], ["test"]),
    ("MintakaRetrieval", ["ar"], ["test"]),
    ("MKQARetrieval", ["ar"], ["train"]),
    ("XQuADRetrieval", ["ar"], ["validation"]),
    ("MIRACLRetrievalHardNegatives.v2", ["ar"], ["dev"]),
]
WORD = re.compile(r"\w+", re.UNICODE)
N = 8


def toks(t: str) -> list[str]:
    return WORD.findall(normalize_ar(t).lower())


def shingles(t: str) -> set[int]:
    w = toks(t)
    return {hash(tuple(w[i : i + N])) for i in range(max(1, len(w) - N + 1))} if len(w) >= N else set()


def load_split(task, sub: str, sp: str, with_title: bool = False):
    """(queries {id: text}, corpus {id: text}, relevant {qid: {doc: score}}) for one subset/split, whatever way the mteb task stores them."""

    def txt(x):
        if isinstance(x, dict):
            return (x.get("title", "") + " " + x.get("text", "")).strip()
        return x if isinstance(x, str) else str(x)

    if getattr(task, "dataset", None):
        d = task.dataset[sub][sp]
        ctext = (
            lambda r: (txt(r["title"]) + " " + txt(r["text"])).strip() if with_title and "title" in r and r["title"] else txt(r["text"])
        )  # mteb embeds "title text"
        return ({r["id"]: txt(r["text"]) for r in d["queries"]}, {r["id"]: ctext(r) for r in d["corpus"]}, d["relevant_docs"])
    return (
        {k: txt(v) for k, v in task.queries[sub][sp].items()},
        {k: txt(v) for k, v in task.corpus[sub][sp].items()},
        task.relevant_docs[sub][sp],
    )


def main() -> None:
    import mteb

    P = load_passages_v2()
    Q = load_queries_v2()
    tq = Q[Q.split == "train"]
    tag = ""
    if (
        len(sys.argv) > 1
    ):  # python -m ...audit_ood EXCLUDE.json : audit the contamination-FILTERED training set (queries kept + their positive passages)
        ex = set(json.loads(open(sys.argv[1]).read())["qids"])
        tq = tq[~tq.qid.isin(ex)]
        tp = P[P.pid.isin(set(tq.pos_pid))]
        tag = "_filtered"
    else:
        tp = P[P.pool == "train"]
    train_sh: set[int] = set()
    for t in tp.text:
        train_sh |= shingles(t)
    qsets = [frozenset(toks(t)) for t in tq.text]
    inv: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(qsets):
        for w in s:
            inv[w].append(i)
    print(f"train shingles {len(train_sh):,}  train queries {len(qsets):,}", flush=True)
    out = {}
    for name, subs, splits in TASKS:
        task = mteb.get_task(name, hf_subsets=subs, eval_splits=splits)
        task.load_data()
        for sub in subs:
            for sp in splits:
                try:
                    qd, cd, _ = load_split(task, sub, sp)
                except Exception as e:
                    print("skip", name, sub, sp, type(e).__name__, flush=True)
                    continue
                ctext = list(cd.values())
                hit = 0
                for t in ctext:
                    s = shingles(t)
                    if s and len(s & train_sh) / len(s) >= 0.5:
                        hit += 1
                qtext = sorted(set(qd.values()))
                dup = 0
                for t in qtext:
                    s = frozenset(toks(t))
                    if not s:
                        continue
                    c: Counter = Counter()
                    for w in s:
                        lst = inv.get(w, ())
                        if len(lst) <= 300:
                            c.update(lst)
                    if any(n >= 2 and len(s & qsets[i]) / len(s | qsets[i]) >= 0.8 for i, n in c.most_common(30)):
                        dup += 1
                out[f"{name}/{sub}/{sp}"] = {
                    "corpus": len(ctext),
                    "corpus_passages_>=50%_8gram_overlap_with_train": hit,
                    "queries": len(qtext),
                    "queries_near_duplicate_of_a_train_query": dup,
                }
                print(name, sub, sp, out[f"{name}/{sub}/{sp}"], flush=True)
    (V2RES / f"audit_ood{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
