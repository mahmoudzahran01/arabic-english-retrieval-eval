"""Build the exclusion list that keeps the v2 training data clear of every evaluation set we (or the MTEB-Arabic project) report on.

Found by audit_ood.py: 408 of the 1,000 MIRACL-ar dev queries are near-duplicates (token Jaccard >= 0.8) of questions in v1's TyDi-QA-ar TRAIN pool, so v1's
training data is not clean with respect to MIRACL (listed in SELECTED_TASKS.md).
KNOWN LIMITATION (found 2026-11-05 by an exact-match re-check; training was NOT repeated): the near-duplicate index skips words with more than 300 postings, so short queries made
of common words get no candidates and EXACT copies of such questions are not caught: 9 kept training rows (8 distinct questions, all slice train_tydi_ar) equal a MIRACL-HN.v2 ar dev query
after normalisation (0.8% of 998 normalised eval queries); NamaaMrTydi: 0.  Disclosed in RESULTS_v2.md and the model cards.  v2 therefore drops, from its training set,
  (A) every training question whose text is a near-duplicate (token Jaccard >= 0.8, >= 2 shared tokens) of a query of any listed eval task
      (MIRACL-HN.v2 ar dev, MLQA val+test, XPQA, Belebele, Mintaka, MKQA, XQuAD) - for translated questions the source question counts too,
  (B) every training example whose POSITIVE passage has >= 50% of its word 8-grams inside a RELEVANT (positive) passage of those tasks.
Writes runs/retrieval/v2/data/exclude_qids.json.     Run (CPU container): NO_GPU=1 python -m arretrieval.retrieval.v2.contam_v2
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd

from .audit_ood import TASKS, load_split, shingles, toks
from .common_v2 import V2DATA, load_passages_v2, load_queries_v2

JACCARD = 0.8


def main() -> None:
    import mteb

    eval_q: dict[str, set[str]] = {}  # task -> query texts
    pos_sh: set[int] = set()
    n_pos = Counter()
    for name, subs, splits in TASKS:
        task = mteb.get_task(name, hf_subsets=subs, eval_splits=splits)
        task.load_data()
        for sub in subs:
            for sp in splits:
                try:
                    qd, cd, rel = load_split(task, sub, sp)
                except Exception as e:
                    print("skip", name, sub, sp, type(e).__name__, flush=True)
                    continue
                eval_q.setdefault(name, set()).update(qd.values())
                relevant = {doc for docs in rel.values() for doc, sc in docs.items() if sc > 0}
                for did, t in cd.items():
                    if did in relevant:
                        pos_sh |= shingles(t)
                        n_pos[name] += 1
        print(name, "queries", len(eval_q[name]), "positive passages (cumulative)", n_pos[name], flush=True)
    all_eval = sorted({t for s in eval_q.values() for t in s})
    esets = [frozenset(toks(t)) for t in all_eval]
    inv: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(esets):
        for w in s:
            inv[w].append(i)

    def near_dup(text: str) -> bool:
        s = frozenset(toks(text))
        if not s:
            return False
        c: Counter = Counter()
        for w in s:
            lst = inv.get(w, ())
            if len(lst) <= 300:
                c.update(lst)
        return any(n >= 2 and len(s & esets[i]) / len(s | esets[i]) >= JACCARD for i, n in c.most_common(40))

    Q = load_queries_v2()
    P = load_passages_v2()
    tq = Q[Q.split == "train"].copy()
    src_text = dict(zip(Q.qid, Q.text))
    cand = pd.read_json(V2DATA / "xling2_candidates.jsonl", lines=True)
    src_text.update(dict(zip(cand.src_qid, cand.text)))
    ptext = dict(zip(P.pid, P.text))
    pass_hit: dict[str, bool] = {}
    for pid in set(tq.pos_pid):
        s = shingles(ptext[pid])
        pass_hit[pid] = bool(s) and len(s & pos_sh) / len(s) >= 0.5
    rows = []
    for r in tq.itertuples():
        a = near_dup(r.text) or (bool(r.src_qid) and r.src_qid in src_text and near_dup(src_text[r.src_qid]))
        b = pass_hit[r.pos_pid]
        rows.append((r.qid, r.slice, a, b))
    df = pd.DataFrame(rows, columns=["qid", "slice", "query_dup", "passage_dup"])
    df["drop"] = df.query_dup | df.passage_dup
    summary = {
        "train_queries": len(df),
        "dropped": int(df["drop"].sum()),
        "dropped_query_dup": int(df.query_dup.sum()),
        "dropped_passage_overlap": int(df.passage_dup.sum()),
        "train_positive_passages_overlapping_eval_positives": int(sum(pass_hit.values())),
        "by_slice": {k: {"n": int(len(g)), "dropped": int(g["drop"].sum())} for k, g in df.groupby("slice")},
        "eval_tasks": {k: len(v) for k, v in eval_q.items()},
    }
    (V2DATA / "exclude_qids.json").write_text(json.dumps({"qids": sorted(df[df["drop"]].qid), "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
