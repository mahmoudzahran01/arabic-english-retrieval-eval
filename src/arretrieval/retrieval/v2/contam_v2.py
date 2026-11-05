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

import argparse
import json
from collections import Counter, defaultdict

import pandas as pd

from .audit_ood import TASKS, load_split, shingles, toks
from .common_v2 import V2DATA, load_passages_v2, load_queries_v2

JACCARD = 0.8
# v2.1 (2026-11-05): the NamaaMrTydi test queries / positives are also treated as an evaluation set, and EXACT normalised copies are flagged in addition to the Jaccard detector
EXTRA_TASKS = [("NamaaMrTydiReranking", ["default"], ["test"])]


def main() -> None:
    import mteb

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exclude_qids.json", help="file name under runs/retrieval/v2/data")
    ap.add_argument(
        "--v21",
        action="store_true",
        help="v2.1: add NamaaMrTydi to the evaluation sets and flag exact normalised copies (the original v2 list was built without either)",
    )
    args = ap.parse_args()
    eval_q: dict[str, set[str]] = {}  # task -> query texts
    pos_sh: set[int] = set()
    n_pos = Counter()
    for name, subs, splits in TASKS + (EXTRA_TASKS if args.v21 else []):
        task = (
            mteb.get_task(name, eval_splits=splits)
            if name == "NamaaMrTydiReranking"
            else mteb.get_task(name, hf_subsets=subs, eval_splits=splits)
        )
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
    norm = lambda t: " ".join(toks(t))  # noqa: E731
    exact_all = {norm(t) for t in all_eval} if args.v21 else set()
    esets = [frozenset(toks(t)) for t in all_eval]
    inv: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(esets):
        for w in s:
            inv[w].append(i)

    def near_dup(text: str) -> bool:
        s = frozenset(toks(text))
        if not s:
            return False
        if norm(text) in exact_all:
            return True
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
        "v21": bool(args.v21),
    }
    # exact-match leak check among the KEPT rows (text, or the source question of a translated row), per evaluation task
    kept = tq[tq.qid.isin(set(df[~df["drop"]].qid))]
    leaks = {}
    for name in ("MIRACLRetrievalHardNegatives.v2", "NamaaMrTydiReranking"):
        if name not in eval_q:
            continue
        ev = {norm(t) for t in eval_q[name]}
        hit = [
            r.qid
            for r in kept.itertuples()
            if norm(r.text) in ev or (bool(r.src_qid) and r.src_qid in src_text and norm(src_text[r.src_qid]) in ev)
        ]
        leaks[name] = {"kept_rows_with_exact_match": len(hit), "eval_queries_normalised": len(ev)}
    summary["exact_leaks_in_kept_rows"] = leaks
    (V2DATA / args.out).write_text(json.dumps({"qids": sorted(df[df["drop"]].qid), "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
