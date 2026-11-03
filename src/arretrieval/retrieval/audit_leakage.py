"""Independent leakage audit over the FINAL tables (does not reuse build_data's logic).

Checks, for every pair of splits (train/dev/test):
  1. gold passage of every query lies in its own split's pool; pools share no passage id
  2. exact duplicate questions across splits (normalized text, per language)
  3. article-title overlap between pools
  4. near-duplicate passages: share of a passage's word 6-gram shingles that also occur in a passage of another pool
  5. near-duplicate questions: word-set Jaccard >= 0.8 between a test/dev question and a train question
  6. translated (cross-lingual) train/dev queries only derive from questions of their own split
Run: python -m arretrieval.retrieval.audit_leakage
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import pandas as pd

from .paths import CARD, DATA
from .textnorm import dedupe_key, tokenize

SH = 6


def shingles(text: str) -> set[int]:
    t = tokenize(text, "norm")
    return {hash(tuple(t[i : i + SH])) for i in range(max(0, len(t) - SH + 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.parquet")
    args = ap.parse_args()
    P, Q = pd.read_parquet(DATA / "passages.parquet"), pd.read_parquet(DATA / args.queries)
    out: dict = {}
    pool_of = dict(zip(P.pid, P.pool))
    out["gold_outside_pool"] = int(sum(pool_of[p] != s for p, s in zip(Q.pos_pid, Q.split)))
    out["duplicate_pids_across_pools"] = int(P.pid.duplicated().sum())
    # exact duplicate questions across splits (same-split duplicates, e.g. XQuAD directions, are fine)
    qk = defaultdict(set)
    for r in Q.itertuples():
        qk[(r.qlang, dedupe_key(r.text))].add(r.split)
    out["questions_in_multiple_splits"] = sum(len(v) > 1 for v in qk.values())
    out["question_split_pairs"] = dict(pd.Series(["+".join(sorted(v)) for v in qk.values() if len(v) > 1]).value_counts())
    # titles
    tp = {pool: {dedupe_key(t) for t in P[(P.pool == pool) & (P.title.str.len() > 0)].title} for pool in ("train", "dev", "test")}
    out["title_overlap_all_passages(incl. distractors)"] = {
        f"{a}&{b}": len(tp[a] & tp[b]) for a, b in (("train", "test"), ("train", "dev"), ("dev", "test"))
    }
    gold = set(Q.pos_pid)
    tg = {
        pool: {dedupe_key(t) for t in P[(P.pool == pool) & P.pid.isin(gold) & (P.title.str.len() > 0)].title}
        for pool in ("train", "dev", "test")
    }
    out["title_overlap_gold_passages_only"] = {
        f"{a}&{b}": len(tg[a] & tg[b]) for a, b in (("train", "test"), ("train", "dev"), ("dev", "test"))
    }
    out["note_title_overlap"] = (
        "Overlap with distractor-only articles comes from the per-source title hash (the same Wikipedia article can be a TyDi "
        "distractor in the test pool and an ARCD train article); identical paragraphs are removed by text key. "
        "Gold test articles are excluded from train/dev by construction (see build_data)."
    )
    # near-duplicate passages
    idx = {pool: defaultdict(set) for pool in ("train", "dev", "test")}
    sh = {}
    for r in P.itertuples():
        s = shingles(r.text)
        sh[r.pid] = s
        for h in s:
            idx[r.pool][h].add(r.pid)
    near = {}
    for a, b in (("test", "train"), ("dev", "train"), ("test", "dev")):
        counts = {0.5: 0, 0.8: 0}
        examples = []
        flagged = []
        for r in P[P.pool == a].itertuples():
            s = sh[r.pid]
            if not s:
                continue
            hits = defaultdict(int)
            for h in s:
                for pid in idx[b].get(h, ()):
                    hits[pid] += 1
            if not hits:
                continue
            best_pid, best = max(hits.items(), key=lambda kv: kv[1])
            frac = best / len(s)
            for th in counts:
                counts[th] += frac >= th
            if frac >= 0.5:
                flagged.append(r.pid)
            if frac >= 0.5 and len(examples) < 3:
                examples.append({"a": r.pid, "b": best_pid, "frac": round(frac, 2)})
        near[f"{a}_vs_{b}"] = {
            "passages_checked": int((P.pool == a).sum()),
            "containment>=0.5": counts[0.5],
            "containment>=0.8": counts[0.8],
            "gold_queries_affected": int(Q[(Q.split == a) & Q.pos_pid.isin(flagged)].shape[0]),
            "examples": examples,
        }
        if (a, b) == ("test", "train"):
            (CARD / "test_pids_near_train.json").write_text(json.dumps(flagged))
    out["near_duplicate_passages(6-gram containment)"] = near
    # near-duplicate questions (token-set Jaccard >= 0.8)
    toks = {r.qid: frozenset(tokenize(r.text, "norm")) for r in Q.itertuples()}
    inv = defaultdict(set)
    for r in Q[Q.split == "train"].itertuples():
        for t in toks[r.qid]:
            inv[(r.qlang, t)].add(r.qid)
    nq = {}
    for sp in ("test", "dev"):
        n_hit, ex = 0, []
        sub = Q[(Q.split == sp) & ~Q.slice.str.contains("xl")]
        for r in sub.itertuples():
            ts = toks[r.qid]
            if len(ts) < 4:
                continue
            cands = defaultdict(int)
            for t in ts:
                for qid in inv.get((r.qlang, t), ()):
                    cands[qid] += 1
            hit = False
            for qid, c in cands.items():
                if c / len(ts) >= 0.8:
                    j = len(ts & toks[qid]) / len(ts | toks[qid])
                    if j >= 0.8:
                        hit = True
                        if len(ex) < 3:
                            ex.append([r.text, Q.loc[Q.qid == qid, "text"].iloc[0]])
                        break
            n_hit += hit
        nq[sp] = {"queries_checked": int(len(sub)), "near_dup_of_a_train_question(J>=0.8)": n_hit, "examples": ex}
    out["near_duplicate_questions"] = nq
    # translated queries derive from their own split
    src_split = dict(zip(Q.qid, Q.split))
    x = Q[Q.source == "gemma_translation"]
    out["translated_queries_with_foreign_source_split"] = int(sum(src_split.get(s) != sp for s, sp in zip(x.src_qid, x.split)))
    out["translated_queries"] = {k: int(v) for k, v in x.groupby("split").size().items()}
    (CARD / ("leakage_audit.json" if args.queries == "queries.parquet" else "leakage_audit_preview.json")).write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
