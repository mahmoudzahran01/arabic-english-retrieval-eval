"""Non-guard, non-MTEB(ara) Arabic-only OOD development proxy used to judge forgetting mitigations.

mMARCO-ar dev (mteb MMarcoRetrievalMultilingual, machine-translated MS MARCO; NOT in the MTEB(ara) list, never trained on, and the base
multilingual-e5 models saw mMARCO in their own supervised stage, so it measures how much of the base model's general Arabic passage ranking survives
fine-tuning).  A fixed subsample keeps it cheap: 1,000 random dev queries, ALL their positives, plus random distractors up to 25,000 passages
(seed 13).  MRR@10 / nDCG@10, mono Arabic.   Run (GPU): python -m arretrieval.retrieval.v2.ood_dev_ar --models NAME=PATH:e5|none ...
"""

from __future__ import annotations

import argparse
import json
import random

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval import metrics as M

from .audit_ood import load_split
from .common_v2 import V2DATA, V2RES

N_Q, N_DOCS = 1000, 25000


def build():
    p = V2DATA / "ood_dev_ar.json"
    if p.exists():
        return json.loads(p.read_text())
    import mteb

    task = mteb.get_task("MMarcoRetrievalMultilingual", hf_subsets=["ar"], eval_splits=["dev"])
    task.load_data()
    qd, cd, rel = load_split(task, "ar", "dev")
    rng = random.Random(13)
    qids = rng.sample(sorted(q for q in qd if any(sc > 0 for sc in rel.get(q, {}).values())), N_Q)
    pos = {q: [d for d, sc in rel[q].items() if sc > 0 and d in cd] for q in qids}
    keep = {d for ds in pos.values() for d in ds}
    others = sorted(set(cd) - keep)
    keep |= set(rng.sample(others, max(0, N_DOCS - len(keep))))
    out = {"queries": {q: qd[q] for q in qids}, "qrels": pos, "corpus": {d: cd[d] for d in sorted(keep)}}
    p.write_text(json.dumps(out, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    d = build()
    qids, dids = list(d["queries"]), list(d["corpus"])
    import pandas as pd

    q = pd.DataFrame({"qid": qids, "text": [d["queries"][i] for i in qids]})
    c = pd.DataFrame({"pid": dids, "text": [d["corpus"][i] for i in dids]})
    qrels = {i: set(d["qrels"][i]) for i in qids}
    path = V2RES / "ood_dev_ar.json"
    try:
        res = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:  # truncated by a crash
        res = {}
    for spec in a.models:
        name, _, rest = spec.partition("=")
        mp, _, pre = rest.partition(":")
        qp, pp = ("query: ", "passage: ") if pre != "none" else ("", "")
        m = E.Dense(mp, name, qp, pp, max_len=512, bs=64 if pre != "none" else 32)
        run = m.rank(q, c)
        pq = M.evaluate_run({k: v[0] for k, v in run.items()}, qrels)
        res[name] = {
            "n_queries": len(qids),
            "n_docs": len(dids),
            "mrr@10": float(pq["mrr@10"].mean()),
            "ndcg@10": float(pq["ndcg@10"].mean()),
        }
        path.write_text(json.dumps(res, indent=1))  # incremental: a crash / reboot must not lose finished models
        print(
            f"[{name}] mMARCO-ar dev proxy ({len(qids)} q, {len(dids)} docs): MRR@10 {res[name]['mrr@10'] * 100:.1f}  nDCG@10 {res[name]['ndcg@10'] * 100:.1f}",
            flush=True,
        )
        del m
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
