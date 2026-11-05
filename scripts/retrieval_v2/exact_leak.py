"""Read-only: do any v2 TRAINING questions (after the exclusion list) exactly equal (normalised) an eval query of MIRACL-HN.v2 ar dev or NamaaMrTydi test?"""
import json, mteb
from lab.retrieval.v2.audit_ood import load_split, toks
from lab.retrieval.v2.common_v2 import load_queries_v2
from lab.retrieval.v2.guard_split import load_tasks
(mq, _, _), (nq, _, _, _) = load_tasks()
ev = {"miracl": {" ".join(toks(t)) for t in mq.values()}, "namaa": {" ".join(toks(t)) for t in nq.values()}}
ex = set(json.load(open("/lab/runs/retrieval/v2/data/exclude_qids.json"))["qids"])
Q = load_queries_v2(); tq = Q[Q.split == "train"]; kept = tq[~tq.qid.isin(ex)]
print("train", len(tq), "kept", len(kept), "cols", list(Q.columns)[:12])
for name, s in ev.items():
    hit = kept[kept.text.map(lambda t: " ".join(toks(t)) in s)] if "text" in kept.columns else None
    if hit is None: print("no text col"); break
    print(name, "eval queries", len(s), "exact matches in KEPT training rows", len(hit), "distinct", hit.text.map(lambda t: " ".join(toks(t))).nunique(), "by slice", hit.slice.value_counts().to_dict() if "slice" in hit else "")
