"""Read-only: how many MIRACL-HN.v2 ar dev queries are EXACT (after normalisation) copies of TyDi-QA-ar train questions, vs near-duplicates (Jaccard>=0.8)."""

import mteb
from lab.retrieval import build_data as B
from lab.retrieval.v2.audit_ood import load_split, toks
from lab.retrieval.v2.guard_split import contaminated_queries

t1 = mteb.get_task("MIRACLRetrievalHardNegatives.v2", hf_subsets=["ar"], eval_splits=["dev"])
t1.load_data()
q, _, _ = load_split(t1, "ar", "dev")
tydi = list(B.load_sources()["tydi_ar_train"].question)
raw = {t.strip() for t in tydi}
norm = {" ".join(toks(t)) for t in tydi}
n_raw = sum(t.strip() in raw for t in q.values())
n_norm = sum(" ".join(toks(t)) in norm for t in q.values())
lab = contaminated_queries(q)
n_near_tydi = sum(v in ("tydi", "both") for v in lab.values())
n_any = sum(bool(v) for v in lab.values())
print(f"queries {len(q)}  exact_raw {n_raw}  exact_normalised {n_norm}  near_dup_tydi {n_near_tydi}  near_dup_tydi_or_miracl_train {n_any}")
import json, time
json.dump({"task": "MIRACLRetrievalHardNegatives.v2", "subset": "ar", "split": "dev", "n_queries": len(q), "exact_raw_tydi_ar_train": n_raw, "exact_normalised_tydi_ar_train": n_norm,
           "flagged_current_detector_tydi": n_near_tydi, "flagged_current_detector_tydi_or_miracl_train": n_any,
           "computed": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
          open("/lab/runs/retrieval/v2/results/exact_overlap.json", "w"), indent=1)
