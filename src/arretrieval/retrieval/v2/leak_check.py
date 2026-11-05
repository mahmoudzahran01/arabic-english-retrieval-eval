"""Read-only exact-match leak check of the v2 TRAINING data against the guard-task evaluation queries (added after the coordinator's exact_overlap / exact_leak checks of 2026-11-05).

The near-duplicate detector behind exclude_qids.json (contam_v2.py) skips words with more than 300 postings, so EXACT copies of short, common-word questions can survive the filter.
This script counts, after normalisation (" ".join(toks(text))), the KEPT training rows that equal an evaluation query of MIRACL-HN.v2 ar dev or NamaaMrTydi test, the translated copies of those
rows (src_qid), and stores the leaked evaluation query ids so the guard-split report can also be recomputed without them.   No retraining.
Run (CPU): NO_GPU=1 python -m arretrieval.retrieval.v2.leak_check      -> results/leak_exact.json
"""

from __future__ import annotations

import json

from .audit_ood import toks
from .common_v2 import V2DATA, V2RES, load_queries_v2
from .guard_split import load_tasks


def main() -> None:
    (mq, _, _), (nq, _, _, _) = load_tasks()
    ex = set(json.loads((V2DATA / "exclude_qids.json").read_text())["qids"])
    q = load_queries_v2()
    tq = q[q.split == "train"]
    kept = tq[~tq.qid.isin(ex)]
    norm = lambda t: " ".join(toks(t))  # noqa: E731
    src_col = next((c for c in ("src_qid", "source_qid", "orig_qid", "parent_qid") if c in q.columns), None)
    out = {"train_rows": int(len(tq)), "kept_rows": int(len(kept)), "columns": list(q.columns), "source_column": src_col, "tasks": {}}
    for name, qd in (("miracl_hn_v2_ar_dev", mq), ("namaa_mrtydi_test", nq)):
        by_norm: dict[str, list[str]] = {}
        for qid, t in qd.items():
            by_norm.setdefault(norm(t), []).append(qid)
        hit = kept[kept.text.map(lambda t, by_norm=by_norm: norm(t) in by_norm)]
        leaked_eval = sorted({qid for t in hit.text for qid in by_norm[norm(t)]})
        tr = kept[kept[src_col].isin(set(hit.qid))] if src_col else kept.iloc[0:0]
        out["tasks"][name] = {
            "eval_queries": len(qd),
            "eval_queries_normalised_distinct": len(by_norm),
            "exact_matches_in_kept_training_rows": int(len(hit)),
            "distinct_questions": int(hit.text.map(norm).nunique()),
            "by_slice": hit.slice.value_counts().to_dict() if len(hit) else {},
            "translated_copies_of_those_rows_in_kept_training_rows": int(len(tr)),
            "translated_copies_by_slice": tr.slice.value_counts().to_dict() if len(tr) else {},
            "leaked_eval_qids": leaked_eval,
            "share_of_normalised_eval_queries": (len({norm(t) for t in hit.text}) / len(by_norm)) if by_norm else 0.0,
        }
        print(
            name,
            {k: v for k, v in out["tasks"][name].items() if k != "leaked_eval_qids"},
            "leaked eval qids:",
            len(leaked_eval),
            flush=True,
        )
    (V2RES / "leak_exact.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", V2RES / "leak_exact.json")


if __name__ == "__main__":
    main()
