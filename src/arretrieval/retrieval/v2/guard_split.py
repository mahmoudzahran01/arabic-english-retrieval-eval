"""Diagnostic (no training): how much of bge-m3's lead on the MTEB(ara) Arabic guards is query CONTAMINATION rather than forgetting on our side?

bge-m3 was trained on TyDi-QA / Mr.TyDi train and MIRACL train; 42% of the MIRACL-ar dev queries are verbatim TyDi-ar TRAIN questions.  The v2 training data deliberately
removed everything that overlaps MIRACL dev, so a part of bge-m3's guard lead can be memorisation.  Test: split the queries of
  * MIRACLRetrievalHardNegatives.v2 (ar, dev)  and  * NamaaMrTydiReranking (test)
into CONTAMINATED (near-duplicate, token Jaccard >= 0.8 with >= 2 shared tokens, of a question of the FULL TyDi-QA-ar train set or of the MIRACL-ar train set) and CLEAN,
then report the metric per subset for several models.  Metric computed here with our own code (nDCG@10 over the full 192k-passage corpus for MIRACL; average precision over the
task's candidate lists for Namaa) - the full-set numbers are printed next to the mteb numbers as a consistency check.
Run (GPU): python -m arretrieval.retrieval.v2.guard_split --models NAME=PATH:e5|none ...      (per-model results are cached in results/guard_split/NAME.json)
Report:    NO_GPU=1 python -m arretrieval.retrieval.v2.guard_split --report
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from arretrieval.retrieval import build_data as B

from .audit_ood import load_split, toks
from .common_v2 import V2RES

OUT = V2RES / "guard_split"
JAC = 0.8


def contaminated_queries(qtexts: dict[str, str]) -> dict[str, str]:
    """qid -> 'tydi' | 'miracl' | 'both' | '' (clean) against the FULL TyDi-ar train questions and the MIRACL-ar train topics."""
    from huggingface_hub import hf_hub_download

    src = B.load_sources()
    tydi = list(src["tydi_ar_train"].question)
    p = hf_hub_download("miracl/miracl", "miracl-v1.0-ar/topics/topics.miracl-v1.0-ar-train.tsv", repo_type="dataset")
    miracl = [ln.rstrip("\n").split("\t")[1] for ln in open(p, encoding="utf-8") if ln.strip()]
    sets = {"tydi": [frozenset(toks(t)) for t in tydi], "miracl": [frozenset(toks(t)) for t in miracl]}
    inv = {k: defaultdict(list) for k in sets}
    for k, ss in sets.items():
        for i, s in enumerate(ss):
            for w in s:
                inv[k][w].append(i)

    def hit(k: str, s: frozenset) -> bool:
        c: Counter = Counter()
        for w in s:
            lst = inv[k].get(w, ())
            if len(lst) <= 300:
                c.update(lst)
        return any(n >= 2 and len(s & sets[k][i]) / len(s | sets[k][i]) >= JAC for i, n in c.items())

    out = {}
    for qid, t in qtexts.items():
        s = frozenset(toks(t))
        a, b = bool(s) and hit("tydi", s), bool(s) and hit("miracl", s)
        out[qid] = "both" if a and b else "tydi" if a else "miracl" if b else ""
    return out


def ndcg10(ranked: list[str], rel: set[str]) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    return float(dcg / idcg) if idcg else 0.0


def avg_prec(ranked: list[str], rel: set[str]) -> float:
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / len(rel) if rel else 0.0


def run_model(name: str, path: str, pre: str, miracl, namaa) -> dict:
    import torch

    from arretrieval.retrieval import evalkit as E

    qp, pp = ("query: ", "passage: ") if pre == "e5" else ("", "")
    m = E.Dense(path, name, qp, pp, max_len=512, bs=32 if pre == "none" else 64)
    out = {}
    # ---- MIRACL-HN.v2 ar: full corpus
    qd, cd, rel = miracl
    dids, dtext = list(cd), list(cd.values())
    de = torch.from_numpy(m.encode(dtext, "p")).cuda()
    qids = list(qd)
    qe = torch.from_numpy(m.encode([qd[q] for q in qids], "q")).cuda()
    sc = qe @ de.T
    top = sc.topk(10, dim=1).indices.cpu().numpy()
    out["miracl"] = {q: ndcg10([dids[j] for j in top[i]], {d for d, s in rel.get(q, {}).items() if s > 0}) for i, q in enumerate(qids)}
    del de, sc
    # ---- Namaa: average precision over each query's candidate list
    qd2, cd2, rel2, ranked2 = namaa
    ids2 = list(cd2)
    d2 = torch.from_numpy(m.encode([cd2[d] for d in ids2], "p")).cuda()
    pos = {d: i for i, d in enumerate(ids2)}
    q2ids = list(qd2)
    q2 = torch.from_numpy(m.encode([qd2[q] for q in q2ids], "q")).cuda()
    out["namaa"] = {}
    for i, q in enumerate(q2ids):
        cands = [d for d in ranked2.get(q, []) if d in pos]
        if not cands:
            continue
        s = (q2[i] @ d2[[pos[d] for d in cands]].T).float().cpu().numpy()
        order = [cands[j] for j in np.argsort(-s)]
        out["namaa"][q] = avg_prec(order, {d for d, sc_ in rel2.get(q, {}).items() if sc_ > 0})
    return out


def load_tasks():
    import mteb

    t1 = mteb.get_task("MIRACLRetrievalHardNegatives.v2", hf_subsets=["ar"], eval_splits=["dev"])
    t1.load_data()
    miracl = load_split(t1, "ar", "dev", with_title=True)
    t2 = mteb.get_task("NamaaMrTydiReranking", eval_splits=["test"])
    t2.load_data()
    qd, cd, rel = load_split(t2, "default", "test", with_title=True)
    d = t2.dataset["default"]["test"] if getattr(t2, "dataset", None) else None
    ranked = {}
    if d is not None and "top_ranked" in d:
        ranked = {k: list(v) for k, v in d["top_ranked"].items()}
    elif hasattr(t2, "top_ranked"):
        ranked = {k: list(v) for k, v in t2.top_ranked["default"]["test"].items()}
    return miracl, (qd, cd, rel, ranked)


def report() -> None:
    miracl, namaa = load_tasks()
    labels = {"miracl": contaminated_queries(miracl[0]), "namaa": contaminated_queries(namaa[0])}
    res = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json")) if not p.stem.startswith("_")}
    lines = []
    for task, key in (
        ("MIRACL-HN.v2 ar (nDCG@10 over the full corpus)", "miracl"),
        ("NamaaMrTydi (average precision over the candidate lists)", "namaa"),
    ):
        lab = labels[key]
        cnt = Counter(v or "clean" for v in lab.values())
        lines += [
            f"\n**{task}** - {len(lab)} queries: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + "\n",
            "| model | all | CLEAN | CONTAMINATED (tydi / miracl / both) | clean - contaminated |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, r in res.items():
            sc = r[key]
            qs = [q for q in sc if q in lab]
            allv = np.mean([sc[q] for q in qs]) * 100
            cl = [sc[q] for q in qs if not lab[q]]
            co = [sc[q] for q in qs if lab[q]]
            lines.append(
                f"| {name} | {allv:.1f} | {np.mean(cl) * 100:.1f} (n={len(cl)}) | {np.mean(co) * 100:.1f} (n={len(co)}) | {(np.mean(cl) - np.mean(co)) * 100:+.1f} |"
            )
    (OUT / "_report.md").write_text("\n".join(lines))
    (OUT / "_labels.json").write_text(json.dumps(labels))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.report:
        report()
        return
    miracl, namaa = load_tasks()
    for spec in a.models:
        name, _, rest = spec.partition("=")
        path, _, pre = rest.rpartition(":")
        if (OUT / f"{name}.json").exists():
            print("cached", name)
            continue
        r = run_model(name, path, pre, miracl, namaa)
        (OUT / f"{name}.json").write_text(json.dumps(r))
        print(
            f"[{name}] MIRACL-HN nDCG@10 {np.mean(list(r['miracl'].values())) * 100:.1f} | Namaa AP {np.mean(list(r['namaa'].values())) * 100:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
