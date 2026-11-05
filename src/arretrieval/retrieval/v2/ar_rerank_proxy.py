"""v2.1 validation component 7: "ar_rerank" - a Namaa-like Arabic reranking proxy built from NON-MTEB data (pre-registered in PROGRESS.md, 2026-11-05 ~08:50).

NamaaMrTydiReranking ranks a short candidate list (relevant vs non-relevant Arabic Wikipedia passages) per Arabic question and reports average precision.  The proxy does the same on the
438 held-out MIRACL-ar TRAIN queries (`devM`, never trained on): the candidates of a query are its annotator-JUDGED passages (judged relevant = 1, judged NOT relevant = 0) and the metric is the
mean average precision of the model's cosine ranking.  Queries without at least one relevant and one non-relevant judged passage in the held-out pool are skipped.
The proxy is only USED for selection if it passes the pre-registered validity gate (arr_validate.py: Spearman rho >= 0.70 against the known NamaaMrTydi scores of existing models).

Run (GPU, behind the gate):  python -m arretrieval.retrieval.v2.ar_rerank_proxy --models NAME=PATH:e5|none ...          results: results/arr/NAME.json (cached per model)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from arretrieval.retrieval import evalkit as E

from .common_v2 import V2DATA, V2RES, load_passages_v2, load_queries_v2

OUT = V2RES / "arr"
CAND = V2DATA / "arr_candidates.json"


def avg_prec(scores: np.ndarray, rel: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    r = rel[order]
    hits = np.cumsum(r)
    return float((hits[r > 0] / (np.flatnonzero(r > 0) + 1)).sum() / max(1, r.sum()))


def build() -> dict:
    """{qid: {"text": q, "cands": {pid: 0|1}}} for the hold-out queries, from the MIRACL-ar train qrels (judged passages that exist in the hold-out pool)."""
    if CAND.exists():
        return json.loads(CAND.read_text())
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("miracl/miracl", "miracl-v1.0-ar/qrels/qrels.miracl-v1.0-ar-train.tsv", repo_type="dataset")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for ln in open(p, encoding="utf-8"):
        r = ln.rstrip("\n").split("\t")
        if len(r) >= 4:
            qrels[r[0]][r[2]] = int(r[3])
    P = load_passages_v2()
    hold_pids = set(P[(P.pool == "devM") & (P.source == "miracl_ar_holdout")].pid)
    Q = load_queries_v2()
    out = {}
    for r in Q[Q.split == "devM"].itertuples():
        orig = r.qid.removeprefix("miracl_ar-hold-")
        cands = {f"miracl-ar-{d}": int(s > 0) for d, s in qrels.get(orig, {}).items() if f"miracl-ar-{d}" in hold_pids}
        if sum(cands.values()) >= 1 and sum(1 - v for v in cands.values()) >= 1:
            out[r.qid] = {"text": r.text, "cands": cands}
    CAND.write_text(json.dumps(out, ensure_ascii=False))
    return out


def run_model(m: E.Dense, d: dict, text: dict[str, str]) -> dict:
    pids = sorted({p for v in d.values() for p in v["cands"]})
    pe = dict(zip(pids, m.encode([text[p] for p in pids], "p")))
    qids = list(d)
    qe = m.encode([d[q]["text"] for q in qids], "q")
    per = {}
    for i, q in enumerate(qids):
        ps = list(d[q]["cands"])
        sc = np.stack([pe[p] for p in ps]) @ qe[i]
        per[q] = avg_prec(sc, np.array([d[q]["cands"][p] for p in ps]))
    return {"ap": float(np.mean(list(per.values()))), "n_queries": len(per), "per_query": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    P = load_passages_v2()
    text = dict(zip(P.pid, P.text))
    print(f"ar_rerank proxy: {len(d)} queries, {sum(len(v['cands']) for v in d.values())} judged candidates", flush=True)
    for spec in a.models:
        name, _, rest = spec.partition("=")
        path, _, pre = rest.rpartition(":")
        if (OUT / f"{name}.json").exists():
            print("cached", name, flush=True)
            continue
        qp, pp = ("query: ", "passage: ") if pre == "e5" else ("", "")
        m = E.Dense(path, name, qp, pp, max_len=512, bs=32 if pre != "e5" else 64)
        r = run_model(m, d, text)
        (OUT / f"{name}.json").write_text(json.dumps(r))
        print(f"[{name}] ar_rerank AP {r['ap'] * 100:.2f} ({r['n_queries']} queries)", flush=True)
        del m


if __name__ == "__main__":
    main()
