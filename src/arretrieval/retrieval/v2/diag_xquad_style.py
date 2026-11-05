"""Diagnostic: how much of the dev -> test gap in ar->en is translation STYLE?

Dev/train cross-lingual queries are Gemma translations, the TEST ones are XQuAD's human translations.  Here the English XQuAD
questions are translated by Gemma too (same prompt, same batching) and every model is scored on both versions of the same 1,190
questions against the same English test pool and the same gold paragraphs.  Nothing here is used for training or selection.

  python -m arretrieval.retrieval.v2.diag_xquad_style translate        (host venv, Gemma on :8001, concurrency 2)
  python -m arretrieval.retrieval.v2.diag_xquad_style eval --models NAME=PATH[:e5|none] ...
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval import metrics as M
from arretrieval.retrieval.translate import api_key, translate_batch, valid

from .common_v2 import V2DATA, V2RES, load_queries_v2

OUT = V2DATA / "xquad_gemma_ar2en.jsonl"


def do_translate() -> None:
    q = load_queries_v2()
    en = q[q.slice == "xquad_en2en"].reset_index(drop=True)
    done = {json.loads(l)["src_qid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    rows = [r for r in en.to_dict("records") if r["src_qid"] not in done]
    key, lock = api_key(), threading.Lock()
    jobs = [rows[i : i + 20] for i in range(0, len(rows), 20)]

    def work(chunk) -> None:
        with httpx.Client() as c:
            outs = translate_batch(c, key, [r["text"] for r in chunk], "en", "ar") or [
                (translate_batch(c, key, [r["text"]], "en", "ar") or [""])[0] for r in chunk
            ]
        with lock, OUT.open("a") as f:
            for r, o in zip(chunk, outs):
                f.write(json.dumps({"src_qid": r["src_qid"], "text": o, "ok": valid(r["text"], o, "ar")}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(work, jobs))
    print("translated", len(rows))


def do_eval(models: list[str]) -> None:
    passages, q = E.load_passages(), load_queries_v2()
    human = q[q.slice == "xquad_ar2en"].reset_index(drop=True)
    gem = pd.read_json(OUT, lines=True).drop_duplicates("src_qid")
    gem = human[["qid", "src_qid", "pos_pid", "plang"]].merge(gem[gem.ok][["src_qid", "text"]], on="src_qid")
    human = human[human.src_qid.isin(set(gem.src_qid))]
    corpus = E.pool_corpus(passages, "test", "en")
    res = {}
    for spec in models:
        name, _, rest = spec.partition("=")
        path, _, pre = rest.partition(":")
        qp, pp = ("query: ", "passage: ") if pre != "none" else ("", "")
        d = E.Dense(path, name, qp, pp, max_len=512, bs=64 if pre != "none" else 32)
        out = {}
        for lab, df in (("human", human), ("gemma", gem)):
            run = d.rank(df, corpus)
            qrels = {q_: {p} for q_, p in zip(df.qid, df.pos_pid)}
            pq = M.evaluate_run({k: v[0] for k, v in run.items()}, qrels)
            out[lab] = {"n": len(df), "mrr@10": float(pq["mrr@10"].mean()), "hit@10": float(pq["hit@10"].mean())}
        res[name] = out
        print(
            f"[{name}] human-translated ar->en MRR@10 {out['human']['mrr@10'] * 100:.1f} | Gemma-translated {out['gemma']['mrr@10'] * 100:.1f} (n={out['human']['n']})",
            flush=True,
        )
        del d
    (V2RES / "diag_xquad_style.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["translate", "eval"])
    ap.add_argument("--models", nargs="*", default=[])
    a = ap.parse_args()
    do_translate() if a.cmd == "translate" else do_eval(a.models)
