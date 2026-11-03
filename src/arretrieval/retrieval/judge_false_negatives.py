"""Estimate the false-negative rate of mined hard negatives with an LLM judge (OpenAI-compatible server, see translate.py).

For a sample of train queries we take the baseline retriever's non-gold candidates in three rank bands (1-3, 4-10, 11-30)
and ask the judge "does this passage contain the answer to the question?".  A "yes" on a passage that the dataset labels
as non-gold is a *probable false negative*.  We also judge the gold passages (calibration: how often does the judge agree
that the labelled positive answers the question?) so the numbers can be read as lower bounds, not truth.

Concurrency 2, thinking disabled.  No GPU needed:
  python -m arretrieval.retrieval.judge_false_negatives --model e5-small
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from .paths import DATA, MINED, RESULTS
from .translate import MODEL, URL, api_key

BANDS = {"rank 1-3": (1, 3), "rank 4-10": (4, 10), "rank 11-30": (11, 30)}
OUT_FILE = RESULTS / "fn_judge.jsonl"


def ask(client: httpx.Client, key: str, question: str, passage: str) -> str:
    body = {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 4,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "system",
                "content": "You judge whether a passage contains the answer to a question. Reply with exactly one word: yes or no.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassage: {passage[:1500]}\n\nDoes the passage contain the answer to the question?",
            },
        ],
    }
    for attempt in range(3):
        try:
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip().lower()
        except (httpx.HTTPError, KeyError):
            time.sleep(2 * (attempt + 1))
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small")
    ap.add_argument("--per-lang", type=int, default=80)
    args = ap.parse_args()
    raw = pd.read_parquet(MINED / f"raw_{args.model}.parquet")
    Q = pd.read_parquet(DATA / "queries_mono.parquet").set_index("qid")
    P = pd.read_parquet(DATA / "passages.parquet").set_index("pid")
    rng = random.Random(13)
    jobs = []
    for lang in ("ar", "en"):
        sub = raw[[Q.loc[q, "qlang"] == lang and Q.loc[q, "plang"] == lang and "_xl_" not in Q.loc[q, "slice"] for q in raw.qid]]
        for r in sub.sample(args.per_lang, random_state=13).itertuples():
            q = Q.loc[r.qid, "text"]
            jobs.append(
                {
                    "qid": r.qid,
                    "pid": r.pos_pid,
                    "kind": "gold",
                    "band": "gold",
                    "gap": 0.0,
                    "q": q,
                    "p": P.loc[r.pos_pid, "text"],
                    "lang": lang,
                }
            )
            negs = [(pid, sc) for pid, sc in zip(r.cand_pids, r.cand_scores) if pid != r.pos_pid]
            for band, (lo, hi) in BANDS.items():
                pid, sc = rng.choice(negs[lo - 1 : hi])
                jobs.append(
                    {
                        "qid": r.qid,
                        "pid": pid,
                        "kind": "neg",
                        "band": band,
                        "gap": round(r.pos_score - sc, 4),
                        "q": q,
                        "p": P.loc[pid, "text"],
                        "lang": lang,
                    }
                )
    key, lock, done = api_key(), threading.Lock(), 0
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("")

    def work(j: dict) -> None:
        nonlocal done
        with httpx.Client() as c:
            ans = ask(c, key, j["q"], j["p"])
        with lock:
            with OUT_FILE.open("a") as f:
                f.write(
                    json.dumps({k: j[k] for k in ("qid", "pid", "kind", "band", "gap", "lang")} | {"answer": ans}, ensure_ascii=False)
                    + "\n"
                )
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(work, jobs))
    print("done", len(jobs))


if __name__ == "__main__":
    main()
