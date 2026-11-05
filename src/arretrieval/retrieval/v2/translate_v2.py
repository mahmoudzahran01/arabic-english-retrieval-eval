"""v2 data expansion, step 2: translate the xling2 candidates with the shared Gemma server (same prompt/validation as v1).

Rules (same as v1's translate.py): port 8001 only, concurrency <= 2, thinking disabled, API key read from the key file and
never printed, resumable (appends to runs/retrieval/v2/data/xling2_translations.jsonl). Batches of 20 questions per request
(v1 used 10; 20 per call measured at 4.7 s here).  Dev candidates first, then train in shuffled order, so an early stop still
leaves a random subset of every direction.

Run (no GPU, host venv because the docker bridge cannot reach localhost:8001):
  nohup .venv/bin/python -m arretrieval.retrieval.v2.translate_v2 > runs/retrieval/v2/logs/translate_v2.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from arretrieval.retrieval.paths import DATA
from arretrieval.retrieval.translate import api_key, translate_batch, valid

BATCH = 20  # 50 per request breaks the numbered-line format (95 s and unparseable); 20 = 4.7 s idle, ~10 s under GPU contention
SUB = 10
CONCURRENCY = 2

_TAG = (
    sys.argv[1] if len(sys.argv) > 1 else "xling2"
)  # python -m ...translate_v2 xling3  -> xling3_candidates.jsonl / xling3_translations.jsonl
CAND = DATA.parent / "v2" / "data" / f"{_TAG}_candidates.jsonl"
OUT = DATA.parent / "v2" / "data" / f"{_TAG}_translations.jsonl"


def main() -> None:
    cand = pd.read_json(CAND, lines=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line:
                j = json.loads(line)
                done.add(j["src_qid"] + "|" + j["direction"])
    todo = cand[[f"{a}|{b}" not in done for a, b in zip(cand.src_qid, cand.direction)]]
    print(f"{len(cand)} candidates, {len(todo)} to translate", flush=True)
    key, lock = api_key(), threading.Lock()
    jobs = []
    for split in ("dev", "train"):
        g = todo[todo.split == split].sample(frac=1.0, random_state=13)
        # group by direction inside a batch (the prompt names one source language), interleave directions
        per_dir = {d: g[g.direction == d].to_dict("records") for d in ("en2ar", "ar2en")}
        batches = []
        for d, rows in per_dir.items():
            src, tgt = ("en", "ar") if d == "en2ar" else ("ar", "en")
            batches += [(src, tgt, rows[i : i + BATCH]) for i in range(0, len(rows), BATCH)]
        batches = [batches[i] for i in pd.Series(range(len(batches))).sample(frac=1.0, random_state=13)]
        jobs += batches
    t0, n_done = time.time(), 0

    def work(job) -> None:
        nonlocal n_done
        src, tgt, rows = job
        with httpx.Client() as client:
            outs = translate_batch(client, key, [r["text"] for r in rows], src, tgt)
            if outs is None:  # numbering broke: retry in chunks of SUB, then one by one
                outs = []
                for i in range(0, len(rows), SUB):
                    chunk = [r["text"] for r in rows[i : i + SUB]]
                    o = translate_batch(client, key, chunk, src, tgt)
                    outs += o if o is not None else [(translate_batch(client, key, [t], src, tgt) or [""])[0] for t in chunk]
        with lock:
            with OUT.open("a") as f:
                for r, o in zip(rows, outs):
                    f.write(
                        json.dumps(
                            {"src_qid": r["src_qid"], "direction": r["direction"], "translation": o, "ok": valid(r["text"], o, tgt)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            n_done += len(rows)
            if (n_done // BATCH) % 10 == 0:
                print(f"{n_done}/{len(todo)} translated, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(CONCURRENCY) as ex:
        list(ex.map(work, jobs))
    print("done", flush=True)


if __name__ == "__main__":
    main()
