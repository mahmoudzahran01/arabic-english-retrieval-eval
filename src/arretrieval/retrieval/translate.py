"""Stage 2: machine-translate selected real questions with an LLM (cross-lingual train/dev queries).

Cross-lingual TEST queries are NOT produced here (they are XQuAD's human translations). This script exists only
because no public ar->en passage-retrieval *training* set with permissive licence loads as plain parquet.

The LLM is any OpenAI-compatible chat server configured through the environment (`GEMMA_BASE_URL`, `GEMMA_API_KEY`,
`GEMMA_MODEL`, see .env.example; the published run used Gemma-4-26B-A4B). Concurrency is capped at 2, thinking is
disabled through `chat_template_kwargs` (a vLLM / llama.cpp extension), temperature 0.
Resumable: results are appended to runs/retrieval/data/xling_translations.jsonl.
Run (no GPU needed): python -m arretrieval.retrieval.translate
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from ..config import endpoint
from .paths import DATA

_EP = endpoint("gemma")
URL = f"{_EP.base_url}/chat/completions"
MODEL = _EP.model
BATCH = 10
CONCURRENCY = 2
LANG_NAME = {"ar": "Modern Standard Arabic", "en": "English"}
OUT = DATA / "xling_translations.jsonl"
_AR = re.compile(r"[؀-ۿ]")
_LINE = re.compile(r"^\s*(\d+)\s*[\.\)\-:]\s*(.+?)\s*$")


def api_key() -> str:
    return _EP.api_key


def translate_batch(client: httpx.Client, key: str, texts: list[str], src: str, tgt: str) -> list[str] | None:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    system = (
        f"You are a professional translator. Translate each numbered question from {LANG_NAME[src]} to {LANG_NAME[tgt]}. "
        "Keep names and numbers faithful, write digits as 0-9, and do NOT answer the questions. "
        "Output exactly the same numbering, one translated question per line, and nothing else."
    )
    body = {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 80 * len(texts),
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": numbered}],
    }
    for attempt in range(3):
        try:
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=180)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            found = {int(m.group(1)): m.group(2) for line in content.splitlines() if (m := _LINE.match(line))}
            if all(i + 1 in found for i in range(len(texts))):
                return [found[i + 1] for i in range(len(texts))]
        except (httpx.HTTPError, KeyError, ValueError):
            time.sleep(2 * (attempt + 1))
    return None


def valid(src_text: str, out: str, tgt: str) -> bool:
    has_ar = bool(_AR.search(out))
    return bool(out.strip()) and out.strip() != src_text.strip() and (has_ar if tgt == "ar" else not has_ar)


def main() -> None:
    cand = pd.read_json(DATA / "xling_candidates.jsonl", lines=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["src_qid"] + "|" + json.loads(l)["direction"] for l in OUT.read_text().splitlines() if l}
    todo = cand[[f"{a}|{b}" not in done for a, b in zip(cand.src_qid, cand.direction)]]
    print(f"{len(cand)} candidates, {len(todo)} to translate", flush=True)
    key, lock = api_key(), threading.Lock()
    jobs = []  # en2ar first (ar query -> en passage is the production-relevant direction); rows shuffled so that
    for direction in ("en2ar", "ar2en"):  # stopping early still leaves a random subset of train AND dev
        g = todo[todo.direction == direction].sample(frac=1.0, random_state=13)
        src, tgt = ("en", "ar") if direction == "en2ar" else ("ar", "en")
        rows = g.to_dict("records")
        jobs += [(src, tgt, rows[i : i + BATCH]) for i in range(0, len(rows), BATCH)]
    t0, n_done = time.time(), 0

    def work(job) -> None:
        nonlocal n_done
        src, tgt, rows = job
        with httpx.Client() as client:
            outs = translate_batch(client, key, [r["text"] for r in rows], src, tgt)
            if outs is None:  # fall back to one-by-one
                outs = [(translate_batch(client, key, [r["text"]], src, tgt) or [""])[0] for r in rows]
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
