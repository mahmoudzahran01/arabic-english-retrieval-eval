"""Data v3, step 1: more English QA from a clearly licensed source - SQuAD v1.1 TRAIN (CC BY-SA 4.0) - in articles that were never in any pool.

v1/v2 used the SQuAD-train articles whose title hash fell in the train pool (20% of titles); another 10% are the test background and 4% dev, 62% were
never used.  v3 adds those unused articles: their paragraphs join the TRAIN pool (as extra passages and as candidate negatives), 2 questions per paragraph
(like v1) become English training queries and are machine-translated into Arabic (Arabic query -> English passage).
Safety: SQuAD-dev/XQuAD titles and every title of a test source (build_data.build_test) are excluded; passages whose text already exists in any pool are skipped;
overlap with the MTEB(ara) evaluation sets is removed afterwards by contam_v2 (same filter as v2).
Outputs (runs/retrieval/v2/data): passages_extra.parquet, queries_extra_mono.parquet, xling3_candidates.jsonl.
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.build_cands_v3
"""

from __future__ import annotations

import json
import random

import pandas as pd

from arretrieval.retrieval import build_data as B
from arretrieval.retrieval.paths import DATA, SEED
from arretrieval.retrieval.textnorm import dedupe_key

from .common_v2 import V2DATA

PER_CTX = 2


def main() -> None:
    src = B.load_sources()
    reg = B.Registry()
    test_titles = B.build_test(reg, src)
    P = pd.read_parquet(DATA / "passages.parquet")
    known = set(P.pid)
    Q = pd.read_parquet(DATA / "queries.parquet")
    have_q = {(r.qlang, dedupe_key(r.text)) for r in Q.itertuples()}
    rng = random.Random(SEED + 3)
    by_ctx: dict[str, list] = {}
    n_title_test = n_known = n_pool = 0
    for r in src["squad_train"].itertuples():
        if dedupe_key(r.title) in test_titles["en"]:
            n_title_test += 1
            continue
        if B.title_pool("squad_train", r.title) != "unused":
            n_pool += 1
            continue
        pid = B.pid_of("en", r.context)
        if pid in known:
            n_known += 1
            continue
        if ("en", dedupe_key(r.question)) in have_q:
            continue
        by_ctx.setdefault(pid, []).append(r)
    passages, queries, cands = [], [], []
    for pid, rows in by_ctx.items():
        rng.shuffle(rows)
        r0 = rows[0]
        passages.append(
            {
                "pid": pid,
                "lang": "en",
                "text": r0.context.strip(),
                "title": r0.title,
                "source": "squad_train_unused",
                "pool": "train",
                "kind": "gold",
            }
        )
        for r in rows[:PER_CTX]:
            qid = f"squad_en-xtrain-{r.id}"
            queries.append(
                {
                    "qid": qid,
                    "text": r.question.strip(),
                    "qlang": "en",
                    "pos_pid": pid,
                    "plang": "en",
                    "split": "train",
                    "slice": "train_squad_en_x",
                    "source": "squad_train",
                    "title": r.title,
                    "src_qid": "",
                }
            )
            cands.append(
                {
                    "src_qid": qid,
                    "split": "train",
                    "text": r.question.strip(),
                    "qlang": "en",
                    "pos_pid": pid,
                    "plang": "en",
                    "slice": "squad_en_x",
                    "direction": "en2ar",
                }
            )
    pd.DataFrame(passages).to_parquet(V2DATA / "passages_extra.parquet", index=False)
    pd.DataFrame(queries).to_parquet(V2DATA / "queries_extra_mono.parquet", index=False)
    pd.DataFrame(cands).to_json(V2DATA / "xling3_candidates.jsonl", orient="records", lines=True, force_ascii=False)
    stats = {
        "new_passages": len(passages),
        "new_mono_queries": len(queries),
        "translation_candidates": len(cands),
        "skipped_rows_title_in_test_sources": n_title_test,
        "skipped_rows_other_pool": n_pool,
        "skipped_rows_passage_text_already_in_a_pool": n_known,
    }
    (V2DATA / "manifest_v3_extra.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
