"""Stage 2b: merge the LLM translations into the query table -> queries.parquet (+ manifest.json).

A translated query inherits the gold passage of the real question it was translated from, so:
  src en question (en passage) --translate--> ar query, en passage   => slice *_xl_ar2en
  src ar question (ar passage) --translate--> en query, ar passage   => slice *_xl_en2ar
Translations that failed validation, or that exactly duplicate a question owned by a higher-priority split, are dropped.
Run: python -m arretrieval.retrieval.finalize_data
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .paths import CARD, DATA
from .textnorm import dedupe_key

PRIORITY = {"test": 0, "dev": 1, "train": 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="queries.parquet",
        help="use another name (e.g. queries_preview.parquet) to test without changing what the eval harness loads",
    )
    args = ap.parse_args()
    mono = pd.read_parquet(DATA / "queries_mono.parquet")
    cand = pd.read_json(DATA / "xling_candidates.jsonl", lines=True)
    tr = pd.read_json(DATA / "xling_translations.jsonl", lines=True).drop_duplicates(["src_qid", "direction"], keep="last")
    df = cand.merge(tr, on=["src_qid", "direction"], how="left")
    stats = {"candidates": len(df), "missing": int(df.translation.isna().sum()), "invalid": int((df.ok == False).sum())}  # noqa: E712
    df = df[df.ok == True]  # noqa: E712
    owner = {(r.qlang, dedupe_key(r.text)): PRIORITY[r.split] for r in mono.itertuples()}
    rows, dup = [], 0
    for r in df.itertuples():
        qlang = "ar" if r.direction == "en2ar" else "en"
        key = (qlang, dedupe_key(r.translation))
        if owner.get(key, 99) < PRIORITY[r.split]:
            dup += 1
            continue
        owner[key] = min(owner.get(key, 99), PRIORITY[r.split])
        rows.append(
            {
                "qid": f"xl-{r.src_qid}",
                "text": r.translation.strip(),
                "qlang": qlang,
                "pos_pid": r.pos_pid,
                "plang": r.plang,
                "split": r.split,
                "slice": f"{r.split}_xl_{qlang}2{r.plang}",
                "source": "gemma_translation",
                "title": "",
                "src_qid": r.src_qid,
            }
        )
    stats["dropped_duplicate_of_higher_priority_split"] = dup
    q = pd.concat([mono, pd.DataFrame(rows)], ignore_index=True)
    passages = pd.read_parquet(DATA / "passages.parquet")
    pool_of = dict(zip(passages.pid, passages.pool))
    assert all(pool_of[p] == s for p, s in zip(q.pos_pid, q.split)), "gold passage outside split pool"
    assert q.qid.is_unique
    q.to_parquet(DATA / args.out, index=False)
    manifest = json.loads((DATA / "manifest_mono.json").read_text())
    manifest["translation_stats"] = stats
    manifest["queries_by_slice"] = {f"{a}/{b}": int(v) for (a, b), v in q.groupby(["split", "slice"]).size().items()}
    manifest["queries_by_split"] = {k: int(v) for k, v in q.groupby("split").size().items()}
    manifest["passages_total"] = len(passages)
    manifest["passages_by_pool_kind"] = {f"{a}/{b}": int(v) for (a, b), v in passages.groupby(["pool", "kind"]).size().items()}
    (CARD / ("manifest.json" if args.out == "queries.parquet" else "manifest_preview.json")).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    # id-only index of the test queries: lets the report tools recompute leakage sensitivity without the dataset text
    q[q.split == "test"][["qid", "slice", "pos_pid"]].sort_values("qid").to_csv(CARD / "test_query_index.csv", index=False)
    print(json.dumps({k: manifest[k] for k in ("translation_stats", "queries_by_split", "queries_by_slice", "passages_total")}, indent=1))


if __name__ == "__main__":
    main()
