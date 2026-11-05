"""Data v3, step 3: merge the extra SQuAD-train questions + their Gemma translations into queries_v3.parquet / passages_v3.parquet.

queries_v3 = queries_v2 (unchanged rows) + queries_extra_mono (slice train_squad_en_x) + translated Arabic queries (slice train_xl3_ar2en).
passages_v3 = passages + passages_extra (TRAIN pool only; dev/test pools untouched, so every evaluation is identical).
Exact duplicates of a question owned by a higher-priority split are dropped; near-duplicates (Jaccard >= 0.8) of a TEST question are dropped as in data_v2.
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.data_v3
"""

from __future__ import annotations

import json

import pandas as pd

from arretrieval.retrieval.paths import DATA
from arretrieval.retrieval.textnorm import dedupe_key

from .common_v2 import V2DATA
from .data_v2 import near_dups


def main() -> None:
    q2 = pd.read_parquet(V2DATA / "queries_v2.parquet")
    pe = pd.read_parquet(V2DATA / "passages_extra.parquet")
    mono = pd.read_parquet(V2DATA / "queries_extra_mono.parquet")
    cand = pd.read_json(V2DATA / "xling3_candidates.jsonl", lines=True)
    tr = pd.read_json(V2DATA / "xling3_translations.jsonl", lines=True).drop_duplicates(["src_qid", "direction"], keep="last")
    df = cand.merge(tr, on=["src_qid", "direction"], how="inner")
    stats = {
        "extra_passages": len(pe),
        "extra_mono_queries": len(mono),
        "candidates": len(cand),
        "translated": len(df),
        "invalid": int((~df.ok.astype(bool)).sum()),
    }
    df = df[df.ok.astype(bool)]
    owner = {(r.qlang, dedupe_key(r.text)): 0 if r.split == "test" else 1 if r.split == "dev" else 2 for r in q2.itertuples()}
    rows, dup = [], 0
    for r in df.itertuples():
        key = ("ar", dedupe_key(r.translation))
        if owner.get(key, 99) < 2:
            dup += 1
            continue
        owner[key] = 2
        rows.append(
            {
                "qid": f"xl3-{r.src_qid}",
                "text": r.translation.strip(),
                "qlang": "ar",
                "pos_pid": r.pos_pid,
                "plang": "en",
                "split": "train",
                "slice": "train_xl3_ar2en",
                "source": "gemma_translation_v3",
                "title": "",
                "src_qid": r.src_qid,
            }
        )
    new = pd.concat([mono, pd.DataFrame(rows)], ignore_index=True)
    nd = near_dups(new, q2[q2.split == "test"])
    stats["dropped_exact_duplicate"] = dup
    stats["dropped_near_duplicate_of_test_question"] = len(nd)
    new = new[~new.qid.isin(nd)]
    q = pd.concat([q2, new], ignore_index=True)
    P = pd.read_parquet(DATA / "passages.parquet")
    P3 = pd.concat([P, pe], ignore_index=True)
    assert P3.pid.is_unique and q.qid.is_unique
    pool_of = dict(zip(P3.pid, P3.pool))
    assert all(pool_of[p] == s for p, s in zip(q.pos_pid, q.split)), "gold passage outside split pool"
    q.to_parquet(V2DATA / "queries_v3.parquet", index=False)
    P3.to_parquet(V2DATA / "passages_v3.parquet", index=False)
    stats["queries_by_slice"] = {f"{a}/{b}": int(v) for (a, b), v in q.groupby(["split", "slice"]).size().items() if a == "train"}
    (V2DATA / "manifest_v3.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
