"""v2 data, step 3: merge the extra Gemma translations into a v2 query table (queries_v2.parquet).

* train rows  -> slice train_xl2_{ar2en|en2ar}   (v1's train_xl_* rows are kept untouched)
* dev rows    -> slice dev_xl_{ar2en|en2ar}      (SAME slice names as v1: the v1 300 / 200 translated dev queries are
                                                  extended to 1,000 / 800 so that the dev ar->en and en->ar numbers are less noisy)
* the same exact-duplicate rule as v1 (a translation identical to a question of a higher-priority split is dropped),
  plus a near-duplicate audit: any new TRAIN/DEV translation whose token-set Jaccard with a TEST question of the same
  language is >= 0.8 is dropped (the audit result is written to the manifest).

Run: NO_GPU=1 arretrieval.retrieval.v2.data_v2
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd

from arretrieval.retrieval.paths import DATA
from arretrieval.retrieval.textnorm import dedupe_key, tokenize

V2 = DATA.parent / "v2" / "data"
PRIORITY = {"test": 0, "dev": 1, "train": 2}
JACCARD = 0.8


def near_dups(new: pd.DataFrame, test: pd.DataFrame) -> set[str]:
    """qids of `new` whose normalized token set has Jaccard >= JACCARD with a test question of the same language."""
    drop: set[str] = set()
    for lang in ("ar", "en"):
        t = test[test.qlang == lang]
        sets = [frozenset(tokenize(x, "norm")) for x in t.text]
        inv: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(sets):
            for tok in s:
                inv[tok].append(i)
        for r in new[new.qlang == lang].itertuples():
            s = frozenset(tokenize(r.text, "norm"))
            if not s:
                continue
            c: Counter = Counter()
            for tok in s:
                lst = inv.get(tok, ())
                if len(lst) <= 300:  # skip stop-word-like tokens
                    c.update(lst)
            for i, n in c.items():
                if n >= 2 and len(s & sets[i]) / len(s | sets[i]) >= JACCARD:
                    drop.add(r.qid)
                    break
    return drop


def main() -> None:
    q1 = pd.read_parquet(DATA / "queries.parquet")
    cand = pd.read_json(V2 / "xling2_candidates.jsonl", lines=True)
    tr = pd.read_json(V2 / "xling2_translations.jsonl", lines=True).drop_duplicates(["src_qid", "direction"], keep="last")
    df = cand.merge(tr, on=["src_qid", "direction"], how="inner")
    stats = {"candidates": len(cand), "translated": len(df), "invalid": int((df.ok == False).sum())}  # noqa: E712
    df = df[df.ok == True]  # noqa: E712
    owner = {(r.qlang, dedupe_key(r.text)): PRIORITY[r.split] for r in q1.itertuples()}
    rows, dup = [], 0
    for r in df.itertuples():
        qlang = "ar" if r.direction == "en2ar" else "en"
        key = (qlang, dedupe_key(r.translation))
        if owner.get(key, 99) < PRIORITY[r.split]:
            dup += 1
            continue
        owner[key] = min(owner.get(key, 99), PRIORITY[r.split])
        sl = f"dev_xl_{qlang}2{r.plang}" if r.split == "dev" else f"train_xl2_{qlang}2{r.plang}"
        rows.append(
            {
                "qid": f"xl2-{r.src_qid}",
                "text": r.translation.strip(),
                "qlang": qlang,
                "pos_pid": r.pos_pid,
                "plang": r.plang,
                "split": r.split,
                "slice": sl,
                "source": "gemma_translation_v2",
                "title": "",
                "src_qid": r.src_qid,
            }
        )
    new = pd.DataFrame(rows)
    nd = near_dups(new, q1[q1.split == "test"])
    stats["dropped_exact_duplicate_of_higher_priority_split"] = dup
    stats["dropped_near_duplicate_of_test_question(jaccard>=0.8)"] = len(nd)
    new = new[~new.qid.isin(nd)]
    q = pd.concat([q1, new], ignore_index=True)
    passages = pd.read_parquet(DATA / "passages.parquet")
    pool_of = dict(zip(passages.pid, passages.pool))
    assert all(pool_of[p] == s for p, s in zip(q.pos_pid, q.split)), "gold passage outside split pool"
    assert q.qid.is_unique
    q.to_parquet(V2 / "queries_v2.parquet", index=False)
    stats["queries_by_slice"] = {f"{a}/{b}": int(v) for (a, b), v in q.groupby(["split", "slice"]).size().items()}
    (V2 / "manifest_v2.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
