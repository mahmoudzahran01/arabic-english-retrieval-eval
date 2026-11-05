"""v2 data expansion, step 1: choose which real questions to machine-translate (train pool + dev pool only).

v1 translated 3,000 English questions -> Arabic (Arabic query, English passage) and 1,500 Arabic questions -> English.
v2 translates (almost) every remaining question whose gold passage is in the TRAIN pool (SQuAD-train rows beyond the
2-per-paragraph cap, the rest of TyDi/ARCD-ar) and a larger DEV sample, all with the same Gemma prompt as v1.
Nothing here touches the test pool: a candidate is kept only if its gold passage lives in the pool of its split.

Run: NO_GPU=1 arretrieval.retrieval.v2.build_cands_v2
"""

from __future__ import annotations

import json
import random

import pandas as pd

from arretrieval.retrieval import build_data as B
from arretrieval.retrieval.paths import DATA, SEED
from arretrieval.retrieval.textnorm import dedupe_key

OUT = DATA.parent / "v2" / "data" / "xling2_candidates.jsonl"
N_DEV_AR2EN_Q, N_DEV_EN2AR_Q = 700, 600  # new dev source questions (disjoint from the 300 / 200 translated dev questions of v1)


def main() -> None:
    src = B.load_sources()
    P = pd.read_parquet(DATA / "passages.parquet")
    Q = pd.read_parquet(DATA / "queries.parquet")
    pool = dict(zip(P.pid, P.pool))
    mono = Q[~Q.slice.str.contains("_xl_")]
    qid_by_key = {(r.qlang, dedupe_key(r.text)): r.qid for r in mono.itertuples()}
    done_v1_src = set(Q[Q.slice.str.contains("_xl_")].src_qid)
    all_text_keys = {(r.qlang, dedupe_key(r.text)) for r in Q.itertuples()}
    rows: list[dict] = []
    plan = (
        ("squad_train", "en", "squad_en"),
        ("tydi_en_train", "en", "tydi_en"),
        ("tydi_ar_train", "ar", "tydi_ar"),
        ("arcd_train", "ar", "arcd_ar"),
    )
    for source, lang, slice_ in plan:
        for r in src[source].itertuples():
            pid = B.pid_of(lang, r.context)
            p = pool.get(pid)
            if p not in ("train", "dev"):
                continue
            key = (lang, dedupe_key(r.question))
            qid = qid_by_key.get(key)
            if qid is None:  # a question v1 never used (beyond the 2-per-paragraph cap)
                if key in all_text_keys:
                    continue  # same text owned by another split/slice
                qid = f"{slice_}-{p}-{r.id}"
            if qid in done_v1_src:
                continue
            rows.append(
                {
                    "src_qid": qid,
                    "split": p,
                    "text": r.question.strip(),
                    "qlang": lang,
                    "pos_pid": pid,
                    "plang": lang,
                    "slice": slice_,
                    "direction": "en2ar" if lang == "en" else "ar2en",
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(["src_qid", "direction"])
    rng = random.Random(SEED + 2)

    def sample_dev(direction: str, n: int) -> pd.DataFrame:
        d = df[(df.split == "dev") & (df.direction == direction)]
        return d.sample(n=min(n, len(d)), random_state=rng.randrange(10**6))

    dev = pd.concat([sample_dev("en2ar", N_DEV_AR2EN_Q), sample_dev("ar2en", N_DEV_EN2AR_Q)])
    train = df[df.split == "train"]
    out = pd.concat([dev, train]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(OUT, orient="records", lines=True, force_ascii=False)
    print(json.dumps({f"{a}/{b}": int(n) for (a, b), n in out.groupby(["split", "direction"]).size().items()}, indent=1))


if __name__ == "__main__":
    main()
