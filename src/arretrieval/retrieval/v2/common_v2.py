"""Paths and model table shared by the v2 scripts (v1 code in lab/retrieval is imported, never modified)."""

from __future__ import annotations

import pandas as pd

from arretrieval.retrieval.paths import OUT

V2 = OUT / "v2"
V2DATA, V2RES, V2MODELS, V2MINED, V2LOGS = V2 / "data", V2 / "results", V2 / "models", V2 / "mined", V2 / "logs"
for _p in (V2DATA, V2RES, V2MODELS, V2MINED, V2LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# tag -> (hf id, query prefix, passage prefix)
BASES = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),  # v2-m3: the 568M teacher/baseline fine-tuned itself (no prefixes)
}
TEACHER = ("BAAI/bge-m3", "", "")


def load_queries_v2() -> pd.DataFrame:
    """queries_v3.parquet (= queries_v2 + extra SQuAD-train questions and their translations) if it exists, else queries_v2.parquet.
    Dev/test rows are identical in both, so every evaluation is unaffected; training scripts choose their rows with --data v1|v2|v3."""
    p = V2DATA / "queries_v3.parquet"
    q = pd.read_parquet(p if p.exists() else V2DATA / "queries_v2.parquet")
    m = V2DATA / "queries_miracl.parquet"  # MIRACL-ar TRAIN replay (slice train_miracl_ar) + the held-out selection slice (split devM)
    return pd.concat([q, pd.read_parquet(m)], ignore_index=True) if m.exists() else q


def load_passages_v2() -> pd.DataFrame:
    """passages.parquet (+ the extra TRAIN-pool passages of data v3, passages_v3.parquet, if built)."""
    from arretrieval.retrieval import evalkit as E

    p = V2DATA / "passages_v3.parquet"
    P = pd.read_parquet(p) if p.exists() else E.load_passages()
    m = V2DATA / "passages_miracl.parquet"
    return pd.concat([P, pd.read_parquet(m)], ignore_index=True) if m.exists() else P


def data_family(slice_: pd.Series) -> pd.Series:
    """'v1' | 'v2' | 'v3' : the data generation a TRAIN row belongs to (v2 = + train_xl2_*, v3 = + extra SQuAD articles train_squad_en_x / train_xl3_*)."""
    fam = pd.Series("v1", index=slice_.index)
    fam[slice_.str.contains("_xl2_")] = "v2"
    fam[slice_.str.contains("_xl3_") | slice_.str.endswith("_squad_en_x")] = "v3"
    fam[slice_ == "train_miracl_ar"] = "replay"
    return fam
