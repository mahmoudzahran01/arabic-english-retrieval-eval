"""Hard-negative mining.

mine_raw():   for every TRAIN query retrieve the top-K passages of the query's gold-language TRAIN pool with a first-stage
              model and record scores, the gold passage's own score and rank. Nothing is thrown away here.
select():     turn the raw candidates into a per-query negative list under an explicit false-negative policy:
                - always drop the gold passage
                - rank window [rank_lo, rank_hi]      (skipping the very top ranks removes the most suspicious candidates)
                - score margin: drop candidates with  score > pos_score - margin   (they score "as well as the gold",
                  so they are more likely to be unlabeled positives than true negatives)
                - near-duplicate of the gold (token Jaccard >= dup_jaccard), e.g. the same paragraph from two sources
                - same article as the gold (optional; kills most "sibling paragraph" false negatives, and many good negatives)
              then keep the n_neg highest-scoring survivors ("hardest first") or a random sample of them.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import evalkit as E
from .paths import MINED
from .textnorm import tokenize


def mine_raw(model: E.Dense, queries: pd.DataFrame, passages: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """Rows: qid, pos_pid, pos_score, pos_rank (1-based, 0 if gold not in top_k), cand_pids, cand_scores (gold included)."""
    import torch

    train_q = queries[queries.split == "train"]
    rows = []
    for plang, qg in train_q.groupby("plang"):
        corpus = E.pool_corpus(passages, "train", plang)
        ce = model.encode(list(corpus.text), "p")
        qe = model.encode(list(qg.text), "q")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        scores = (torch.from_numpy(qe).to(dev) @ torch.from_numpy(ce).to(dev).T).float().cpu().numpy()
        pid_idx = {p: i for i, p in enumerate(corpus.pid)}
        for i, (qid, pos) in enumerate(zip(qg.qid, qg.pos_pid)):
            top = np.argpartition(-scores[i], top_k)[:top_k]
            top = top[np.argsort(-scores[i][top], kind="stable")]
            pos_score = float(scores[i][pid_idx[pos]])
            pos_rank = int((scores[i] > pos_score).sum()) + 1
            rows.append(
                {
                    "qid": qid,
                    "pos_pid": pos,
                    "pos_score": pos_score,
                    "pos_rank": pos_rank,
                    "cand_pids": [corpus.pid.iloc[j] for j in top],
                    "cand_scores": scores[i][top].astype(float).tolist(),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class NegPolicy:
    n_neg: int = 3
    rank_lo: int = 1  # first admissible rank (1 = hardest candidate allowed)
    rank_hi: int = 30
    margin: float = 0.0  # drop candidates with score > pos_score - margin  (0.0 = only drop those beating the gold)
    use_margin: bool = True  # False = no score filtering at all
    dup_jaccard: float = 0.8
    same_title: bool = False  # True = drop candidates from the gold's article
    pick: str = "hardest"  # 'hardest' | 'random' (among survivors) | 'random_corpus' (ignore mining entirely)
    seed: int = 13

    def tag(self) -> str:
        bits = [f"n{self.n_neg}", f"r{self.rank_lo}-{self.rank_hi}", f"m{self.margin:g}" if self.use_margin else "nomargin"]
        if self.same_title:
            bits.append("notitle")
        if self.pick != "hardest":
            bits.append(self.pick)
        return "_".join(bits)


def select(raw: pd.DataFrame, passages: pd.DataFrame, policy: NegPolicy) -> tuple[dict[str, list[str]], dict]:
    """Returns (qid -> negative pids, stats about what the filters removed)."""
    title = dict(zip(passages.pid, passages.title.map(lambda t: t.strip().lower())))
    lang = dict(zip(passages.pid, passages.lang))
    toks: dict[str, frozenset] = {}

    def tk(pid: str) -> frozenset:
        if pid not in toks:
            toks[pid] = frozenset(tokenize(text_of[pid], "norm"))
        return toks[pid]

    text_of = dict(zip(passages.pid, passages.text))
    rng = random.Random(policy.seed)
    train_by_lang = {l: list(passages[(passages.pool == "train") & (passages.lang == l)].pid) for l in ("ar", "en")}
    out, removed = {}, {"gold": 0, "rank_window": 0, "margin": 0, "near_dup": 0, "same_title": 0, "kept": 0, "short_lists": 0}
    for r in raw.itertuples():
        if policy.pick == "random_corpus":
            pool = train_by_lang[lang[r.pos_pid]]
            out[r.qid] = [p for p in rng.sample(pool, policy.n_neg + 1) if p != r.pos_pid][: policy.n_neg]
            continue
        keep = []
        for rank, (pid, sc) in enumerate(zip(r.cand_pids, r.cand_scores), start=1):
            if pid == r.pos_pid:
                removed["gold"] += 1
            elif not (policy.rank_lo <= rank <= policy.rank_hi):
                removed["rank_window"] += 1
            elif policy.use_margin and sc > r.pos_score - policy.margin:
                removed["margin"] += 1
            elif policy.same_title and title[pid] and title[pid] == title[r.pos_pid]:
                removed["same_title"] += 1
            elif len(tk(pid) & tk(r.pos_pid)) / max(1, len(tk(pid) | tk(r.pos_pid))) >= policy.dup_jaccard:
                removed["near_dup"] += 1
            else:
                keep.append(pid)
        chosen = keep[: policy.n_neg] if policy.pick == "hardest" else rng.sample(keep, min(policy.n_neg, len(keep)))
        removed["kept"] += len(chosen)
        removed["short_lists"] += len(chosen) < policy.n_neg
        out[r.qid] = chosen
    removed["policy"] = asdict(policy)
    return out, removed


def raw_path(tag: str):
    return MINED / f"raw_{tag}.parquet"
