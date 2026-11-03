"""Hand-built example for the hard-negative filters in negatives.select()."""

import pandas as pd

from arretrieval.retrieval.negatives import NegPolicy, select


def _fixture():
    passages = pd.DataFrame(
        [
            {
                "pid": "gold",
                "lang": "en",
                "text": "the eiffel tower is in paris france and was built in 1889",
                "title": "Eiffel Tower",
                "pool": "train",
            },
            {
                "pid": "dup",
                "lang": "en",
                "text": "the eiffel tower is in paris france and was built in 1889 .",
                "title": "Other",
                "pool": "train",
            },
            {
                "pid": "sib",
                "lang": "en",
                "text": "construction of the tower took two years and 18000 pieces",
                "title": "Eiffel Tower",
                "pool": "train",
            },
            {"pid": "n1", "lang": "en", "text": "the louvre is a museum in paris", "title": "Louvre", "pool": "train"},
            {"pid": "n2", "lang": "en", "text": "big ben is a clock tower in london", "title": "Big Ben", "pool": "train"},
            {"pid": "n3", "lang": "en", "text": "mount fuji is a volcano in japan", "title": "Fuji", "pool": "train"},
        ]
    )
    #                       rank:    1       2       3      4      5      6
    raw = pd.DataFrame(
        [
            {
                "qid": "q",
                "pos_pid": "gold",
                "pos_score": 0.80,
                "pos_rank": 2,
                "cand_pids": ["dup", "gold", "sib", "n1", "n2", "n3"],
                "cand_scores": [0.81, 0.80, 0.79, 0.70, 0.60, 0.50],
            }
        ]
    )
    return raw, passages


def test_gold_and_near_duplicate_and_beating_candidates_are_removed():
    raw, passages = _fixture()
    negs, stats = select(raw, passages, NegPolicy(n_neg=3, rank_hi=6, margin=0.0))
    assert "gold" not in negs["q"] and "dup" not in negs["q"]  # gold removed; dup scores above gold (and is a near-duplicate)
    assert negs["q"] == ["sib", "n1", "n2"]  # hardest first, sibling paragraph survives margin=0
    assert stats["gold"] == 1 and stats["margin"] == 1


def test_margin_drops_candidates_too_close_to_the_gold():
    raw, passages = _fixture()
    negs, _ = select(raw, passages, NegPolicy(n_neg=3, rank_hi=6, margin=0.05))  # drops anything above 0.75: dup, sib
    assert negs["q"] == ["n1", "n2", "n3"]


def test_same_title_filter_and_rank_window():
    raw, passages = _fixture()
    negs, _ = select(raw, passages, NegPolicy(n_neg=3, rank_hi=6, use_margin=False, same_title=True))
    assert "sib" not in negs["q"]  # sibling paragraph of the gold's article
    negs, _ = select(raw, passages, NegPolicy(n_neg=3, rank_lo=4, rank_hi=5, use_margin=False))
    assert negs["q"] == ["n1", "n2"]  # window keeps ranks 4..5 only


def test_random_corpus_ignores_mining():
    raw, passages = _fixture()
    negs, _ = select(raw, passages, NegPolicy(n_neg=2, pick="random_corpus"))
    assert len(negs["q"]) == 2 and "gold" not in negs["q"]
