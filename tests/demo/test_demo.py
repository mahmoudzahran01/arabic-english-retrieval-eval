"""CPU unit tests for the demo helpers (no models, no gradio)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from demo.engines import top_k

from demo import app as demo_app
from demo import build_index


def test_top_k_orders_by_score_and_respects_the_language_mask():
    corpus = np.eye(4, dtype=np.float32)
    q = np.array([0.1, 0.9, 0.0, 0.5], dtype=np.float32)
    assert [i for i, _ in top_k(q, corpus, 2)] == [1, 3]
    allowed = np.array([True, False, True, True])
    assert [i for i, _ in top_k(q, corpus, 2, allowed)] == [3, 0]
    assert len(top_k(q, corpus, 10, allowed)) == 3  # never more than the allowed rows


def test_query_language_heuristic():
    assert demo_app.query_language("متى ولد طه حسين؟") == "ar"
    assert demo_app.query_language("When was James Tobin born?") == "en"
    assert demo_app.query_language("Airport فيلم") == "ar" or demo_app.query_language("Airport فيلم") == "en"
    assert demo_app.query_language("1234") == "en"


def test_every_example_is_a_string_and_both_languages_are_present():
    langs = {demo_app.query_language(q) for q in demo_app.EXAMPLES}
    assert langs == {"ar", "en"}


def _synthetic_data(tmp_path):
    rows = []
    for pool in ("train", "dev", "test"):
        for lang, src in (("ar", "tydi_ar_train"), ("en", "squad_train")):
            for i in range(60):
                rows.append(
                    {
                        "pid": f"{lang}-{pool}-{i}",
                        "lang": lang,
                        "text": "x" * 200 + f" {i}",
                        "title": f"t{i}",
                        "source": src,
                        "pool": pool,
                        "kind": "gold",
                    }
                )
    P = pd.DataFrame(rows)
    Q = []
    for sl, ql, pl in (
        ("train_tydi_ar", "ar", "ar"),
        ("train_arcd_ar", "ar", "ar"),
        ("train_xl_ar2en", "ar", "en"),
        ("train_squad_en", "en", "en"),
        ("train_tydi_en", "en", "en"),
        ("train_xl_en2ar", "en", "ar"),
    ):
        for i in range(10):
            Q.append(
                {
                    "qid": f"{sl}-{i}",
                    "text": f"question {sl} {i} ?",
                    "qlang": ql,
                    "pos_pid": f"{pl}-train-{i}",
                    "plang": pl,
                    "split": "train",
                    "slice": sl,
                    "source": "s",
                    "title": "t",
                    "src_qid": f"q{i}",
                }
            )
    P.to_parquet(tmp_path / "passages.parquet")
    pd.DataFrame(Q).to_parquet(tmp_path / "queries.parquet")


def test_select_corpus_uses_train_pool_only_and_keeps_the_gold_passages(tmp_path):
    _synthetic_data(tmp_path)
    corpus, examples = build_index.select_corpus(tmp_path, n_per_lang=40, n_examples=2)
    assert set(corpus.pid.str.split("-").str[1]) == {"train"}
    assert corpus.pid.is_unique and set(corpus.lang) == {"ar", "en"}
    assert corpus.licence.notna().all()
    assert {e["gold_pid"] for e in examples} <= set(corpus.pid)
    assert len(examples) == 2 * len(build_index.EXAMPLE_SLICES)
    assert corpus.groupby("lang").size().min() >= 40


def test_select_corpus_is_deterministic(tmp_path):
    _synthetic_data(tmp_path)
    a, _ = build_index.select_corpus(tmp_path, n_per_lang=30)
    b, _ = build_index.select_corpus(tmp_path, n_per_lang=30)
    assert list(a.pid) == list(b.pid)


class _FakeEncoder:
    """Maps a query to a fixed unit vector so that the ranking is known."""

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def encode(self, texts, kind="q", bs=1):
        return np.tile(self.vec / np.linalg.norm(self.vec), (len(texts), 1))


def _tiny_index(tmp_path):
    P = pd.DataFrame(
        {
            "pid": ["en-0", "ar-1", "en-2", "ar-3"],
            "lang": ["en", "ar", "en", "ar"],
            "title": ["Tobin <b>", "طه حسين", "Dieppe", "ابن خلدون"],
            "text": [
                "James Tobin was born in 1918 & died in 2002.",
                "ولد طه حسين عام 1889.",
                "Dieppe is in Normandy.",
                "ولد ابن خلدون في تونس.",
            ],
            "source": ["squad_train", "tydi_ar_train", "tydi_en_train", "arcd_train"],
            "licence": ["x"] * 4,
        }
    )
    P.to_parquet(tmp_path / "passages.parquet")
    np.save(tmp_path / "emb_v2.npy", np.eye(4, dtype=np.float16))
    np.save(tmp_path / "emb_v1.npy", np.eye(4, dtype=np.float16)[::-1].copy())
    return tmp_path


def test_search_and_render_end_to_end_with_fake_encoders(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path)
    demo = demo_app.Demo(index=idx, v2_dir=idx, threads=1)
    monkeypatch.setattr(demo, "encoder", lambda key: _FakeEncoder([0, 1, 0, 0] if key == "v2" else [0, 0, 0, 1]))
    assert demo.available_baselines() == ["v1 (e5-small fine-tune)"]  # bge-m3 has no embeddings in this index
    left, right, note = demo_app.run(demo, "متى ولد طه حسين؟", 2, "both languages", "v1 (e5-small fine-tune)")
    assert "#1" in left and "AR" in left and 'dir="rtl"' in left.replace("'", '"') and "ar-1" not in left
    assert "طه حسين" in left and "overlap" in note and "Arabic" in note
    assert "cross-lingual" in right  # v1's top hit is the English passage: language differs from the (Arabic) query
    left2, _, _ = demo_app.run(demo, "who?", 3, "English passages only", demo_app.NONE)
    assert "AR</span>" not in left2 and "EN" in left2
    assert "&lt;b&gt;" in left2 or "&lt;b&gt;" in demo_app.render_hits(
        "t", demo.search("x", 4, None, ("v2",))["v2"], "en", 4
    )  # titles are escaped


def test_empty_query_is_handled(tmp_path):
    idx = _tiny_index(tmp_path)
    demo = demo_app.Demo(index=idx, v2_dir=idx, threads=1)
    left, right, note = demo_app.run(demo, "   ", 5, "both languages", demo_app.NONE)
    assert "Type a question" in left and right == "" and note == ""


def test_batches_cover_every_text_once_and_group_by_length():
    from demo.engines import _batches

    texts = ["aaaa", "a", "aaa", "aa", "aaaaa"]
    idx = [i for b in _batches(texts, 2) for i in b]
    assert sorted(idx) == list(range(5))
    assert [len(texts[i]) for i in idx] == sorted(len(t) for t in texts)
