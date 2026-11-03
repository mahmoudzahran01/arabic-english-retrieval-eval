"""The in-process exact-cosine index that replaces the vector-database service (no models involved)."""

import numpy as np
import pytest

from arretrieval.rageval import retrieval as R


@pytest.fixture()
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "INDEX_DIR", tmp_path / "index")
    R._index.clear()
    R.create_collection("c", 3, recreate=True)
    R.upsert(
        "c",
        [
            {"id": 0, "vector": [1.0, 0.0, 0.0], "payload": {"id": "P000", "lang": "en", "text": "x"}},
            {"id": 1, "vector": [0.0, 2.0, 0.0], "payload": {"id": "P001", "lang": "ar", "text": "y"}},
            {"id": 2, "vector": [1.0, 1.0, 0.0], "payload": {"id": "P002", "lang": "en", "text": "z"}},
        ],
    )
    yield "c"
    R._index.clear()


def test_search_returns_cosine_scores_best_first(index):
    hits = R.search(index, [1.0, 0.0, 0.0], 3)
    assert [h["id"] for h in hits] == ["P000", "P002", "P001"]
    assert np.isclose(hits[0]["score"], 1.0) and np.isclose(hits[1]["score"], 1 / np.sqrt(2)) and np.isclose(hits[2]["score"], 0.0)
    assert hits[0]["lang"] == "en" and "_id" not in hits[0]  # payload comes back, internal point id does not


def test_scores_are_scale_invariant(index):
    assert [h["id"] for h in R.search(index, [0.0, 50.0, 0.0], 1)] == ["P001"]  # unnormalised query and passage vectors


def test_limit_and_upsert_replacement(index):
    assert len(R.search(index, [1.0, 0.0, 0.0], 2)) == 2
    R.upsert(index, [{"id": 1, "vector": [1.0, 0.1, 0.0], "payload": {"id": "P001", "lang": "ar", "text": "y2"}}])
    top = R.search(index, [1.0, 0.0, 0.0], 3)
    assert len(top) == 3 and top[1]["id"] == "P001" and top[1]["text"] == "y2"


def test_index_is_persisted_and_missing_collection_is_reported(index, tmp_path, monkeypatch):
    R._index.clear()  # force a reload from disk
    assert R.search(index, [0.0, 1.0, 0.0], 1)[0]["id"] == "P001"
    with pytest.raises(FileNotFoundError, match="ingest"):
        R.search("missing", [1.0, 0.0, 0.0], 1)


def test_chunker_keeps_windows_short_and_covers_the_text():
    from arretrieval.rageval.ingest import chunk_passage

    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve. Thirteen fourteen."
    chunks = chunk_passage(text, max_words=6, stride_sents=2, window=3)
    assert chunks[0] == "One two three. Four five six." and all(len(c.split()) <= 6 for c in chunks)
    assert "Thirteen fourteen." in chunks[-1]
