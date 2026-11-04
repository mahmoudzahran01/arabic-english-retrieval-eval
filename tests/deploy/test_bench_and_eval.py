"""CPU tests of the benchmark and quality-evaluation plumbing with fake encoders (no torch / onnxruntime)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from deploy.eval_quality import CachedRetriever, SplitEncoder

from deploy import bench


class FakeEncoder:
    def __init__(self, dim: int = 4):
        self.calls: list[tuple[str, int]] = []
        self.dim = dim

    def encode(self, texts, kind, bs=None):
        self.calls.append((kind, len(texts)))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, int(t.split()[-1]) % self.dim] = 1.0  # the last token of a text picks its axis
        return out


def test_run_workload_batches_and_statistics():
    enc = FakeEncoder()
    texts = [f"text {i}" for i in range(50)]
    res = bench.run_workload(enc, texts, "q", bs=4, calls=6, warm=2)
    assert len(enc.calls) == 8 and all(k == "q" and n == 4 for k, n in enc.calls)
    assert res["calls"] == 6 and res["batch"] == 4
    assert 0 <= res["p50_ms"] <= res["p95_ms"] and res["throughput_sent_per_s"] > 0


def test_run_workload_pads_the_last_batch_when_texts_run_out():
    enc = FakeEncoder()
    bench.run_workload(enc, ["a 1", "b 2", "c 3"], "p", bs=2, calls=3, warm=0)
    assert all(n == 2 for _, n in enc.calls)


def test_sample_texts_is_deterministic_and_test_split_only(tmp_path):
    q = pd.DataFrame({"split": ["train"] * 5 + ["test"] * 30, "text": [f"q{i}" for i in range(35)]})
    p = pd.DataFrame({"pool": ["dev"] * 5 + ["test"] * 30, "text": [f"p{i}" for i in range(35)]})
    q.to_parquet(tmp_path / "queries.parquet")
    p.to_parquet(tmp_path / "passages.parquet")
    a = bench.sample_texts(tmp_path, n_q=10, n_p=10)
    b = bench.sample_texts(tmp_path, n_q=10, n_p=10)
    assert a == b and len(a[0]) == 10 and len(a[1]) == 10
    assert all(int(t[1:]) >= 5 for t in a[0]) and all(int(t[1:]) >= 5 for t in a[1])  # train / dev rows excluded


def _units():
    c = pd.DataFrame({"pid": ["p0", "p1", "p2", "p3"], "text": ["x 0", "x 1", "x 2", "x 3"]})
    q = pd.DataFrame({"qid": ["q1", "q3"], "text": ["q 1", "q 3"]})
    return q, c


def test_cached_retriever_ranks_and_reuses_the_disk_cache(tmp_path):
    from arretrieval.retrieval import evalkit as E

    q, c = _units()
    enc = FakeEncoder()
    r1 = CachedRetriever(SplitEncoder(enc, enc), E, tmp_path, "m__fp32")
    run = r1.rank(q, c)
    assert run["q1"][0][0] == "p1" and run["q3"][0][0] == "p3"
    assert list(tmp_path.glob("m__fp32__*.npy"))
    enc2 = FakeEncoder()
    r2 = CachedRetriever(SplitEncoder(enc2, enc2), E, tmp_path, "m__fp32")
    run2 = r2.rank(q, c)
    assert run2["q1"][0][0] == "p1"
    assert ("p", 4) not in enc2.calls  # the corpus came from the cache, only the queries were encoded
    assert r2.corpus_seconds == 0.0


def test_split_encoder_routes_queries_and_passages():
    a, b = FakeEncoder(), FakeEncoder()
    s = SplitEncoder(a, b)
    s.encode(["x 1"], "q")
    s.encode(["x 2", "x 3"], "p")
    assert a.calls == [("q", 1)] and b.calls == [("p", 2)]
