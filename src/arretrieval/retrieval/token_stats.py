"""Passage / query length in tokens (XLM-R sentencepiece used by e5-small and the mMiniLM reranker).

Run: python -m arretrieval.retrieval.token_stats      (downloads the e5-small tokenizer only)
"""

import numpy as np

from . import evalkit as E


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    P, Q = E.load_passages(), E.load_queries()
    for lang in ("ar", "en"):
        lens = np.array([len(x) for x in tok(list(P[P.lang == lang].text), add_special_tokens=True)["input_ids"]])
        print(
            lang,
            "passages",
            len(lens),
            "tokens p50/p90/p95/p99/max",
            np.percentile(lens, [50, 90, 95, 99]).astype(int).tolist(),
            lens.max(),
            "share >256:",
            round(float((lens > 256).mean()), 3),
            "share >512:",
            round(float((lens > 512).mean()), 3),
        )
    for lang in ("ar", "en"):
        lens = np.array([len(x) for x in tok(list(Q[Q.qlang == lang].text))["input_ids"]])
        print(lang, "queries p50/p95/max", np.percentile(lens, [50, 95]).astype(int).tolist(), lens.max())


if __name__ == "__main__":
    main()
