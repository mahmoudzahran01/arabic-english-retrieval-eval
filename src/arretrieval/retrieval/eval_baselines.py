"""Stage 3: off-the-shelf baselines on dev + test: BM25 (3 tokenizations), dense (small + strong), RRF hybrids.

Run: python -m arretrieval.retrieval.eval_baselines [--splits test dev]
"""

from __future__ import annotations

import argparse
import gc

from . import evalkit as E

DENSE = {  # tag -> (hf id, query prefix, passage prefix)
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "minilm-paraphrase": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "dev"])
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--dense-tags", nargs="+", default=list(DENSE), choices=list(DENSE))
    ap.add_argument("--skip-arnorm", action="store_true", help="skip the Arabic-normalized dense variants")
    ap.add_argument("--bm25-modes", nargs="+", default=["raw", "norm", "stem"])
    args = ap.parse_args()
    passages, queries = E.load_passages(), E.load_queries()
    for split in args.splits:
        runs: dict[str, dict] = {}
        for mode in args.bm25_modes:
            with E.Timer() as t:
                runs[f"bm25-{mode}"] = E.run_system(E.BM25(mode), queries, passages, split)
            print(f"[{split}] bm25-{mode} done in {t.dt:.0f}s", flush=True)
        if not args.skip_dense:
            for tag in args.dense_tags:
                hf, qp, pp = DENSE[tag]
                for norm in (False, True):
                    if norm and (tag == "minilm-paraphrase" or args.skip_arnorm):
                        continue
                    name = f"{tag}+arnorm" if norm else tag
                    model = E.Dense(hf, tag, qp, pp, normalize_text=norm, cache=True, bs=32 if tag == "bge-m3" else 64)
                    with E.Timer() as t:
                        runs[name] = E.run_system(model, queries, passages, split)
                    print(f"[{split}] {name} done in {t.dt:.0f}s", flush=True)
                    del model
                    gc.collect()
            for dense in args.dense_tags:
                for mode in args.bm25_modes:
                    runs[f"hybrid-{dense}+bm25-{mode}"] = {
                        sl: E.rrf_runs([runs[f"bm25-{mode}"][sl], runs[dense][sl]]) for sl in runs[dense]
                    }
        for name, run in runs.items():
            res = E.evaluate_runs(run, queries, split)
            E.save_result(name, split, res, runs=run)
            print(f"[{split}] {name:34s} mrr@10: {E.fmt_row(res)}", flush=True)


if __name__ == "__main__":
    main()
