"""Build the demo index OFFLINE: pick a mixed Arabic + English corpus of TRAIN-pool passages and embed it with each model.

The Space then only encodes the user's query. Run one model at a time (the baselines are large):

    python -m demo.build_index --out demo/index --data-dir runs/retrieval/data --select                       # corpus + examples
    python -m demo.build_index --out demo/index --data-dir ... --model v2 --onnx-root artifacts/v2-base --variant fp32
    python -m demo.build_index --out demo/index --data-dir ... --model bge-m3 --hf BAAI/bge-m3
    python -m demo.build_index --out demo/index --data-dir ... --model v1 --hf mzahran1/arabic-english-e5-small-retrieval

Corpus licences: TyDi QA GoldP (Apache-2.0), SQuAD v1.1 train (CC BY-SA 4.0), ARCD train (MIT); the passage text is Wikipedia text
(CC BY-SA). Only the TRAIN pool is used (never dev/test passages). The fine-tuned models saw many of these passages in training, so
the side-by-side view is a demonstration, not a benchmark (see demo/README.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 13
MIN_CHARS, MAX_CHARS = 150, 1200
EMBED_MAX_LEN = 192  # tokens; keeps the offline CPU build of the large baseline affordable
LICENCES = {
    "tydi_ar_train": "TyDi QA GoldP, Apache-2.0 (Wikipedia text, CC BY-SA)",
    "tydi_en_train": "TyDi QA GoldP, Apache-2.0 (Wikipedia text, CC BY-SA)",
    "squad_train": "SQuAD v1.1, CC BY-SA 4.0",
    "arcd_train": "ARCD, MIT (Wikipedia text, CC BY-SA)",
}
EXAMPLE_SLICES = {  # slice -> (query language, passage language, human label)
    "train_tydi_ar": ("ar", "ar", "TyDi-ar question"),
    "train_arcd_ar": ("ar", "ar", "ARCD question"),
    "train_xl_ar2en": ("ar", "en", "Arabic question, English passage (machine-translated query)"),
    "train_squad_en": ("en", "en", "SQuAD question"),
    "train_tydi_en": ("en", "en", "TyDi-en question"),
    "train_xl_en2ar": ("en", "ar", "English question, Arabic passage (machine-translated query)"),
}


def select_corpus(data_dir: Path, n_per_lang: int, n_examples: int = 4) -> tuple[pd.DataFrame, list[dict]]:
    """Stratified (by source) sample of train-pool passages, always containing the gold passages of the example queries."""
    P = pd.read_parquet(data_dir / "passages.parquet")
    Q = pd.read_parquet(data_dir / "queries.parquet")
    P = P[(P.pool == "train") & P.text.str.len().between(MIN_CHARS, MAX_CHARS)].drop_duplicates("pid")
    rng = np.random.default_rng(SEED)
    pid_ok = set(P.pid)
    examples, forced = [], set()
    for sl, (ql, _pl, label) in EXAMPLE_SLICES.items():
        q = Q[(Q.split == "train") & (Q.slice == sl) & Q.pos_pid.isin(pid_ok) & Q.text.str.len().between(15, 90)]
        for r in q.iloc[rng.permutation(len(q))[:n_examples]].itertuples():
            examples.append({"query": r.text, "qlang": ql, "gold_pid": r.pos_pid, "slice": sl, "label": label})
            forced.add(r.pos_pid)
    parts = []
    for lang, g in P.groupby("lang"):
        need = max(n_per_lang - len(forced & set(g.pid)), 0)
        rest = g[~g.pid.isin(forced)]
        # equal share per source inside a language, then top up from the pool if a source is small
        srcs = sorted(rest.source.unique())
        pick = [s.iloc[rng.permutation(len(s))[: need // len(srcs)]] for _, s in rest.groupby("source")]
        chosen = pd.concat(pick) if pick else rest.iloc[:0]
        extra = rest[~rest.pid.isin(chosen.pid)]
        chosen = pd.concat([chosen, extra.iloc[rng.permutation(len(extra))[: need - len(chosen)]]])
        parts.append(pd.concat([g[g.pid.isin(forced)], chosen]))
    corpus = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    corpus["licence"] = corpus.source.map(LICENCES)
    return corpus[["pid", "lang", "title", "text", "source", "licence"]], examples


def make_encoder(args):
    from .engines import E5_PREFIXES, NO_PREFIXES, HFEmbedder, OnnxEmbedder

    if args.model == "v2":
        return OnnxEmbedder(args.onnx_root, args.variant, E5_PREFIXES, EMBED_MAX_LEN, args.threads)
    if args.model == "bge-m3":
        return HFEmbedder(args.hf, "cls", NO_PREFIXES, EMBED_MAX_LEN, args.threads)
    if args.model == "v1":
        return HFEmbedder(args.hf, "mean", E5_PREFIXES, EMBED_MAX_LEN, args.threads)
    raise SystemExit(f"unknown model {args.model}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True, help="folder with the v1 passages.parquet / queries.parquet")
    ap.add_argument("--select", action="store_true", help="(re)build passages.parquet + examples.json")
    ap.add_argument("--n-per-lang", type=int, default=2000)
    ap.add_argument("--model", choices=["v2", "bge-m3", "v1"])
    ap.add_argument("--onnx-root", help="v2: folder with <variant>/model.onnx, tokenizer.json, config.json")
    ap.add_argument("--variant", default="fp32", help="v2 ONNX variant used for the CORPUS (queries use the ONNX graph in the app)")
    ap.add_argument("--hf", help="baselines: HF id or local folder")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-mem-gb", type=float, default=15.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.select or not (out / "passages.parquet").exists():
        corpus, examples = select_corpus(Path(args.data_dir), args.n_per_lang)
        corpus.to_parquet(out / "passages.parquet")
        (out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))
        print(f"[index] corpus {len(corpus)} passages: {corpus.groupby(['lang', 'source']).size().to_dict()}", flush=True)
    if not args.model:
        return
    try:
        from deploy.common import wait_for_memory

        wait_for_memory(args.min_mem_gb)
    except ImportError:  # the demo folder can be used on its own
        pass
    corpus = pd.read_parquet(out / "passages.parquet")
    enc = make_encoder(args)
    t0 = time.time()
    emb = enc.encode(list(corpus.text), "p")
    np.save(out / f"emb_{args.model}.npy", emb.astype(np.float16))
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.setdefault("models", {})[args.model] = {
        "dim": int(emb.shape[1]),
        "corpus_encoder": args.variant if args.model == "v2" else args.hf,
        "embed_max_len": EMBED_MAX_LEN,
        "seconds": round(time.time() - t0, 1),
    }
    meta["n_passages"] = int(len(corpus))
    meta["seed"] = SEED
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"[index] {args.model}: {emb.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
