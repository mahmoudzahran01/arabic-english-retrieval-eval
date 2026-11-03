"""Fine-tune a cross-encoder reranker on (query, positive, mined hard negatives).

Pointwise BCE (default): one row per (query, passage) with label 1 for the gold and 0 for each mined negative.
Listwise CE ('--loss ce'): per query a softmax over [gold, negatives] - the cross-encoder analogue of in-batch MNRL.
Negatives come from the *baseline first-stage* retriever, i.e. from the distribution the reranker will see at inference.

Run: python -m arretrieval.retrieval.train_reranker --name rr_ft_bce --neg-model e5-small --n-neg 4 --xling
"""

from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd

from . import evalkit as E
from .negatives import NegPolicy, raw_path, select
from .paths import MODELS, RESULTS, SEED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--neg-model", default="e5-small")
    ap.add_argument("--pick", default="hardest", choices=["hardest", "random", "random_corpus"])
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--rank-hi", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--xling", action="store_true")
    ap.add_argument("--loss", default="bce", choices=["bce", "ce"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, CrossEntropyLoss
    from transformers import set_seed

    set_seed(args.seed)
    passages, queries = E.load_passages(), E.load_queries()
    text = dict(zip(passages.pid, passages.text))
    lang = dict(zip(passages.pid, passages.lang))
    raw = pd.read_parquet(raw_path(args.neg_model))
    policy = NegPolicy(
        n_neg=args.n_neg, rank_hi=args.rank_hi, margin=args.margin, use_margin=not args.no_margin, pick=args.pick, seed=args.seed
    )
    negs, mining_stats = select(raw, passages, policy)
    tq = queries[queries.split == "train"]
    if not args.xling:
        tq = tq[~tq.slice.str.contains("_xl_")]
    if args.frac < 1:
        tq = tq.sample(frac=args.frac, random_state=args.seed)
    tq = tq.sample(frac=1.0, random_state=args.seed)
    pool = {l: list(passages[(passages.pool == "train") & (passages.lang == l)].pid) for l in ("ar", "en")}
    rng = random.Random(args.seed)
    rows = []
    for r in tq.itertuples():
        ns = list(negs.get(r.qid, []))
        while len(ns) < args.n_neg:
            c = rng.choice(pool[lang[r.pos_pid]])
            if c != r.pos_pid and c not in ns:
                ns.append(c)
        if args.loss == "bce":
            rows.append({"query": r.text, "passage": text[r.pos_pid], "label": 1.0})
            rows += [{"query": r.text, "passage": text[n], "label": 0.0} for n in ns[: args.n_neg]]
        else:
            rows.append(
                {"query": r.text, "docs": [text[r.pos_pid]] + [text[n] for n in ns[: args.n_neg]], "labels": [1] + [0] * args.n_neg}
            )
    ds = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
    print(f"[{args.name}] {len(tq)} queries -> {len(ds)} rows", flush=True)

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_len, device="cuda")
    if args.loss == "bce":
        loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(args.n_neg)))
    else:
        loss = CrossEntropyLoss(model)
    steps = (len(ds) // args.bs) * args.epochs
    targs = CrossEncoderTrainingArguments(
        output_dir=str(MODELS / args.name / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_steps=max(1, int(0.1 * steps)),
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=True,
    )
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    model.save_pretrained(str(MODELS / args.name))
    logs = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    meta = {
        "name": args.name,
        "args": vars(args),
        "rows": len(ds),
        "queries": len(tq),
        "train_seconds": round(dt, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "first_loss": logs[0],
        "last_loss": logs[-1],
        "mining": mining_stats,
    }
    (RESULTS / f"{args.name}.train.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[{args.name}] done in {dt:.0f}s, peak {meta['peak_gpu_gb']} GB, loss {logs[0]:.4f} -> {logs[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
