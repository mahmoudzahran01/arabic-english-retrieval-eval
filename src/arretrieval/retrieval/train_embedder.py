"""Fine-tune a bi-encoder (default multilingual-e5-small) with MultipleNegativesRankingLoss.

Every training example is (query, gold passage [, mined hard negatives]).  Besides the explicit negatives, every other
passage in the batch is a negative for the query (in-batch negatives); NoDuplicates batching makes sure two copies of
the same passage (SQuAD/TyDi paragraphs with several questions!) never land in one batch, which would otherwise create
false in-batch negatives.

Examples
  python -m arretrieval.retrieval.train_embedder --name e5s_inbatch_s13 --neg-model none
  python -m arretrieval.retrieval.train_embedder --name e5s_hard3_s13 --neg-model e5-small --n-neg 3
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

BASES = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
}


def build_examples(queries: pd.DataFrame, passages: pd.DataFrame, args, negs: dict[str, list[str]] | None) -> tuple[pd.DataFrame, dict]:
    tq = queries[queries.split == "train"]
    if not args.xling:
        tq = tq[~tq.slice.str.contains("_xl_")]
    if args.frac < 1.0:  # stratified random subset, deterministic in (seed-independent) data_seed
        tq = tq.groupby("slice", group_keys=False).sample(frac=args.frac, random_state=args.data_seed)
    tq = tq.sample(frac=1.0, random_state=args.data_seed).reset_index(drop=True)
    text = dict(zip(passages.pid, passages.text))
    lang = dict(zip(passages.pid, passages.lang))
    pool = {l: list(passages[(passages.pool == "train") & (passages.lang == l)].pid) for l in ("ar", "en")}
    rng = random.Random(args.data_seed)
    rows, padded = [], 0
    for r in tq.itertuples():
        row = {"anchor": args.q_prefix + r.text, "positive": args.p_prefix + text[r.pos_pid]}
        if negs is not None:
            ns = list(negs.get(r.qid, []))
            while len(ns) < args.n_neg:  # too few survivors after filtering -> pad with random same-language negatives
                cand = rng.choice(pool[lang[r.pos_pid]])
                if cand != r.pos_pid and cand not in ns:
                    ns.append(cand)
                    padded += 1
            for i, n in enumerate(ns[: args.n_neg], start=1):
                row[f"negative_{i}"] = args.p_prefix + text[n]
        rows.append(row)
    return pd.DataFrame(rows), {
        "n_examples": len(rows),
        "padded_negatives": padded,
        "by_slice": {k: int(v) for k, v in tq.groupby("slice").size().items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="e5-small", choices=list(BASES))
    ap.add_argument("--neg-model", default="none", help="'none' = in-batch negatives only; else the baseline used for mining")
    ap.add_argument("--n-neg", type=int, default=0)
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--same-title", action="store_true", help="drop negatives from the gold passage's article")
    ap.add_argument("--pick", default="hardest", choices=["hardest", "random", "random_corpus"])
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--xling", action="store_true", help="include Gemma-translated cross-lingual training queries")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="MNRL scale = 1/temperature")
    ap.add_argument("--loss", default="mnrl", choices=["mnrl", "cached"])
    ap.add_argument("--mini-batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dev-curve", action="store_true", help="evaluate dev after every epoch")
    ap.add_argument("--keep-model", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    args.data_seed = args.seed

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback, set_seed

    hf, args.q_prefix, args.p_prefix = BASES[args.base]
    set_seed(args.seed)
    passages, queries = E.load_passages(), E.load_queries()
    negs, mining_stats = None, None
    if args.neg_model != "none":
        assert args.n_neg > 0
        raw = pd.read_parquet(raw_path(args.neg_model))
        policy = NegPolicy(
            n_neg=args.n_neg,
            rank_lo=args.rank_lo,
            rank_hi=args.rank_hi,
            margin=args.margin,
            use_margin=not args.no_margin,
            same_title=args.same_title,
            pick=args.pick,
            seed=args.seed,
        )
        negs, mining_stats = select(raw, passages, policy)
    df, data_stats = build_examples(queries, passages, args, negs)
    ds = Dataset.from_pandas(df, preserve_index=False)
    steps_per_epoch = len(df) // args.bs
    total_steps = steps_per_epoch * args.epochs
    print(f"[{args.name}] {len(df)} examples, {steps_per_epoch} steps/epoch, columns={list(df.columns)}", flush=True)

    model = SentenceTransformer(hf, device="cuda")
    model.max_seq_length = args.max_len
    loss_cls = losses.CachedMultipleNegativesRankingLoss if args.loss == "cached" else losses.MultipleNegativesRankingLoss
    loss = loss_cls(model, scale=args.scale, **({"mini_batch_size": args.mini_batch} if args.loss == "cached" else {}))
    out_dir = MODELS / args.name
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_steps=max(1, int(0.1 * total_steps)),
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )

    dev_curve: list[dict] = []

    class DevCurve(TrainerCallback):
        def on_epoch_end(self, a, state, control, model=None, **kw):
            d = E.Dense(None, "tmp", args.q_prefix, args.p_prefix, max_len=512, bs=128, model=model, fp16=False)
            res = E.evaluate_runs(E.run_system(d, queries, passages, "dev"), queries, "dev")
            d.restore_max_len()
            model.train()
            dev_curve.append(
                {
                    "epoch": round(state.epoch, 2),
                    "dev_macro_mrr@10": res["directions"]["macro"]["mrr@10"],
                    "dev_by_direction": {k: v["mrr@10"] for k, v in res["directions"].items()},
                }
            )
            print(f"[{args.name}] epoch {state.epoch:.1f} dev {dev_curve[-1]}", flush=True)

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=ds, loss=loss, callbacks=[DevCurve()] if args.dev_curve else []
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    losses_log = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    model.save(str(out_dir))
    (out_dir / "model_meta.json").write_text(json.dumps({"q_prefix": args.q_prefix, "p_prefix": args.p_prefix, "args": vars(args)}))
    meta = {
        "name": args.name,
        "args": {k: v for k, v in vars(args).items()},
        "train_seconds": round(train_s, 1),
        "peak_gpu_gb": round(peak_gb, 2),
        "steps": total_steps,
        "first_loss": losses_log[0] if losses_log else None,
        "last_loss": losses_log[-1] if losses_log else None,
        "data": data_stats,
        "mining": mining_stats,
        "dev_curve": dev_curve,
    }
    del trainer, model
    torch.cuda.empty_cache()
    if not args.skip_eval:
        d = E.Dense(str(out_dir), args.name, args.q_prefix, args.p_prefix, max_len=512, bs=128)
        for split in ("dev", "test"):
            run = E.run_system(d, queries, passages, split)
            res = E.evaluate_runs(run, queries, split)
            E.save_result(args.name, split, res, runs=run if split == "test" else None, extra={"train": meta})
            print(f"[{args.name}] {split} mrr@10: {E.fmt_row(res)}  macro={res['directions']['macro']['mrr@10']:.4f}", flush=True)
    (RESULTS / f"{args.name}.train.json").write_text(json.dumps(meta, indent=1, default=str))
    if not args.keep_model:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
    print(f"[{args.name}] done: train {train_s:.0f}s, peak GPU {peak_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
