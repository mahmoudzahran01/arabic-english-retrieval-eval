"""v2 embedder training: the v1 recipe (train_embedder.py) with switchable changes, one change per ablation.

Differences to v1 (everything else is identical: MNRL, NoDuplicates batches, bs 64, lr 3e-5, scale 20, warm-up 10%, bf16, max 256):
  --negs "same:1,other:1"   negatives per query by kind. `same` = hard negatives from the gold passage's language pool (v1),
                            `other` = hard negatives from the OTHER language pool (mixed-index negatives). Same false-negative policy
                            (rank 1-30, 0.02 cosine margin, near-duplicate filter) for both kinds.
  --data v1|v2              v1 = the 20,918 training queries of v1; v2 = + the extra Gemma translations (slices train_xl2_*)
  --xl2-frac F              keep a fraction of the new translated queries (v2 data only)
  --kd W [--t-scale T]      add an in-batch listwise KL term towards bge-m3 (see losses_v2.py); --kd-only drops the hard-label term
  --base e5-small|e5-base   student size
Dev (the extended dev table: 1,000 ar->en and 800 en->ar translated dev queries) is evaluated after training; test only with --test.

Run (GPU): arretrieval.retrieval.v2.train_v2 --name NAME ...
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time

import numpy as np
import pandas as pd

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval.negatives import NegPolicy, select
from arretrieval.retrieval.paths import SEED

from . import eval_v2 as EV
from . import lora_v2 as LORA
from .common_v2 import BASES, V2DATA, V2MINED, V2MODELS, V2RES, data_family, load_passages_v2, load_queries_v2


def parse_negs(spec: str) -> list[tuple[str, int]]:
    if spec in ("", "none"):
        return []
    return [(k, int(n)) for k, n in (part.split(":") for part in spec.split(","))]


def load_raw(kind: str, base: str, parts: list[str]) -> pd.DataFrame:
    fs = [V2MINED / f"raw_{kind}_{base}_{p}.parquet" for p in parts]
    return pd.concat([pd.read_parquet(f) for f in fs if f.exists()], ignore_index=True)


def load_teacher(tag: str = "bgem3") -> tuple[dict, np.ndarray, dict, np.ndarray]:
    zq, zp = np.load(V2DATA / f"teacher_{tag}_q.npz", allow_pickle=True), np.load(V2DATA / f"teacher_{tag}_p.npz", allow_pickle=True)
    return ({i: k for k, i in enumerate(zq["ids"])}, zq["emb"], {i: k for k, i in enumerate(zp["ids"])}, zp["emb"])


def build_examples(
    args, queries: pd.DataFrame, passages: pd.DataFrame, negs: dict[str, dict[str, list[str]]], teacher
) -> tuple[pd.DataFrame, list[str], dict]:
    tq = queries[queries.split == "train"]
    fam = data_family(tq.slice)
    keep_fam = {"v1": ["v1"], "v2": ["v1", "v2"], "v3": ["v1", "v2", "v3"]}[args.data] + (["replay"] if args.replay else [])
    tq = tq[fam.isin(keep_fam)]
    if args.data != "v1" and args.xl2_frac < 1.0:
        keep = tq[tq.slice.str.contains("_xl2_")].groupby("slice", group_keys=False).sample(frac=args.xl2_frac, random_state=args.seed)
        tq = pd.concat([tq[~tq.slice.str.contains("_xl2_")], keep])
    if args.no_xl:
        tq = tq[~tq.slice.str.contains("_xl")]
    n_excluded = 0
    if args.exclude:  # contamination filter (contam_v2.py): questions/passages shared with a listed MTEB-Arabic evaluation set
        ex = set(json.loads(open(args.exclude).read())["qids"])
        n_excluded = int(tq.qid.isin(ex).sum())
        tq = tq[~tq.qid.isin(ex)]
    if args.replay and args.replay_x > 1:  # the replay set is small (1.5k queries): repeat it so that it is ~10% of the mix
        rp = tq[data_family(tq.slice) == "replay"]
        tq = pd.concat([tq] + [rp] * (args.replay_x - 1))
    if args.xl_epochs > 1:  # more passes over the cross-lingual part only (monolingual part: 1 pass)
        xl = tq[tq.slice.str.contains("_xl")]
        tq = pd.concat([tq] + [xl] * (args.xl_epochs - 1))
    tq = tq.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    replay_negs = json.loads((V2DATA / "miracl_negs.json").read_text()) if args.replay else {}
    for kind, _ in args.negs_spec:
        miss = int((~(tq.qid.isin(set(negs[kind])) | tq.qid.isin(set(replay_negs)))).sum())
        assert miss == 0, f"{miss} training queries have no mined candidates for '{kind}' negatives - re-run mine_v2 for the new rows"
    if teacher is not None:
        assert tq.qid.isin(set(teacher[0])).all(), "teacher embeddings missing for some queries - re-run teacher_v2"
    text, lang = dict(zip(passages.pid, passages.text)), dict(zip(passages.pid, passages.lang))
    # random padding negatives come only from passages the run is allowed to know: no replay / v3 passages unless that data is in the mix, and (for KD) only those with a teacher embedding
    allowed = pd.Series(True, index=passages.index)
    if not args.replay:
        allowed &= passages.source != "miracl_ar_train"
    if args.data != "v3":
        allowed &= passages.source != "squad_train_unused"
    if teacher is not None:
        allowed &= passages.pid.isin(set(teacher[2]))
    pools = {l: list(passages[(passages.pool == "train") & (passages.lang == l) & allowed].pid) for l in ("ar", "en")}
    rng = random.Random(args.seed)
    q_prefix, p_prefix = args.q_prefix, args.p_prefix
    rows, groups, padded = [], [], {"same": 0, "other": 0}
    for r in tq.itertuples():
        groups.append("all" if args.batching == "mixed" else (r.plang if args.batching == "plang" else f"{r.qlang}2{r.plang}"))
        row = {"anchor": q_prefix + r.text, "positive": p_prefix + text[r.pos_pid]}
        negp: list[str] = []
        for kind, n in args.negs_spec:
            ns = list((replay_negs.get(r.qid) if (kind == "same" and r.qid in replay_negs) else negs[kind].get(r.qid, [])))[:n]
            target = lang[r.pos_pid] if kind == "same" else ("en" if lang[r.pos_pid] == "ar" else "ar")
            while len(ns) < n:
                cand = rng.choice(pools[target])
                if cand != r.pos_pid and cand not in ns and cand not in negp:
                    ns.append(cand)
                    padded[kind] += 1
            negp += ns
        for i, n in enumerate(negp, start=1):
            row[f"negative_{i}"] = p_prefix + text[n]
        if teacher is not None:
            qi, qe, pi, pe = teacher
            w = 1.0 if (args.kd_scope == "all" or r.qlang != r.plang) else 0.0  # per-example KD mask
            wp = 1.0 if lang[r.pos_pid] == "ar" else 0.0  # preservation mask: Arabic positives only
            if (
                args.kd > 0
            ):  # KD needs the teacher's q / pos / neg embeddings; preservation alone only needs the positive (4x smaller labels: the host has little spare RAM)
                lab = [qe[qi[r.qid]], pe[pi[r.pos_pid]]] + [pe[pi[n]] for n in negp]
            else:
                lab = [pe[pi[r.pos_pid]]]
            row["label"] = np.concatenate(lab + [np.array([w, wp])]).astype(np.float32)
        rows.append(row)
    stats = {
        "n_examples": len(rows),
        "excluded_by_contamination_filter": n_excluded,
        "padded_negatives": padded,
        "by_slice": {k: int(v) for k, v in tq.groupby("slice").size().items()},
    }
    return pd.DataFrame(rows), groups, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="e5-small", choices=list(BASES))
    ap.add_argument("--data", default="v1", choices=["v1", "v2", "v3"])
    ap.add_argument("--xl2-frac", type=float, default=1.0)
    ap.add_argument("--no-xl", action="store_true", help="drop every translated cross-lingual query (diagnostic)")
    ap.add_argument("--negs", default="same:1", help="'none' or kind:n[,kind:n], kinds same|other")
    ap.add_argument("--neg-model", default="e5-small", help="miner (raw files raw_{kind}_{neg-model}_{part}.parquet)")
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--kd", type=float, default=0.0, help="weight of the distillation KL term (0 = off)")
    ap.add_argument("--kd-only", action="store_true", help="no hard-label term, distillation only")
    ap.add_argument("--t-scale", type=float, default=20.0, help="teacher softmax scale (1/temperature)")
    ap.add_argument("--replay", action="store_true", help="add the MIRACL-ar TRAIN replay rows (build_miracl_replay.py; judged negatives)")
    ap.add_argument(
        "--lora", type=int, default=0, help="LoRA rank on attention + MLP of the transformer (merged back into the weights before saving)"
    )
    ap.add_argument("--lora-alpha", type=int, default=0, help="default: 2 x rank")
    ap.add_argument(
        "--freeze-bottom",
        type=int,
        default=-1,
        help="freeze the embeddings and the bottom N encoder layers (fine-tune only the layers above)",
    )
    ap.add_argument("--replay-x", type=int, default=1, help="repeat the replay rows this many times")
    ap.add_argument(
        "--pres",
        type=float,
        default=0.0,
        help="weight of the embedding-preservation term towards the (frozen) teacher on Arabic positives (needs --teacher with the student's dimension)",
    )
    ap.add_argument(
        "--teacher", default="bgem3", help="teacher embedding tag (teacher_v2.py --tag): bgem3 or e.g. m3b (the fine-tuned v2-m3)"
    )
    ap.add_argument(
        "--kd-scope",
        default="all",
        choices=["all", "xl"],
        help="distil on every example, or only on cross-lingual ones (query language != passage language)",
    )
    ap.add_argument(
        "--batching",
        default="mixed",
        choices=["mixed", "plang", "direction"],
        help="mixed = v1 (random batches); plang = every batch holds passages of ONE language; direction = every batch holds one query->passage direction",
    )
    ap.add_argument("--exclude", default="", help="json with {'qids': [...]} to drop from the training set (contam_v2.py output)")
    ap.add_argument("--xl-epochs", type=int, default=1, help="repeat the translated cross-lingual examples this many times per epoch")
    ap.add_argument("--sym", action="store_true", help="symmetric InfoNCE: add the passage->query direction (separate softmax)")
    ap.add_argument("--hardness", default="", help="EmbeddingGemma-style hardness weighting 'hard_negatives:5' or 'in_batch_negatives:9'")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument(
        "--loss",
        default="mnrl",
        choices=["mnrl", "cached"],
        help="cached = CachedMNRL (GradCache): same gradient, memory independent of the batch size",
    )
    ap.add_argument("--mini-batch", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--test", action="store_true", help="also evaluate TEST (final candidates only)")
    ap.add_argument("--keep-model", action="store_true")
    ap.add_argument("--dev-curve", action="store_true")
    args = ap.parse_args()
    args.negs_spec = parse_negs(args.negs)

    import torch
    from datasets import Dataset, DatasetDict
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
    from transformers import TrainerCallback, set_seed

    from .losses_v2 import DistillMNRL

    hf, args.q_prefix, args.p_prefix = BASES[args.base]
    set_seed(args.seed)
    passages, queries = load_passages_v2(), load_queries_v2()
    parts = {"v1": ["v1"], "v2": ["v1", "xl2"], "v3": ["v1", "xl2", "v3"]}[args.data]
    negs, mining_stats = {}, {}
    for kind, n in args.negs_spec:
        raw = load_raw(kind, args.neg_model, parts)
        policy = NegPolicy(n_neg=n, rank_lo=args.rank_lo, rank_hi=args.rank_hi, margin=args.margin, seed=args.seed)
        negs[kind], mining_stats[kind] = select(raw, passages, policy)
    teacher = load_teacher(args.teacher) if (args.kd > 0 or args.pres > 0) else None
    df, groups, data_stats = build_examples(args, queries, passages, negs, teacher)
    if args.batching == "mixed":
        ds = Dataset.from_pandas(df, preserve_index=False)
        steps_per_epoch = len(df) // args.bs
    else:  # one dataset per group + the proportional multi-dataset sampler = every batch comes from a single group
        g = np.array(groups)
        ds = DatasetDict({k: Dataset.from_pandas(df[g == k].reset_index(drop=True), preserve_index=False) for k in sorted(set(groups))})
        steps_per_epoch = sum(len(d) // args.bs for d in ds.values())
    total_steps = int(steps_per_epoch * args.epochs)
    print(
        f"[{args.name}] {len(df)} examples, {steps_per_epoch} steps/epoch, columns={[c for c in df.columns]}, padded={data_stats['padded_negatives']}",
        flush=True,
    )

    model = SentenceTransformer(hf, device="cuda")
    model.max_seq_length = args.max_len
    peft_handle = None
    if args.lora > 0 or args.freeze_bottom >= 0:
        tm = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
        if args.lora > 0:
            peft_handle = LORA.wrap(model, args.lora, args.lora_alpha)
        else:
            for p_ in tm.embeddings.parameters():
                p_.requires_grad = False
            for layer in tm.encoder.layer[: args.freeze_bottom]:
                for p_ in layer.parameters():
                    p_.requires_grad = False
            n_tr = sum(p_.numel() for p_ in tm.parameters() if p_.requires_grad)
            print(f"[{args.name}] frozen embeddings + bottom {args.freeze_bottom} layers; trainable params {n_tr / 1e6:.1f}M", flush=True)
    if (
        args.grad_ckpt
    ):  # checkpoint the transformer layers ourselves (the non-cached DistillMNRL at batch 64 would otherwise need > 50 GB for a 568M model)
        tm = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
        tm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.kd > 0 or args.pres > 0:
        loss = DistillMNRL(
            model,
            scale=args.scale,
            w_ce=0.0 if args.kd_only else 1.0,
            w_kd=args.kd,
            t_scale=args.t_scale,
            dim=teacher[1].shape[1],
            w_pres=args.pres,
        )
    else:
        kw = {}
        if args.sym:
            kw.update(directions=("query_to_doc", "doc_to_query"), partition_mode="per_direction")
        if args.hardness:
            mode, strength = args.hardness.split(":")
            kw.update(hardness_mode=mode, hardness_strength=float(strength))
        if args.loss == "cached":
            from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

            loss = CachedMultipleNegativesRankingLoss(model, scale=args.scale, mini_batch_size=args.mini_batch)
        else:
            loss = MultipleNegativesRankingLoss(model, scale=args.scale, **kw)
    out_dir = V2MODELS / args.name
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
        **({"multi_dataset_batch_sampler": MultiDatasetBatchSamplers.PROPORTIONAL} if args.batching != "mixed" else {}),
    )

    dev_curve: list[dict] = []

    class DevCurve(TrainerCallback):
        def on_epoch_end(self, a, state, control, model=None, **kw):
            d = E.Dense(None, "tmp", args.q_prefix, args.p_prefix, max_len=512, bs=128, model=model, fp16=False)
            res = E.evaluate_runs(E.run_system(d, queries, passages, "dev"), queries, "dev")
            d.restore_max_len()
            model.train()
            dev_curve.append({"epoch": round(state.epoch, 2), "dev_by_direction": {k: v["mrr@10"] for k, v in res["directions"].items()}})
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
    if (
        args.lora > 0
    ):  # fold the adapters into the weights (adapter written to disk first) so that the saved folder is a plain sentence-transformers model
        LORA.merge(model, peft_handle, adapter_dir=str(out_dir / "lora_adapter"))
    model.save(str(out_dir))
    (out_dir / "lab_retrieval.json").write_text(
        json.dumps({"q_prefix": args.q_prefix, "p_prefix": args.p_prefix, "args": {k: v for k, v in vars(args).items()}})
    )
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
    del trainer, model, ds, df
    torch.cuda.empty_cache()
    d = E.Dense(str(out_dir), args.name, args.q_prefix, args.p_prefix, max_len=512, bs=args.eval_bs)
    EV.evaluate(d, args.name, ["dev"] + (["test"] if args.test else []), queries, passages, extra={"train": meta})
    EV.evaluate_devM(d, args.name, queries, passages)
    for sp in ["dev"] + (["test"] if args.test else []):
        EV.mixed_eval(d, args.name, sp, queries, passages)
    (V2RES / f"{args.name}.train.json").write_text(json.dumps(meta, indent=1, default=str))
    if not args.keep_model:
        shutil.rmtree(out_dir, ignore_errors=True)
    else:
        shutil.rmtree(out_dir / "ckpt", ignore_errors=True)
    print(f"[{args.name}] done: train {train_s:.0f}s, peak GPU {peak_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
