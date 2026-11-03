"""Assemble every table used in docs/retrieval/RESULTS.md from the stored result json files -> <results dir>/tables.md.

Reads the committed snapshot `results/retrieval` by default (set ARR_RESULTS_DIR=runs/retrieval/results after a full run
plus the data stage); pure CPU, no model downloads.  Run: python -m arretrieval.retrieval.make_report [--out FILE]
All numbers x100 (percent-style points). Several seeds -> mean ± sample std.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import evalkit as E
from . import report_lib as R
from .paths import REPORT_CARD, REPORT_RESULTS

RESULTS = REPORT_RESULTS


DIRS_ = E.DIRECTIONS


def section(title: str, body: str) -> str:
    return f"\n### {title}\n\n{body}\n"


def seeds(prefix: str, split: str = "test") -> list[str]:
    return R.seed_names(prefix, split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "tables.md"), help="where to write the markdown tables")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    one = lambda n: [n]  # noqa: E731

    # ---- baselines --------------------------------------------------------------------------------------------
    base_rows = [
        ("BM25 raw tokens", one("bm25-raw")),
        ("BM25 + Arabic normalization", one("bm25-norm")),
        ("BM25 + normalization + light stemmer", one("bm25-stem")),
        ("MiniLM-L12 paraphrase-multilingual (dense)", one("minilm-paraphrase")),
        ("multilingual-e5-small (dense)", one("e5-small")),
        ("multilingual-e5-small, Arabic-normalized input", one("e5-small+arnorm")),
        ("bge-m3 (dense)", one("bge-m3")),
        ("bge-m3, Arabic-normalized input", one("bge-m3+arnorm")),
        ("RRF e5-small + BM25(stem)", one("hybrid-e5-small+bm25-stem")),
        ("RRF e5-small + BM25(stem), language-gated", one("hybrid-gated-e5-small+bm25-stem")),
        ("RRF bge-m3 + BM25(raw)", one("hybrid-bge-m3+bm25-raw")),
        ("RRF bge-m3 + BM25(norm)", one("hybrid-bge-m3+bm25-norm")),
        ("RRF bge-m3 + BM25(stem)", one("hybrid-bge-m3+bm25-stem")),
        ("RRF bge-m3 + BM25(stem), language-gated", one("hybrid-gated-bge-m3+bm25-stem")),
    ]
    for metric in ("mrr@10", "hit@1", "hit@10", "ndcg@10"):
        out.append(
            section(
                f"Baselines on TEST - {metric.upper()} (x100)",
                R.table([(l, n) for l, n in base_rows if R.load(n[0], "test")], "test", metric),
            )
        )
    out.append(
        section(
            "Baselines on TEST - every slice, every metric (x100)",
            R.slice_table([(l, n[0]) for l, n in base_rows if R.load(n[0], "test")], "test"),
        )
    )

    # ---- headline ----------------------------------------------------------------------------------------------------
    FTE0, HB0 = "e5s_hard1_xl_e1", "hybrid-gated-bge-m3+bm25-stem"
    head = [
        ("BM25 (normalized + light stemmer)", one("bm25-stem")),
        ("multilingual-e5-small, off the shelf (118M)", one("e5-small")),
        ("bge-m3 dense, off the shelf (568M)", one("bge-m3")),
        ("e5-small + BM25(stem), language-gated RRF", one("hybrid-gated-e5-small+bm25-stem")),
        ("**fine-tuned e5-small** (hard negatives + cross-lingual pairs; 3 seeds)", seeds(FTE0)),
        ("baseline e5-small -> top-50 -> off-the-shelf mMiniLM reranker", one("e5-small>>rr_mmarco_off")),
        ("baseline e5-small -> top-50 -> **fine-tuned reranker**", one("e5-small>>rr_ft_bce_s13")),
        ("**fine-tuned e5-small -> top-50 -> fine-tuned reranker**", one(f"{FTE0}_s13>>rr_ft_bce_s13")),
        ("gated RRF(bge-m3, BM25) -> top-50 -> fine-tuned reranker", one(f"{HB0}>>rr_ft_bce_s13")),
    ]
    head = [(l, n) for l, n in head if n and R.load(n[0], "test")]
    out.append(section("HEADLINE - TEST MRR@10 (x100)", R.table(head, "test", "mrr@10")))
    out.append(section("HEADLINE_H1 - TEST Hit@1 (x100)", R.table(head, "test", "hit@1")))

    # ---- curated experiment tables (used verbatim in RESULTS.md) ---------------------------------------------------
    def grp(label: str, prefix: str):
        return (label, seeds(prefix))

    A = [
        ("e5-small, no fine-tuning", one("e5-small")),
        grp("fine-tuned, in-batch negatives only", "e5s_inb_e1"),
        grp("+ 1 RANDOM negative per query", "e5s_randneg1_e1"),
        grp("+ 1 mined hard negative, NO false-negative filter", "e5s_hard1_nomargin_e1"),
        grp("+ 1 mined hard negative, margin 0.02 (default)", "e5s_hard1_e1"),
        grp("+ 1 mined hard negative, margin 0.05", "e5s_hard1_m05_e1"),
        grp("+ 1 mined hard negative, margin 0.02, skip ranks 1-9", "e5s_hard1_rank10_e1"),
        grp("+ 1 mined hard negative, margin 0.02, drop same-article", "e5s_hard1_notitle_e1"),
        grp("+ 1 mined hard negative (mined by bge-m3), margin 0.02", "e5s_hard1_bgem3_e1"),
        grp("+ 3 mined hard negatives, margin 0.02", "e5s_hard3_e1"),
    ]
    B = [
        grp("in-batch, 25% of queries", "e5s_inb_f25_e1"),
        grp("in-batch, 50%", "e5s_inb_f50_e1"),
        grp("in-batch, 100%", "e5s_inb_e1"),
        grp("in-batch, 25% x 4 epochs (same #steps as 100% x 1)", "e5s_inb_f25_e4"),
        grp("hard neg, 25% of queries", "e5s_hard1_f25_e1"),
        grp("hard neg, 50%", "e5s_hard1_f50_e1"),
        grp("hard neg, 100%", "e5s_hard1_e1"),
        grp("hard neg, 25% x 4 epochs (same #steps as 100% x 1)", "e5s_hard1_f25_e4"),
    ]
    C = [
        grp("in-batch, 1 epoch", "e5s_inb_e1"),
        grp("in-batch, 3 epochs", "e5s_inb_e3"),
        grp("hard neg, 1 epoch", "e5s_hard1_e1"),
        grp("hard neg, 3 epochs", "e5s_hard1_e3"),
    ]
    D = [
        grp("default: bs 64, lr 3e-5, scale 20", "e5s_hard1_e1"),
        grp("scale 50 (temperature 0.02)", "e5s_hard1_sc50_e1"),
        grp("lr 1e-5", "e5s_hard1_lr1e5_e1"),
        grp("lr 6e-5", "e5s_hard1_lr6e5_e1"),
        grp("batch 32", "e5s_hard1_bs32_e1"),
        grp("batch 128 (CachedMNRL)", "e5s_hard1_bs128c_e1"),
        grp("batch 512 (CachedMNRL, 32 steps)", "e5s_hard1_cached512_e1"),
    ]
    E = [
        ("e5-small, no fine-tuning", one("e5-small")),
        grp("in-batch, monolingual data", "e5s_inb_e1"),
        grp("in-batch, + translated cross-lingual queries", "e5s_inb_xl_e1"),
        grp("hard neg, monolingual data", "e5s_hard1_e1"),
        grp("hard neg, + translated cross-lingual queries  (FINAL RECIPE)", "e5s_hard1_xl_e1"),
        grp("hard neg x3 (cached), + cross-lingual", "e5s_hard3c_xl_e1"),
        grp("hard neg, + cross-lingual, 2 epochs", "e5s_hard1_xl_e2"),
        grp("hard neg mined by bge-m3, + cross-lingual", "e5s_hard1_xl_bgem3_e1"),
        ("bge-m3 dense (568M, off the shelf)", one("bge-m3")),
    ]
    for tag, rws in (("A_negatives", A), ("B_datasize", B), ("C_epochs", C), ("D_hyperparams", D), ("E_crosslingual", E)):
        rws = [(l, n) for l, n in rws if n]
        out.append(section(f"EXP {tag} - TEST MRR@10", R.table(rws, "test", "mrr@10", nseeds_col=True)))
        out.append(section(f"EXP {tag} - TEST Hit@1", R.table(rws, "test", "hit@1", nseeds_col=True)))
        out.append(section(f"EXP {tag} - TEST Hit@10", R.table(rws, "test", "hit@10", nseeds_col=True)))
        drws = [(l, n) for l, n in rws if any(R.load(x, "dev") for x in n)]
        out.append(section(f"EXP {tag} - DEV MRR@10", R.table(drws, "dev", "mrr@10", nseeds_col=True, macro_col="macro:ar->ar,en->en")))
    dp = [
        ("hard neg (3 seeds, seed 13) vs in-batch only (seed 13)", "e5s_inb_e1_s13", "e5s_hard1_e1_s13"),
        ("random neg (seed 13) vs in-batch only (seed 13)", "e5s_inb_e1_s13", "e5s_randneg1_e1_s13"),
        ("hard neg margin 0.02 vs no filter (seed 13)", "e5s_hard1_nomargin_e1_s13", "e5s_hard1_e1_s13"),
        ("+ cross-lingual data vs mono, hard neg (seed 13)", "e5s_hard1_e1_s13", "e5s_hard1_xl_e1_s13"),
        ("FINAL (hard neg + xling, seed 13) vs e5-small baseline", "e5-small", "e5s_hard1_xl_e1_s13"),
        ("FINAL (hard neg + xling, seed 13) vs bge-m3 dense", "bge-m3", "e5s_hard1_xl_e1_s13"),
    ]
    dp = [x for x in dp if R.load(x[1], "test") and R.load(x[2], "test")]
    out.append(section("Paired bootstrap, TEST MRR@10 difference [95% CI] in points (* = CI excludes 0)", R.delta_table(dp)))

    # ---- discovered experiment groups ------------------------------------------------------------------------
    groups = sorted({p.name.split(".")[0].rsplit("_s", 1)[0] for p in RESULTS.glob("e5s_*.test.json")})
    rows = [(g.replace("__", ">>"), seeds(g)) for g in groups if seeds(g)]  # file names use "__" for the pipeline separator ">>"
    ref = [("e5-small baseline", one("e5-small"))]
    out.append(section("Fine-tuned embedders - TEST MRR@10 (x100)", R.table(ref + rows, "test", "mrr@10", nseeds_col=True)))
    out.append(section("Fine-tuned embedders - TEST Hit@1 (x100)", R.table(ref + rows, "test", "hit@1", nseeds_col=True)))
    out.append(section("Fine-tuned embedders - TEST Hit@10 (x100)", R.table(ref + rows, "test", "hit@10", nseeds_col=True)))
    out.append(section("Fine-tuned embedders - TEST nDCG@10 (x100)", R.table(ref + rows, "test", "ndcg@10", nseeds_col=True)))
    dev_ref = [("e5-small baseline", one("e5-small"))] if R.load("e5-small", "dev") else []
    out.append(
        section(
            "Fine-tuned embedders - DEV MRR@10 (x100, model selection only; mono macro = mean of ar->ar and en->en, the only directions every run was scored on; ar->en / en->ar are shown where the run was evaluated after the translated dev queries existed)",
            R.table(dev_ref + rows, "dev", "mrr@10", nseeds_col=True, macro_col="macro:ar->ar,en->en"),
        )
    )

    # ---- training cost + dev curves ----------------------------------------------------------------------------
    lines = [
        "| Run | examples | steps | train s | peak GPU GB | first loss | last loss | padded negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    curves = []
    for p in sorted(RESULTS.glob("e5s_*.train.json")):
        t = json.loads(p.read_text())
        lines.append(
            f"| {t['name']} | {t['data']['n_examples']} | {t['steps']} | {t['train_seconds']:.0f} | {t['peak_gpu_gb']} | "
            f"{t['first_loss']:.3f} | {t['last_loss']:.3f} | {t['data']['padded_negatives']} |"
        )
        if t.get("dev_curve"):
            curves.append(
                f"- {t['name']}: "
                + ", ".join(f"epoch {c['epoch']:g}: dev macro MRR@10 {c['dev_macro_mrr@10'] * 100:.1f}" for c in t["dev_curve"])
            )
    out.append(section("Training cost", "\n".join(lines)))
    if curves:
        out.append(section("Dev curves (evaluated at the end of every epoch)", "\n".join(curves)))

    # ---- paired bootstrap vs baseline --------------------------------------------------------------------------
    pairs = []
    for g, names in rows:
        if names:
            pairs.append((f"{g} (seed {names[0].rsplit('_s', 1)[1]}) - e5-small", "e5-small", names[0]))
    out.append(
        section("TEST MRR@10 change vs the e5-small baseline, paired bootstrap 95% CI (points; * = CI excludes 0)", R.delta_table(pairs))
    )

    # ---- rerank ---------------------------------------------------------------------------------------------------
    FTE = "e5s_hard1_xl_e1_s13"
    HFT = f"hybrid-gated-{FTE}+bm25-stem"
    HB = "hybrid-gated-bge-m3+bm25-stem"
    RR = [
        ("(a) first stage e5-small, no reranker", "e5-small"),
        ("(a) + mMiniLM cross-encoder, off the shelf", "e5-small>>rr_mmarco_off"),
        ("(a) + fine-tuned reranker (mined hard negatives, + cross-lingual)", "e5-small>>rr_ft_bce_s13"),
        ("(a) + fine-tuned reranker, RANDOM negatives", "e5-small>>rr_ft_randneg_s13"),
        ("(a) + fine-tuned reranker, no cross-lingual training data", "e5-small>>rr_ft_bce_noxl_s13"),
        ("(a) + fine-tuned reranker, rerank only top-10", "e5-small>>rr_ft_bce_s13_top10"),
        ("(a) + fine-tuned reranker, rerank only top-20", "e5-small>>rr_ft_bce_s13_top20"),
        ("(b) first stage fine-tuned e5-small, no reranker", FTE),
        ("(b) + mMiniLM cross-encoder, off the shelf", f"{FTE}>>rr_mmarco_off"),
        ("(b) + fine-tuned reranker", f"{FTE}>>rr_ft_bce_s13"),
        ("(c) first stage bge-m3 + BM25(stem), language-gated RRF", HB),
        ("(c) + mMiniLM cross-encoder, off the shelf", f"{HB}>>rr_mmarco_off"),
        ("(c) + fine-tuned reranker", f"{HB}>>rr_ft_bce_s13"),
        ("(d) first stage fine-tuned e5-small + BM25(stem), language-gated RRF", HFT),
        ("(d) + fine-tuned reranker", f"{HFT}>>rr_ft_bce_s13"),
    ]
    RR = [(l, [n]) for l, n in RR if R.load(n, "test")]
    if RR:
        for metric in ("mrr@10", "hit@1", "hit@10", "ndcg@10"):
            out.append(
                section(f"RERANK - TEST {metric.upper()} (x100); every reranker sees the first stage's top-50", R.table(RR, "test", metric))
            )
        out.append(
            section(
                "RERANK - candidate recall: Hit@50 of each first stage (x100) = ceiling for reranking top-50",
                R.table([(l, n) for l, n in RR if "no reranker" in l or "first stage" in l], "test", "hit@50"),
            )
        )
        rp = [
            ("(a) FT reranker vs off-the-shelf mMiniLM", "e5-small>>rr_mmarco_off", "e5-small>>rr_ft_bce_s13"),
            ("(a) FT reranker (mined negs) vs FT reranker (random negs)", "e5-small>>rr_ft_randneg_s13", "e5-small>>rr_ft_bce_s13"),
            ("(a) FT reranker vs no reranker", "e5-small", "e5-small>>rr_ft_bce_s13"),
            ("(b) FT reranker behind FT embedder vs FT embedder alone", FTE, f"{FTE}>>rr_ft_bce_s13"),
            (
                "(b) vs (a): FT embedder + FT reranker vs baseline embedder + FT reranker",
                "e5-small>>rr_ft_bce_s13",
                f"{FTE}>>rr_ft_bce_s13",
            ),
        ]
        rp = [x for x in rp if R.load(x[1], "test") and R.load(x[2], "test")]
        out.append(section("RERANK - paired bootstrap on TEST MRR@10 [95% CI], points", R.delta_table(rp)))
    sub = RESULTS / "subset_150.scores.json"
    if sub.exists():
        d = json.loads(sub.read_text())
        ls = [
            "| System (same 1050 test queries: 150 per slice) | ar->ar | en->en | ar->en | en->ar | macro |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        nice = {
            "e5-small": "baseline e5-small, no reranker",
            "e5-small>>rr_mmarco_off": "+ mMiniLM cross-encoder (118M), off the shelf",
            "e5-small>>rr_ft_bce_s13": "+ mMiniLM cross-encoder, fine-tuned (mined negatives)",
            "e5-small>>rr_ft_randneg_s13": "+ mMiniLM cross-encoder, fine-tuned (random negatives)",
            "e5-small>>rr_bge_v2m3_off_sub150": "+ bge-reranker-v2-m3 (568M), off the shelf",
        }
        for k, v in d.items():
            ds = [v[x]["mrr@10"] * 100 for x in DIRS_ if x in v]
            ls.append(f"| {nice.get(k, k)} | " + " | ".join(f"{x:.1f}" for x in ds) + f" | {np.mean(ds):.1f} |")
        out.append(section("RERANK - strong 568M reference reranker on a 1050-query subset, MRR@10 (x100)", "\n".join(ls)))
    lat = RESULTS / "latency.json"
    if lat.exists():
        d = json.loads(lat.read_text())
        ls = ["| Component | p50 ms | p95 ms | mean ms |", "|---|---:|---:|---:|"]
        ls += [
            f"| {k.replace(' | ', ', ')} | {v['p50_ms']} | {v['p95_ms']} | {v['mean_ms']} |" for k, v in d.items() if isinstance(v, dict)
        ]
        out.append(section("Per-query latency", "\n".join(ls)))
    sens = [
        (l, n)
        for l, n in [
            ("e5-small baseline", "e5-small"),
            ("bge-m3 baseline", "bge-m3"),
            ("e5s_inb_e1_s13", "e5s_inb_e1_s13"),
            ("e5s_hard1_e1_s13", "e5s_hard1_e1_s13"),
            ("e5s_hard1_xl_e1_s13", "e5s_hard1_xl_e1_s13"),
        ]
        if R.load(n, "test")
    ]
    if (REPORT_CARD / "test_pids_near_train.json").exists() and sens:
        out.append(
            section(
                "Sensitivity to near-duplicate leakage (test queries whose gold passage overlaps a training passage removed)",
                R.contamination_sensitivity(sens),
            )
        )
    rob = RESULTS / "robustness.json"
    if rob.exists():
        d = json.loads(rob.read_text())
        ls = [
            f"Fraction of Arabic test queries changed by the corruption: {d['fraction_of_queries_changed']}",
            "",
            "| System | clean MRR@10 | corrupted-query MRR@10 | change |",
            "|---|---:|---:|---:|",
        ]
        ls += [
            f"| {k} | {v['clean']['macro'] * 100:.1f} | {v['noisy']['macro'] * 100:.1f} | {v['delta_macro'] * 100:+.1f} |"
            for k, v in d["systems"].items()
        ]
        out.append(section("Orthographic-noise robustness (ar->ar slices tydi_ar, arcd_ar, xquad_ar2ar; macro MRR@10 x100)", "\n".join(ls)))
    for f in sorted(RESULTS.glob("mining_*.json")):
        out.append(section(f"Mining analysis ({f.stem})", "```json\n" + f.read_text() + "\n```"))
    audit = REPORT_CARD / "leakage_audit.json"
    if audit.exists():
        out.append(section("Leakage audit", "```json\n" + audit.read_text() + "\n```"))
    manifest = REPORT_CARD / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        out.append(
            section(
                "Data manifest",
                "```json\n"
                + json.dumps({k: m[k] for k in m if k not in ("title_splits(train,dev,dev_bg,test_bg)",)}, indent=1, ensure_ascii=False)
                + "\n```",
            )
        )
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path} ({len(out)} sections)")


if __name__ == "__main__":
    main()
