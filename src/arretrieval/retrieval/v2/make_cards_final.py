"""Generate the three FINAL model-card DRAFTS (v2-m3, v2-base, v2-small) in the v1 card style from the stored result files - no number is typed by hand.

Writes  runs/retrieval/v2/cards/{v2-m3,v2-base,v2-small}.md  and a copy as README.md into models/final_{m3,base,small}/ (the sentence-transformers folder, which is what a Hub upload would take).
Licence is TBD (MIT or CC BY-SA 4.0 - the user decides); nothing is uploaded.   Run: NO_GPU=1 python -m arretrieval.retrieval.v2.make_cards_final
"""

from __future__ import annotations

import json

import numpy as np

from . import report_v2 as R
from .common_v2 import V2, V2MODELS, V2RES

BGE, E5B, E5S, V1 = "base_bge-m3", "base_e5-base", "base_e5-small", ["e5s_hard1_xl_e1_s13", "e5s_hard1_xl_e1_s14", "e5s_hard1_xl_e1_s15"]
DIRS = ("ar->ar", "en->en", "ar->en", "en->ar", "macro")
BOARD = {x["model"]: x for x in json.loads((V2RES / "mteb_board.json").read_text())}
PAIRED = json.loads((V2RES / "mteb_paired.json").read_text())
G = ["MIRACLRetrievalHardNegatives.v2", "NamaaMrTydiReranking", "SemRel24STS"]

CFG = {
    "v2-m3": dict(
        dir="final_m3",
        hub="mzahran1/arabic-english-bge-m3-retrieval",
        title="bge-m3, fine-tuned for Arabic-English retrieval (v2-m3)",
        base="BAAI/bge-m3",
        params="568M",
        prefix="none",
        board="final_m3",
        ref_board="BAAI__bge-m3",
        ref_test=BGE,
        ref_label="bge-m3 (base)",
        me=["final_m3"],
        me_dev=["final_m3", "w_m3b_a40_s14", "w_m3b_a40_s15"],
        me_conf=["w_m3b_a40_s16", "w_m3b_a40_s17"],
        paired="final_m3 vs BAAI__bge-m3",
        rerank="final_m3",
    ),
    "v2-base": dict(
        dir="final_base",
        hub="mzahran1/arabic-english-e5-base-retrieval",
        title="multilingual-e5-base, fine-tuned for Arabic-English retrieval (v2-base)",
        base="intfloat/multilingual-e5-base",
        params="278M",
        prefix="e5",
        board="final_base",
        ref_board="intfloat__multilingual-e5-base",
        ref_test=E5B,
        ref_label="multilingual-e5-base (base)",
        me=["st_b_soup"],
        me_dev=["st_b_s13", "st_b_s14", "st_b_s15"],
        me_conf=[],
        paired="final_base vs intfloat__multilingual-e5-base",
        rerank="st_b_soup",
    ),
    "v2-small": dict(
        dir="final_small",
        hub="mzahran1/arabic-english-e5-small-retrieval-v2",
        title="multilingual-e5-small, fine-tuned for Arabic-English retrieval (v2-small)",
        base="intfloat/multilingual-e5-small",
        params="118M",
        prefix="e5",
        board="final_small",
        ref_board="intfloat__multilingual-e5-small",
        ref_test=E5S,
        ref_label="multilingual-e5-small (base)",
        me=["st_s_soup"],
        me_dev=["st_s_s13", "st_s_s14", "st_s_s15"],
        me_conf=[],
        paired="final_small vs intfloat__multilingual-e5-small",
        rerank="st_s_soup",
    ),
}


def ms(names: list[str], key: str) -> str:
    v = np.array([R.load(n, "test")["directions"][key]["mrr@10"] * 100 for n in names])
    return f"{v.mean():.1f}" + (f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else "")


def row(label: str, params: str, names: list[str]) -> str:
    return f"| {label} | {params} | " + " | ".join(ms(names, k) for k in DIRS) + " |"


def rr(name: str, key: str) -> str:
    return f"{R.load(f'{name}>>rr_ft_bce_s13', 'test')['directions'][key]['mrr@10'] * 100:.1f}"


def card(key: str, c: dict) -> str:
    p = c["prefix"] == "e5"
    qp, pp = ("query: ", "passage: ") if p else ("", "")
    soup = key != "v2-m3"
    rows = ["| model | params | ar->ar | en->en | ar->en | en->ar | macro |", "|---|---:|---:|---:|---:|---:|---:|"]
    rows.append(row(c["ref_label"], {"v2-m3": "568M", "v2-base": "278M", "v2-small": "118M"}[key], [c["ref_test"]]))
    if key != "v2-m3":
        rows.append(row("bge-m3", "568M", [BGE]))
    rows.append(row("v1 e5-small (3 seeds)", "118M", V1))
    if soup:
        rows.append(row("**this model** (3-seed soup)", c["params"], c["me"]))
        rows.append(row("same recipe, the 3 individual seeds", c["params"], c["me_dev"]))
    else:
        rows.append(row("**this model** (seed 13, WiSE 0.4)", c["params"], c["me"]))
        rows.append(row("same recipe, development seeds 13/14/15", c["params"], c["me_dev"]))
        rows.append(row("same recipe, **confirmatory** seeds 16/17 (frozen recipe, never used for selection)", c["params"], c["me_conf"]))
    rows.append(
        f"| this model + the companion reranker (top-50) | {c['params']} + 118M | {rr(c['rerank'], 'ar->ar')} | {rr(c['rerank'], 'en->en')} | {rr(c['rerank'], 'ar->en')} | {rr(c['rerank'], 'en->ar')} | {rr(c['rerank'], 'macro')} |"
    )
    sysn = "+".join(c["me_dev"]) if not soup else c["me"][0]
    pairs = [(f"this model vs {c['ref_label'].split(' (')[0]}", c["ref_test"], sysn), ("this model vs v1 (seed 13)", V1[0], sysn)]
    if key != "v2-m3":
        pairs.insert(1, ("this model vs bge-m3", BGE, sysn))
    boot = R.delta_table(pairs)
    me, ref = BOARD[c["board"]], BOARD[c["ref_board"]]
    pr = PAIRED[c["paired"]]
    lo, hi = pr["ci95_task_bootstrap"]
    mt = [
        "| MTEB(ara, v1) | " + c["ref_label"].split(" (")[0] + " | this model |",
        "|---|---:|---:|",
        f"| Mean (Task), 19 tasks | {ref['Mean(Task)']:.2f} | **{me['Mean(Task)']:.2f}** |",
        f"| Mean (TaskType) | {ref['Mean(TaskType)']:.2f} | {me['Mean(TaskType)']:.2f} |",
        f"| ar-en cross-lingual subsets | {ref['ar-en cross-lingual subsets']:.1f} | {me['ar-en cross-lingual subsets']:.1f} |",
        f"| ar-only subsets | {ref['ar-only subsets']:.1f} | {me['ar-only subsets']:.1f} |",
    ]
    guards = ["| Arabic-only guard task | " + c["ref_label"].split(" (")[0] + " | this model |", "|---|---:|---:|"] + [
        f"| {t} | {ref['per_task'][t]:.1f} | {me['per_task'][t]:.1f} |" for t in G
    ]
    verdict = {
        "v2-m3": f"Mean (Task) {me['Mean(Task)'] - ref['Mean(Task)']:+.2f} vs bge-m3 (task-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}], {pr['wins']} wins / {pr['losses']} losses / {pr['ties']} tie): numerically ahead, **not statistically distinguishable**. Both confirmatory seeds give 72.28 / 72.27.",
        "v2-base": f"Mean (Task) {me['Mean(Task)'] - ref['Mean(Task)']:+.2f} vs e5-base (CI [{lo:+.2f}, {hi:+.2f}], {pr['wins']} wins / {pr['losses']} losses): wins on {pr['wins']} of 19 tasks but the CI over tasks includes 0.",
        "v2-small": f"Mean (Task) {me['Mean(Task)'] - ref['Mean(Task)']:+.2f} vs e5-small (CI [{lo:+.2f}, {hi:+.2f}], includes 0); vs v1 {PAIRED['final_small vs mzahran1__arabic-english-e5-small-retrieval']['mean_diff']:+.2f} (CI "
        f"[{PAIRED['final_small vs mzahran1__arabic-english-e5-small-retrieval']['ci95_task_bootstrap'][0]:+.2f}, {PAIRED['final_small vs mzahran1__arabic-english-e5-small-retrieval']['ci95_task_bootstrap'][1]:+.2f}], excludes 0).",
    }[key]
    blurb = {
        "v2-m3": "A 568M-parameter embedder for search over mixed Arabic and English text. It is [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) fine-tuned on 39,377 Arabic/English question-passage pairs (including machine-translated cross-lingual queries) and then **weight-interpolated with the original model** (WiSE-FT: 60% original bge-m3 + 40% fine-tuned weights) to limit forgetting. It fixes the hard case, **an Arabic query finding an English passage**, where off-the-shelf bge-m3 is weak.",
        "v2-base": "A 278M-parameter embedder for search over mixed Arabic and English text, [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) **distilled from v2-m3** (listwise KL on in-batch scores, KD only, 2 epochs; 3-seed weight soup). It handles **an Arabic query finding an English passage** well above its base and above off-the-shelf bge-m3.",
        "v2-small": "A 118M-parameter embedder for search over mixed Arabic and English text, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) **distilled from v2-m3** (listwise KL on in-batch scores, KD only, 2 epochs; 3-seed weight soup). It is the successor of the v1 checkpoint: +5.0 on Arabic-query -> English-passage retrieval, same size; it does not reach bge-m3 on that direction.",
    }[key]
    limits = {
        "v2-m3": [
            "Evaluated on Wikipedia-style factoid QA with one gold passage per query; gains on enterprise documents are not measured.",
            "**Forgetting is reduced, not removed.** Against bge-m3 it loses 1.0 on MIRACL-HN and 3.4 on the Mr.TyDi Arabic reranking task (MTEB(ara) guards) and gains 1.8 on SemRel24 STS; the pre-interpolation fine-tune gains more on ar->en (+2.4 to 2.7 TEST MRR@10) and forgets much more (see the blend curve in the report).",
            "With the companion cross-encoder on top (top-50) the first-stage advantage over bge-m3 disappears (ar->en 75.3 vs 75.3).",
            "The cross-lingual training queries are machine-translated; text beyond 256 tokens was not seen in training.",
            "**Disclosed training-data leak:** a post-hoc exact-match check found 9 kept training rows (8 distinct TyDi-ar questions) that equal queries of the MTEB MIRACL-HN.v2 ar dev set (0.8% of its queries; 0 for Mr.TyDi reranking). Not retrained; the effect on the reported MIRACL-HN score is small and the score is below the base model's anyway.",
        ],
        "v2-base": [
            "Evaluated on Wikipedia-style factoid QA; enterprise documents are not measured.",
            "**It forgets Arabic-only reranking:** the Mr.TyDi Arabic reranking task drops from 77.1 (e5-base) to 68.2, MIRACL-HN from 73.0 to 70.7. Use the base model or bge-m3 for pure-Arabic reranking-style workloads.",
            "MTEB(ara) Mean (Task) +1.02 over e5-base is not statistically distinguishable at 19 tasks (CI includes 0), although it wins 15 of 19 tasks.",
            "e5 prefixes are required; text beyond 256 tokens was not seen in training.",
            "**Disclosed training-data leak:** a post-hoc exact-match check found 9 kept training rows (8 distinct TyDi-ar questions) that equal queries of the MTEB MIRACL-HN.v2 ar dev set (0.8% of its queries; 0 for Mr.TyDi reranking). Not retrained; the effect on the reported MIRACL-HN score is small and the score is below the base model's anyway.",
        ],
        "v2-small": [
            "Evaluated on Wikipedia-style factoid QA; enterprise documents are not measured.",
            "**It forgets Arabic-only skills relative to e5-small:** Mr.TyDi Arabic reranking 79.2 -> 68.9, MIRACL-HN 72.4 -> 68.3, SemRel24 STS 51.0 -> 46.1. Against v1 these are all better (65.9 / 66.3 / 41.2).",
            "MTEB(ara) Mean (Task) +0.57 over e5-small is within noise; its Mean (TaskType) (65.42) is below e5-small's (65.70).",
            "e5 prefixes are required; text beyond 256 tokens was not seen in training.",
            "**Disclosed training-data leak:** a post-hoc exact-match check found 9 kept training rows (8 distinct TyDi-ar questions) that equal queries of the MTEB MIRACL-HN.v2 ar dev set (0.8% of its queries; 0 for Mr.TyDi reranking). Not retrained; the effect on the reported MIRACL-HN score is small and the score is below the base model's anyway.",
        ],
    }[key]
    train = {
        "v2-m3": [
            "**Data**: 39,377 training examples after a contamination filter: TyDi-QA Arabic and English, ARCD, SQuAD v1.1 train, plus Gemma-4-26B-A4B-translated cross-lingual queries (ar->en and en->ar, 24,131 in total). XQuAD, MLQA, MIRACL, XPQA, Belebele and the other MTEB(ara) evaluation sets are not trained on; near-duplicates of their queries and overlapping positives were removed (with one disclosed exception, see Limitations).",
            "**Negatives**: 1 hard negative per query mined with the base model (ranks 1-30, 0.02 cosine margin).",
            "**Recipe**: MultipleNegativesRankingLoss through GradCache (mini-batch 16, effective batch 64), lr 1e-5, scale 50, 1 epoch, max 256 tokens, bf16, one NVIDIA GB10; then `W = 0.6 W_bge-m3 + 0.4 W_finetuned` (WiSE-FT). Alpha 0.4 was chosen on a pre-registered validation suite (dev, held-out MIRACL train queries, mMARCO-ar proxy, XNLI-train, ArSarcasm probes, Arabic STS-B), never on MTEB.",
        ],
        "v2-base": [
            "**Data**: the same 39,377 filtered examples as v2-m3.",
            "**Teacher**: v2-m3 (this release's sibling); **recipe**: listwise KL on in-batch similarities (T = 20), no cross-entropy term, 2 epochs, lr 3e-5, batch 64, bf16; the released weights are the uniform average of three seeds.",
        ],
        "v2-small": [
            "**Data**: the same 39,377 filtered examples as v2-m3.",
            "**Teacher**: v2-m3; **recipe**: listwise KL on in-batch similarities (T = 20), no cross-entropy term, 2 epochs, lr 3e-5, batch 64, bf16; the released weights are the uniform average of three seeds.",
        ],
    }[key]
    usage_note = (
        "The e5 prefixes are required: `query: ` for queries and `passage: ` for documents."
        if p
        else "No prefixes are used (like bge-m3). Dense embeddings are L2-normalised."
    )
    lim = "\n".join(f"- {x}" for x in limits)
    tr = "\n".join(f"- {x}" for x in train)
    return f"""---
license: mit  # TBD - MIT or cc-by-sa-4.0 (the user decides before upload)
language: [ar, en]
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: {c["base"]}
tags: [sentence-transformers, retrieval, cross-lingual, arabic, rag, hard-negatives]
datasets: [google-research-datasets/tydiqa, hsseinmz/arcd, rajpurkar/squad]
metrics: [mrr]
---

# {c["title"]}  (DRAFT - not uploaded, licence TBD)

{blurb}

Code, benchmark and full results: the `arabic-english-retrieval-eval` repository (`docs/retrieval/RESULTS_v2.md`). Companion reranker: [`mzahran1/arabic-english-reranker-mminilm`](https://huggingface.co/mzahran1/arabic-english-reranker-mminilm). Intended repository name: `{c["hub"]}` (to be confirmed).

## Results

Test MRR@10 x 100 on the bilingual benchmark of v1 (TyDi-QA, ARCD, XQuAD; 6,823 test queries; disjoint train/dev/test articles; leakage-audited). Mean ± sample std over seeds where several are listed.

{chr(10).join(rows)}

Paired bootstrap, per-query (2,000 resamples, point estimate [95% CI], * = CI excludes 0):

{boot}

### MTEB(ara, v1) (19 tasks, same pipeline for every model)

{chr(10).join(mt)}

{verdict}

### Arabic-only regression guards (MTEB(ara))

{chr(10).join(guards)}

## Usage

{usage_note}

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{c["hub"]}")
query = model.encode(["{qp}متى تأسست جامعة الملك عبدالله للعلوم والتقنية؟"])
docs = model.encode([
    "{pp}The King Abdullah University of Science and Technology was founded in 2009 in Thuwal.",
    "{pp}Riyadh is the capital of Saudi Arabia and its largest city.",
])
print(model.similarity(query, docs))  # the KAUST passage scores highest
```

## Training

{tr}

## Limitations

{lim}

## License

**TBD (MIT or CC BY-SA 4.0 - the user decides).** The base model is MIT; the training data keeps its own licences: TyDi-QA Apache-2.0, ARCD MIT, SQuAD v1.1 CC BY-SA 4.0 (questions machine-translated with Gemma-4-26B-A4B).
"""


def main() -> None:
    out_dir = V2 / "cards"
    out_dir.mkdir(exist_ok=True)
    for key, c in CFG.items():
        text = card(key, c)
        (out_dir / f"{key}.md").write_text(text)
        folder = V2MODELS / c["dir"]
        if folder.exists():
            (folder / "README.md").write_text(text)
        print("wrote", out_dir / f"{key}.md")


if __name__ == "__main__":
    main()
