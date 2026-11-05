"""Tables for RESULTS_v2.md from runs/retrieval/v2/results/*.json (x100, MRR@10).

dev     : every training run + baselines on the extended DEV table, with a one-line description of the change
test    : final candidates on TEST, paired-bootstrap deltas (per direction, queries resampled within slice) vs v1 / bge-m3
python -m arretrieval.retrieval.v2.report_v2 dev | test NAME... | boot A B
"""

from __future__ import annotations

import json
import sys

import numpy as np

from arretrieval.retrieval import evalkit as E
from arretrieval.retrieval.paths import RESULTS as V1RES

from .common_v2 import V2RES

DIRS = E.DIRECTIONS


def load(name: str, split: str) -> dict | None:
    for d in (V2RES, V1RES):
        p = d / f"{name}.{split}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def perq(name: str, split: str) -> dict:
    if "+" in name:  # 'a+b+c' = per-query mean over several training seeds
        parts = [perq(n, split) for n in name.split("+")]
        return {sl: {m: np.mean([p[sl][m] for p in parts], axis=0) for m in parts[0][sl]} for sl in parts[0]}
    for d in (V2RES, V1RES):
        p = d / f"{name}.{split}.perq.npz"
        if p.exists():
            z = np.load(p)
            out: dict = {}
            for k in z.files:
                sl, m = k.split("|")
                out.setdefault(sl, {})[m] = z[k]
            return out
    raise FileNotFoundError(name)


def describe(a: dict) -> str:
    bits = [a.get("base", "e5-small"), f"data {a.get('data', 'v1')}"]
    if a.get("xl2_frac", 1.0) != 1.0:
        bits.append(f"xl2 x{a['xl2_frac']}")
    bits.append(f"negs {a.get('negs', 'same:1')}")
    if a.get("margin", 0.02) != 0.02:
        bits.append(f"margin {a['margin']}")
    if a.get("kd", 0):
        bits.append(f"KD w={a['kd']} T={a.get('t_scale', 20)}" + (" (KD only)" if a.get("kd_only") else ""))
    if a.get("epochs", 1) != 1:
        bits.append(f"{a['epochs']:g} ep")
    if a.get("lr", 3e-5) != 3e-5:
        bits.append(f"lr {a['lr']:g}")
    if a.get("bs", 64) != 64:
        bits.append(f"bs {a['bs']}")
    if a.get("scale", 20.0) != 20.0:
        bits.append(f"scale {a['scale']:g}")
    if a.get("lora"):
        bits.append(f"LoRA r={a['lora']}")
    if a.get("freeze_bottom", -1) >= 0:
        bits.append(f"frozen embeddings + bottom {a['freeze_bottom']} layers")
    if a.get("replay"):
        bits.append(f"+ MIRACL-ar train replay x{a.get('replay_x', 1)}")
    if a.get("pres"):
        bits.append(f"preservation w={a['pres']} (teacher {a.get('teacher', 'bgem3')})")
    if a.get("exclude"):
        bits.append("contamination-filtered")
    if a.get("xl_epochs", 1) != 1:
        bits.append(f"xling part x{a['xl_epochs']} passes")
    if a.get("sym"):
        bits.append("symmetric InfoNCE")
    if a.get("hardness"):
        bits.append(f"hardness {a['hardness']}")
    if a.get("kd_scope", "all") != "all":
        bits.append(f"KD on {a['kd_scope']} only")
    if a.get("batching", "mixed") != "mixed":
        bits.append(f"batches by {a['batching']}")
    if a.get("no_xl"):
        bits.append("no xling data")
    return ", ".join(bits)


def row(label: str, res: dict, extra: str = "", ref: float | None = None) -> str:
    d = res["directions"]
    cells = [f"{d[k]['mrr@10'] * 100:.1f}" for k in DIRS] + [f"{d['macro']['mrr@10'] * 100:.1f}"]
    delta = "" if ref is None else f" {(d['macro']['mrr@10'] * 100 - ref):+.1f} |"
    return f"| {label} | " + " | ".join(cells) + " |" + (delta or "") + (f" {extra} |" if extra else "")


def devm(name: str) -> str:
    r = load(name, "devM")
    return f"{r['slices']['dev_miracl_ar']['mrr@10'] * 100:.1f}" if r else ""


def dev_table(names: list[str] | None = None, ref_name: str = "p1_r0") -> str:
    ref = load(ref_name, "dev")
    ref_macro = ref["directions"]["macro"]["mrr@10"] * 100 if ref else None
    lines = [
        "| run | what changed vs the v1 recipe | ar->ar | en->en | ar->en | en->ar | macro | d macro | devM (MIRACL hold-out) | train s / GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    base = {"base_e5-small": "multilingual-e5-small, off the shelf", "base_bge-m3": "bge-m3, off the shelf (568M)"}
    for n, lab in base.items():
        r = load(n, "dev")
        if r:
            lines.append(
                f"| `{n}` | {lab} | " + " | ".join(f"{r['directions'][k]['mrr@10'] * 100:.1f}" for k in DIRS + ("macro",)) + " | | | |"
            )
    v1 = [load(f"v1_s{s}", "dev") for s in (13, 14, 15)]
    if all(v1):
        m = {k: np.mean([r["directions"][k]["mrr@10"] for r in v1]) * 100 for k in DIRS + ("macro",)}
        lines.append(
            "| `v1_s13/14/15` | published v1 recipe, 3 seeds (mean) | " + " | ".join(f"{m[k]:.1f}" for k in DIRS + ("macro",)) + " | | | |"
        )
    allnames = names or sorted(p.name[: -len(".dev.json")] for p in V2RES.glob("p*.dev.json"))
    for n in allnames:
        r = load(n, "dev")
        if not r or "train" not in r:
            continue
        t = r["train"]
        d = r["directions"]
        delta = "" if ref_macro is None else f"{d['macro']['mrr@10'] * 100 - ref_macro:+.1f}"
        lines.append(
            f"| `{n}` | {describe(t['args'])} | "
            + " | ".join(f"{d[k]['mrr@10'] * 100:.1f}" for k in DIRS + ("macro",))
            + f" | {delta} | {devm(n)} | {t['train_seconds']:.0f} / {t['peak_gpu_gb']:.1f} |"
        )
    return "\n".join(lines)


def boot_dir(a: str, b: str, split: str, direction: str, metric: str = "mrr@10", n_boot: int = 2000, seed: int = 0) -> dict:
    pa, pb = perq(a, split), perq(b, split)
    sls = [s for s in pa if E.DIRECTION_OF[s] == direction and s in pb]
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot)
    for s in sls:
        x, y = pa[s][metric], pb[s][metric]
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        diffs += (y[idx].mean(axis=1) - x[idx].mean(axis=1)) / len(sls)
    point = float(np.mean([pb[s][metric].mean() - pa[s][metric].mean() for s in sls]))
    return {"delta": point, "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975))}


def boot_macro(a: str, b: str, split: str, metric: str = "mrr@10", n_boot: int = 2000, seed: int = 0) -> dict:
    pa, pb = perq(a, split), perq(b, split)
    sls = [s for s in pa if s in pb]
    dirs = {s: E.DIRECTION_OF[s] for s in sls}
    nd = len(set(dirs.values()))
    rng = np.random.default_rng(seed)
    diffs, point = np.zeros(n_boot), 0.0
    for d in set(dirs.values()):
        ss = [s for s in sls if dirs[s] == d]
        for s in ss:
            x, y = pa[s][metric], pb[s][metric]
            idx = rng.integers(0, len(x), size=(n_boot, len(x)))
            diffs += (y[idx].mean(axis=1) - x[idx].mean(axis=1)) / len(ss) / nd
            point += (y.mean() - x.mean()) / len(ss) / nd
    return {"delta": float(point), "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975))}


def fmt(r: dict) -> str:
    sig = "*" if (r["lo"] > 0 or r["hi"] < 0) else ""
    return f"{r['delta'] * 100:+.1f} [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]{sig}"


def delta_table(pairs: list[tuple[str, str, str]], split: str = "test") -> str:
    lines = ["| comparison (system minus reference) | " + " | ".join(DIRS) + " | macro |", "|---|" + "---:|" * (len(DIRS) + 1)]
    for label, ref, sysn in pairs:
        cells = [fmt(boot_dir(ref, sysn, split, d)) for d in DIRS] + [fmt(boot_macro(ref, sysn, split))]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dev":
        print(dev_table())
    elif cmd == "boot":
        print(delta_table([(f"{sys.argv[3]} vs {sys.argv[2]}", sys.argv[2], sys.argv[3])]))


# ---------------------------------------------------------------------------------------------------- out of domain (mteb)
MTEB_ROWS = [
    ("XPQARetrieval", "ara-eng", "XPQA ar->en (e-commerce QA)"),
    ("XPQARetrieval", "eng-ara", "XPQA en->ar"),
    ("XPQARetrieval", "ara-ara", "XPQA ar->ar"),
    ("MLQARetrieval", "ara-eng", "MLQA ar->en (Wikipedia)"),
    ("MLQARetrieval", "eng-ara", "MLQA en->ar"),
    ("MLQARetrieval", "ara-ara", "MLQA ar->ar"),
    ("BelebeleRetrieval", "arb_Arab-eng_Latn", "Belebele ar->en (FLORES passages)"),
    ("BelebeleRetrieval", "eng_Latn-arb_Arab", "Belebele en->ar"),
    ("BelebeleRetrieval", "arb_Arab-arb_Arab", "Belebele ar->ar"),
    ("MintakaRetrieval", "ar", "Mintaka ar (Wikidata QA)"),
    ("MKQARetrieval", "ar", "MKQA ar (answer retrieval)"),
    ("XQuADRetrieval", "ar", "XQuAD ar->ar (v1 test source)"),
    ("MIRACLRetrievalHardNegatives.v2", "ar", "MIRACL-HN.v2 ar (Arabic Wikipedia)"),
    ("NamaaMrTydiReranking", "default", "GUARD Mr.TyDi-ar reranking (MAP)"),
    ("SemRel24STS", "arb", "GUARD SemRel24 STS arb (Spearman)"),
    ("STS17", "en-ar", "STS17 en-ar (Spearman)"),
    ("STS17", "ar-ar", "STS17 ar-ar (Spearman)"),
    ("Tatoeba", "ara-eng", "Tatoeba ar-en bitext (F1)"),
    ("NTREXBitextMining", "arb_Arab-eng_Latn", "NTREX ar->en bitext (F1)"),
    ("FloresBitextMining", "arb_Arab-eng_Latn", "FLORES ar->en bitext (F1)"),
    ("FloresBitextMining", "eng_Latn-arb_Arab", "FLORES en->ar bitext (F1)"),
]
RETRIEVAL = {
    "XPQARetrieval",
    "MLQARetrieval",
    "BelebeleRetrieval",
    "MintakaRetrieval",
    "MKQARetrieval",
    "XQuADRetrieval",
    "MIRACLRetrievalHardNegatives.v2",
}
SHARED = {
    "MLQARetrieval": "SQuAD-style Wikipedia QA (same family as SQuAD/XQuAD training data)",
    "XQuADRetrieval": "XQuAD = the v1 TEST source",
    "MIRACLRetrievalHardNegatives.v2": "Arabic Wikipedia (TyDi-ar training articles also come from it)",
    "Belebele": "FLORES passages, no training overlap",
    "XPQARetrieval": "no overlap (product QA)",
}


def mteb_val(tag: str, task: str, sub: str, metric: str) -> float | None:
    from .common_v2 import V2

    p = V2 / "mteb" / tag / f"{task}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    for split, subs in d["scores"].items():
        if sub in subs:
            s = subs[sub]
            if task in RETRIEVAL:
                return s.get(metric)
            return s.get("main_score")
    return None


def ood_table(tags: list[str], labels: list[str], metric: str = "ndcg_at_10") -> str:
    lines = ["| task / subset | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(tags)]
    for task, sub, lab in MTEB_ROWS:
        vals = [mteb_val(t, task, sub, metric) for t in tags]
        if all(v is None for v in vals):
            continue
        lines.append(f"| {lab} | " + " | ".join("-" if v is None else f"{v * 100:.1f}" for v in vals) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- test tables
def _cells(names: list[str], split: str, metric: str = "mrr@10") -> list[str]:
    rs = [load(n, split) for n in names]
    rs = [r for r in rs if r]
    out = []
    for k in DIRS + ("macro",):
        vals = np.array([r["directions"][k][metric] * 100 for r in rs])
        out.append(f"{vals.mean():.1f}" + (f" ± {vals.std(ddof=1):.1f}" if len(vals) > 1 else ""))
    return out


def test_table(groups: list[tuple[str, list[str]]], split: str = "test", metric: str = "mrr@10") -> str:
    lines = ["| system | params | ar->ar | en->en | ar->en | en->ar | macro | seeds |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, names in groups:
        params = "568M" if "bge" in label.lower() else ("278M" if "base" in label.lower() else "118M")
        have = [n for n in names if load(n, split)]
        if have:
            lines.append(f"| {label} | {params} | " + " | ".join(_cells(have, split, metric)) + f" | {len(have)} |")
    return "\n".join(lines)


def mixed_table(names: list[tuple[str, str]], split: str = "dev") -> str:
    from .common_v2 import V2RES as R

    lines = ["| model | slice | strict MRR@10 | lenient MRR@10 | top-1 in the query's own language |", "|---|---|---:|---:|---:|"]
    for label, n in names:
        p = R / f"{n}.{split}.mixed.json"
        if p.exists():
            for sl, v in json.loads(p.read_text()).items():
                lines.append(
                    f"| {label} | {sl} | {v['strict_mrr@10'] * 100:.1f} | {v['lenient_mrr@10'] * 100:.1f} | {v['wrong_lang_top1'] * 100:.0f}% |"
                )
    return "\n".join(lines)
