"""Fill RESULTS_v2.template.md -> RESULTS_v2.md with tables computed from runs/retrieval/v2 (no number in the tables is typed by hand).

Placeholders  {{NAME}}  are replaced by the output of the function registered below.
Run: NO_GPU=1 python -m arretrieval.retrieval.v2.build_results_v2 [template] [out]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import report_v2 as R
from .common_v2 import V2, V2RES


def seeds(prefix: str) -> list[str]:
    return sorted(p.name[: -len(".test.json")] for p in V2RES.glob(f"{prefix}_s1[345].test.json"))


def test_headline() -> str:
    g = [
        ("bge-m3 (off the shelf)", ["bge-m3"]),
        ("multilingual-e5-small (off the shelf)", ["e5-small"]),
        ("multilingual-e5-base (off the shelf)", []),
        ("v1 e5-small (3 seeds)", ["e5s_hard1_xl_e1_s13", "e5s_hard1_xl_e1_s14", "e5s_hard1_xl_e1_s15"]),
        ("**v2-small** (3 seeds)", seeds("p3_s_e1")),
        ("**v2-base** (3 seeds)", seeds("p3_b_e1")),
    ]
    return R.test_table([x for x in g if x[1]])


def test_boot() -> str:
    pairs = []
    s, b = "p3_s_e1_s13", "p3_b_e1_s13"
    for lab, ref, sysn in (
        ("v2-small s13 vs v1 s13", "e5s_hard1_xl_e1_s13", s),
        ("v2-base s13 vs v1 s13", "e5s_hard1_xl_e1_s13", b),
        ("v2-small s13 vs bge-m3", "bge-m3", s),
        ("v2-base s13 vs bge-m3", "bge-m3", b),
        ("v2-base s13 vs v2-small s13", s, b),
    ):
        if (V2RES / f"{sysn}.test.perq.npz").exists():
            pairs.append((lab, ref, sysn))
    return R.delta_table(pairs)


def dev_table() -> str:
    return R.dev_table()


def ood(tags: list[str], labels: list[str]) -> str:
    return R.ood_table(tags, labels)


def style_table() -> str:
    p = V2RES / "diag_xquad_style.json"
    if not p.exists():
        return "(not run)"
    d = json.loads(p.read_text())
    lines = ["| model | human-translated (XQuAD ar) | Gemma-translated (same questions) | gap |", "|---|---:|---:|---:|"]
    for k, v in d.items():
        lines.append(
            f"| {k} | {v['human']['mrr@10'] * 100:.1f} | {v['gemma']['mrr@10'] * 100:.1f} | {100 * (v['gemma']['mrr@10'] - v['human']['mrr@10']):+.1f} |"
        )
    return "\n".join(lines)


def audit_table() -> str:
    a, f = V2RES / "audit_ood.json", V2RES / "audit_ood_filtered.json"
    if not a.exists():
        return "(not run)"
    da, df = json.loads(a.read_text()), (json.loads(f.read_text()) if f.exists() else {})
    lines = [
        "| eval set (split) | corpus | passages >=50% 8-gram overlap with training passages (v1 data -> v2 filtered) | queries | queries near-duplicate of a training question (v1 data -> v2 filtered) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in da.items():
        w = df.get(k, {})
        lines.append(
            f"| {k} | {v['corpus']:,} | {v['corpus_passages_>=50%_8gram_overlap_with_train']} -> {w.get('corpus_passages_>=50%_8gram_overlap_with_train', '?')} | {v['queries']:,} | "
            f"{v['queries_near_duplicate_of_a_train_query']} -> {w.get('queries_near_duplicate_of_a_train_query', '?')} |"
        )
    return "\n".join(lines)


def proxy_table() -> str:
    p = V2RES / "ood_dev_ar.json"
    if not p.exists():
        return "(not run)"
    d = json.loads(p.read_text())
    lines = ["| model | mMARCO-ar dev proxy MRR@10 | nDCG@10 |", "|---|---:|---:|"]
    for k, v in d.items():
        lines.append(f"| `{k}` | {v['mrr@10'] * 100:.1f} | {v['ndcg@10'] * 100:.1f} |")
    return "\n".join(lines)


def cost_table() -> str:
    secs, n = 0.0, 0
    for p in V2RES.glob("*.train.json"):
        t = json.loads(p.read_text())
        secs += float(t.get("train_seconds", 0))
        n += 1
    return f"{n} training runs, {secs / 3600:.1f} h of training wall-clock"


def guard_split() -> str:
    p = V2RES / "guard_split" / "_report.md"
    return p.read_text() if p.exists() else "(not run)"


def mixed() -> str:
    return R.mixed_table(
        [
            ("e5-small (base)", "base_e5-small"),
            ("v1 (seed 13)", "v1_s13"),
            ("v2-small (seed 13)", "p3_s_e1_s13"),
            ("v2-base (seed 13)", "p3_b_e1_s13"),
        ],
        "test",
    )


REG = {
    k: v
    for k, v in dict(
        TEST_HEADLINE=test_headline,
        TEST_BOOT=test_boot,
        DEV_TABLE=dev_table,
        STYLE=style_table,
        AUDIT=audit_table,
        PROXY=proxy_table,
        COST=cost_table,
        MIXED_TEST=mixed,
        GUARD_SPLIT=guard_split,
    ).items()
}


def main() -> None:
    tpl = Path(sys.argv[1] if len(sys.argv) > 1 else V2 / "RESULTS_v2.template.md").read_text()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else V2 / "RESULTS_v2.md")
    for k, f in REG.items():
        tpl = tpl.replace("{{" + k + "}}", f())
    import re

    for m in re.finditer(r"\{\{OOD:([^|}]+)\|([^}]+)\}\}", tpl):
        tags, labels = m.group(1).split(","), m.group(2).split(",")
        tpl = tpl.replace(m.group(0), ood(tags, labels))
    out.write_text(tpl)
    print("wrote", out)


if __name__ == "__main__":
    main()
