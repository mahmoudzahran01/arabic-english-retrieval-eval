"""Score every judge against the planted ground truth and print/write the calibration tables.

  python -m arretrieval.rageval.analyze_judges [--split test|dev|all] [--md out.md]

For each judge the "positive" class is DEFECT (the thing a judge exists to catch):
    truth_defect = label on the judge's target axis == 0 ;  predicted_defect = judge said fail
Reported: precision / recall / F1 of defect detection, Cohen's kappa vs truth, accuracy, clustered-bootstrap 95% CI
(clusters = base question, because several items share one base answer), unparsable-reply rate.
A second table gives the per-error-type FLAG RATE: P(judge flags | item type). For defect types this is recall; for
clean types it is the false-positive rate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .paths import DATA, RUNS
from .stats import bootstrap_ci, cohen_kappa, prf
from .textnorm import contains_ref, is_abstention

AXIS = {
    "faith_v0": "faithful",
    "faith_v1": "faithful",
    "faith_v2": "faithful",
    "faith_v3": "faithful",
    "cite_v1": "attributed",
    "correct_v0": "correct",
    "correct_v1": "correct",
    "overall_v0": "acceptable",
    "overall_pipeline": "acceptable",
    "overall_pipeline_v3": "acceptable",
}
ETYPES = [
    "clean",
    "clean_numeral",
    "wrong_number",
    "wrong_number_arindic",
    "wrong_entity",
    "unsupported_claim",
    "omitted_fact",
    "wrong_language",
    "wrong_citation",
    "correct_abstention",
    "wrong_abstention",
    "halluc_unanswerable",
]
SHORT = {
    "clean": "clean",
    "clean_numeral": "numeral-var",
    "wrong_number": "wrong#",
    "wrong_number_arindic": "wrong#-AR-digits",
    "wrong_entity": "wrong-ent",
    "unsupported_claim": "halluc+",
    "omitted_fact": "omitted",
    "wrong_language": "wrong-lang",
    "wrong_citation": "wrong-cite",
    "correct_abstention": "ok-abstain",
    "wrong_abstention": "bad-abstain",
    "halluc_unanswerable": "halluc-unans",
}


def load_items():
    return {i["id"]: i for i in (json.loads(l) for l in open(DATA / "meta_items.jsonl")) if not i.get("excluded")}


def label_of(it, axis):
    """Ground-truth label on an axis. `attributed` = every claim is supported by the passage(s) it CITES
    = faithful AND citation_ok (what cite_v1 is asked); faithful=None (wrong abstention) is treated as OK there."""
    if axis == "acceptable":
        return it["acceptable"]
    if axis == "attributed":
        f, c = it["labels"]["faithful"], it["labels"]["citation_ok"]
        return int(f != 0 and c != 0)
    return it["labels"][axis]


def load_dir(d):
    out = {}
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            out[p.stem] = {r["id"]: r for r in (json.loads(l) for l in open(p))}
    return out


def string_correct(it) -> int:
    """Non-LLM baseline for the `correct` axis: reference string must occur in the answer (all ';' parts)."""
    if not it["answerable"]:
        return int(is_abstention(it["answer"]))
    if is_abstention(it["answer"]):
        return 0
    return int(all(contains_ref(it["answer"], p) for p in it["reference"].split(";") if p.strip()))


def ragas_predictions(items, tau_fc=None, tau_f=1.0):
    """Turn Ragas scores into pass/fail. faithfulness: pass iff score >= tau_f; factual correctness: pass iff >= tau_fc."""
    p = RUNS / "ragas_meta.jsonl"
    if not p.exists():
        return {}
    rows = {r["id"]: r for r in (json.loads(l) for l in open(p))}
    out = {}
    if True:
        out["ragas_faithfulness"] = {
            i: (None if rows[i].get("faithfulness") is None else int(rows[i]["faithfulness"] >= tau_f)) for i in rows
        }
    if tau_fc is not None:
        out["ragas_factual_correctness"] = {
            i: (None if rows[i].get("factual_correctness") is None else int(rows[i]["factual_correctness"] >= tau_fc)) for i in rows
        }
    return out


def best_tau(items, scores: dict, axis: str, dev_ids, grid=np.arange(0.1, 1.01, 0.05)):
    best, bt = -2, 0.5
    for t in grid:
        y = [label_of(items[i], axis) for i in dev_ids if label_of(items[i], axis) is not None and scores.get(i) is not None]
        p = [int(scores[i] >= t) for i in dev_ids if label_of(items[i], axis) is not None and scores.get(i) is not None]
        k = cohen_kappa(y, p) if len(set(y)) > 1 and len(set(p)) > 0 else -2
        if k == k and k > best:
            best, bt = k, float(t)
    return bt, best


def score_judge(items, verdict: dict, axis: str, ids):
    use = [i for i in ids if label_of(items[i], axis) is not None]
    truth = np.array([int(label_of(items[i], axis) == 0) for i in use])  # 1 = defect
    pred = np.array([int(verdict.get(i) == 0) if verdict.get(i) is not None else 0 for i in use])
    unp = sum(1 for i in use if verdict.get(i) is None)
    r = prf(truth, pred)
    clusters = [items[i]["base"].split("|")[0] for i in use]
    k = cohen_kappa(truth, pred)
    _, klo, khi = (
        bootstrap_ci(np.arange(len(use)), stat=lambda idx: _k(truth, pred, idx), n_boot=1000, clusters=clusters)
        if len(use) > 5
        else (k, np.nan, np.nan)
    )
    return {
        "n": len(use),
        "defects": int(truth.sum()),
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "kappa": k,
        "kappa_lo": klo,
        "kappa_hi": khi,
        "acc": float((truth == pred).mean()),
        "unparsable": unp,
    }


def _k(truth, pred, idx):
    idx = idx.astype(int)
    return cohen_kappa(truth[idx], pred[idx])


def flag_rates(items, verdict, ids, axis_ok=None):
    by = defaultdict(list)
    for i in ids:
        v = verdict.get(i)
        by[items[i]["etype"]].append(0 if v is None else int(v == 0))
    return {e: (float(np.mean(by[e])), len(by[e])) for e in ETYPES if by[e]}


RAW = {}


def collect(items):
    judges = {}
    for model in ("gemma", "qwen"):
        for j, rows in load_dir(RUNS / f"judge_{model}").items():
            judges[f"{model}:{j}"] = ({i: r["pass"] for i, r in rows.items()}, AXIS.get(j, "acceptable"))
            RAW[f"{model}:{j}"] = rows
    return judges


def extra_tables(items, judges, ids, split):
    """Ablation of the pipeline, per-language breakdown, judge-vs-judge agreement."""
    L = []
    # 1. pipeline ablation: which component buys the improvement?
    for model in ("gemma", "qwen"):
        for pj in ("overall_pipeline", "overall_pipeline_v3"):
            rows = RAW.get(f"{model}:{pj}")
            if not rows:
                continue

            def comp(keys):
                return {
                    i: (
                        None
                        if all(rows[i]["parts"].get(k) is None for k in keys)
                        else int(all(rows[i]["parts"].get(k) in (1, None) for k in keys))
                    )
                    for i in rows
                }

            variants = {
                "LLM parts only (faithful+correct+citation)": ["faithful", "correct", "citation"],
                "+ language check": ["faithful", "correct", "citation", "language"],
                "+ abstention check": ["faithful", "correct", "citation", "abstention"],
                "full pipeline (+both deterministic checks)": ["faithful", "correct", "citation", "language", "abstention"],
                "deterministic checks only (language+abstention)": ["language", "abstention"],
            }
            L += [
                f"\n#### Ablation of `{model}:{pj}` (axis acceptable, split={split})",
                "",
                "| components | recall of defects | precision | F1 | kappa |",
                "|---|--:|--:|--:|--:|",
            ]
            for name, keys in variants.items():
                r = score_judge(items, comp(keys), "acceptable", ids)
                L.append(f"| {name} | {r['recall']:.2f} | {r['precision']:.2f} | {r['f1']:.2f} | {r['kappa']:.2f} |")
    # 2. per-language
    L += [
        "",
        f"#### Per question-language (split={split}): kappa / defect recall / false-positive rate on clean-type items",
        "",
        "| judge | en: n | en kappa | en recall | en FPR | ar: n | ar kappa | ar recall | ar FPR |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    clean_types = {"clean", "clean_numeral", "correct_abstention"}
    for name, (verdict, axis) in sorted(judges.items()):
        if name.startswith("ragas") or name.startswith("string"):
            continue
        cells = []
        for lang in ("en", "ar"):
            sub = [i for i in ids if items[i]["q_lang"] == lang and i in verdict]
            r = score_judge(items, verdict, axis, sub)
            fp = [verdict.get(i) == 0 for i in sub if items[i]["etype"] in clean_types and verdict.get(i) is not None]
            cells += [str(r["n"]), f"{r['kappa']:.2f}", f"{r['recall']:.2f}", f"{(sum(fp) / len(fp)) if fp else float('nan'):.2f}"]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    # 2b. numbers: recall by subtype and by digit script (the "Arabic numeral variants / units" question) + FPR on numeral variants
    num_ids_all = [i for i in ids if items[i]["etype"] in ("wrong_number", "wrong_number_arindic")]
    cl_ids_all = [i for i in ids if items[i]["etype"] == "clean_numeral"]
    cols = [
        ("near (+-small)", lambda it: it["extra"].get("subtype") == "near"),
        ("scale (x10)", lambda it: it["extra"].get("subtype") == "scale"),
        ("far", lambda it: it["extra"].get("subtype") == "far"),
        ("ASCII digits", lambda it: it["extra"].get("script") == "ascii"),
        ("Arabic-Indic digits", lambda it: it["extra"].get("script") == "ar"),
    ]
    L += [
        "",
        f"#### Number errors: defect recall by subtype and digit script, and false-positive rate on correct numeral variants (split={split}; n shown for judges that scored every item)",
        "",
        "| judge | "
        + " | ".join(f"{c} (n={sum(1 for i in num_ids_all if f(items[i]))})" for c, f in cols)
        + f" | FPR on correct numeral variants (n={len(cl_ids_all)}) |",
        "|---|" + "--:|" * (len(cols) + 1),
    ]
    for name, (verdict, axis) in sorted(judges.items()):
        if axis in ("citation_ok", "attributed") and "cite" in name:
            continue
        row = []
        num_ids = [i for i in num_ids_all if i in verdict]
        cl_ids = [i for i in cl_ids_all if i in verdict]
        for c, f in cols:
            sub = [i for i in num_ids if f(items[i]) and verdict.get(i) is not None]
            row.append(f"{np.mean([verdict[i] == 0 for i in sub]):.2f}" if sub else "-")
        fp = [verdict[i] == 0 for i in cl_ids if verdict.get(i) is not None]
        L.append(f"| {name} | " + " | ".join(row) + f" | {np.mean(fp):.2f} |" if fp else f"| {name} | " + " | ".join(row) + " | - |")
    # 2c. cross-lingual vs mono-lingual evidence
    L += [
        "",
        f"#### Question language != evidence language (cross-lingual) vs same language: defect recall / clean FPR (split={split})",
        "",
        "| judge | cross: recall | cross: FPR | mono: recall | mono: FPR |",
        "|---|--:|--:|--:|--:|",
    ]
    for name, (verdict, axis) in sorted(judges.items()):
        if name.startswith("string"):
            continue
        cells = []
        for cross in (True, False):
            sub = [i for i in ids if (items[i]["q_lang"] != items[i]["gold_lang"]) == cross and i in verdict]
            r = score_judge(items, verdict, axis, sub)
            fp = [verdict.get(i) == 0 for i in sub if items[i]["etype"] in clean_types and verdict.get(i) is not None]
            cells += [f"{r['recall']:.2f}", f"{(sum(fp) / len(fp)) if fp else float('nan'):.2f}"]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    # 3. judge-vs-judge agreement on the same items (no ground truth involved)
    L += [
        "",
        f"#### Judge-vs-judge agreement (Cohen's kappa between the verdicts themselves, on the items both judges scored, split={split})",
        "",
        "| judge | n | gemma vs qwen: kappa | raw agreement |",
        "|---|--:|--:|--:|",
    ]
    for j in sorted({n.split(":")[1] for n in judges if n.startswith("gemma:")}):
        a, b = judges.get(f"gemma:{j}"), judges.get(f"qwen:{j}")
        if not a or not b:
            continue
        both = [i for i in ids if a[0].get(i) is not None and b[0].get(i) is not None]
        if len(both) > 5:
            x, y = [a[0][i] for i in both], [b[0][i] for i in both]
            L.append(f"| {j} | {len(both)} | {cohen_kappa(x, y):.2f} | {np.mean(np.array(x) == np.array(y)):.2f} |")
    # 4. before/after: paired cluster-bootstrap CI for the kappa improvement
    pairs = [
        ("gemma:overall_v0", "gemma:overall_pipeline_v3"),
        ("gemma:correct_v0", "gemma:correct_v1"),
        ("gemma:faith_v0", "gemma:faith_v3"),
        ("qwen:overall_v0", "qwen:overall_pipeline_v3"),
        ("qwen:correct_v0", "qwen:correct_v1"),
        ("qwen:faith_v0", "qwen:faith_v1"),
        ("qwen:faith_v0", "qwen:faith_v3"),
        ("qwen:faith_v2", "qwen:faith_v3"),
        ("qwen:overall_pipeline", "qwen:overall_pipeline_v3"),
    ]
    L += [
        "",
        f"#### Judge improvements: kappa before -> after, with paired cluster-bootstrap 95% CI of the difference (split={split})",
        "",
        "| axis | before | after | n | kappa before | kappa after | diff [95% CI] |",
        "|---|---|---|--:|--:|--:|---|",
    ]
    rng = np.random.default_rng(0)
    for b, a_ in pairs:
        if b not in judges or a_ not in judges:
            continue
        axis = judges[a_][1]
        use = [i for i in ids if i in judges[b][0] and i in judges[a_][0] and label_of(items[i], axis) is not None]
        if len(use) < 10:
            continue
        truth = np.array([int(label_of(items[i], axis) == 0) for i in use])
        pb = np.array([int(judges[b][0].get(i) == 0) for i in use])
        pa = np.array([int(judges[a_][0].get(i) == 0) for i in use])
        cl = np.array([items[i]["base"].split("|")[0] for i in use])
        groups = [np.where(cl == g)[0] for g in sorted(set(cl.tolist()))]
        diffs = []
        for _ in range(2000):
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            kb, ka = cohen_kappa(truth[idx], pb[idx]), cohen_kappa(truth[idx], pa[idx])
            if kb == kb and ka == ka:
                diffs.append(ka - kb)
        kb0, ka0 = cohen_kappa(truth, pb), cohen_kappa(truth, pa)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        L.append(f"| {axis} | {b} | {a_} | {len(use)} | {kb0:.2f} | {ka0:.2f} | {ka0 - kb0:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--md", default=None)
    ap.add_argument("--tau-fc", type=float, default=None, help="override Ragas FC threshold (default: tuned on dev)")
    a = ap.parse_args()
    items = load_items()
    ids = [i for i, it in items.items() if a.split == "all" or it["split"] == a.split]
    dev_ids = [i for i, it in items.items() if it["split"] == "dev"]
    judges = collect(items)
    judges["string_match:contains_ref"] = ({i: string_correct(items[i]) for i in items}, "correct")
    extra = {}
    for evaluator, rp in (("gemma", RUNS / "ragas_meta.jsonl"), ("qwen", RUNS / "ragas_meta_qwen.jsonl")):
        if not rp.exists():
            continue
        rows = {r["id"]: r for r in (json.loads(l) for l in open(rp))}
        dev_cov = [i for i in dev_ids if i in rows]
        judges[f"ragas[{evaluator}]:faithfulness==1"] = (
            {i: (None if rows[i].get("faithfulness") is None else int(rows[i]["faithfulness"] >= 0.999)) for i in rows},
            "faithful",
        )
        taus = {}
        for key, nick in (
            ("factual_correctness", "FC-f1"),
            ("factual_correctness_precision", "FC-precision"),
            ("factual_correctness_recall", "FC-recall"),
        ):
            sc = {i: rows[i].get(key) for i in rows}
            if all(v is None for v in sc.values()):
                continue
            tau, _ = best_tau(items, sc, "correct", dev_cov)  # threshold tuned on the DEV items only
            tau = a.tau_fc if a.tau_fc is not None else tau
            taus[nick] = round(tau, 2)
            judges[f"ragas[{evaluator}]:{nick}>={tau:.2f}"] = ({i: (None if sc[i] is None else int(sc[i] >= tau)) for i in rows}, "correct")
        extra[f"ragas[{evaluator}]"] = {
            "items_scored": len(rows),
            "dev-tuned thresholds": taus,
            "none_scores": {k: sum(1 for i in rows if rows[i].get(k) is None) for k in ("faithfulness", "factual_correctness")},
        }
    lines = [f"### Judge calibration vs planted ground truth - split={a.split} (n_items={len(ids)})", ""]
    lines.append("| judge | axis | n | defects | precision | recall | F1 | kappa [95% CI] | acc | unparsable |")
    lines.append("|---|---|--:|--:|--:|--:|--:|---|--:|--:|")
    results = {}
    for name, (verdict, axis) in sorted(judges.items(), key=lambda kv: (kv[1][1], kv[0])):
        r = score_judge(items, verdict, axis, [i for i in ids if i in verdict])
        results[name] = r
        lines.append(
            f"| {name} | {axis} | {r['n']} | {r['defects']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} | "
            f"{r['kappa']:.2f} [{r['kappa_lo']:.2f}, {r['kappa_hi']:.2f}] | {r['acc']:.2f} | {r['unparsable']} |"
        )
    lines += ["", f"### Flag rate by planted type (fraction of items the judge marks as DEFECT) - split={a.split}", ""]
    n_by = {e: sum(1 for i in ids if items[i]["etype"] == e) for e in ETYPES}
    lines.append("| judge | " + " | ".join(f"{SHORT[e]} (n={n_by[e]})" for e in ETYPES) + " |")
    lines.append("|---|" + "--:|" * len(ETYPES))
    for name, (verdict, axis) in sorted(judges.items(), key=lambda kv: (kv[1][1], kv[0])):
        fr = flag_rates(items, verdict, [i for i in ids if i in verdict])
        lines.append(f"| {name} [{axis}] | " + " | ".join(f"{fr[e][0]:.2f}" if e in fr else "-" for e in ETYPES) + " |")
    lines += ["", "Defect types the judge is *meant* to catch should be near 1.00; clean/ok types should be near 0.00."]
    lines.append(extra_tables(items, judges, ids, a.split))
    if extra:
        lines.append(f"\nRagas notes: {json.dumps(extra)}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    json.dump(results, open(RUNS / f"calibration_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
