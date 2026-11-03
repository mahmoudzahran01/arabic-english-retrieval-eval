"""Human-vs-judge agreement. Fill the blank `human_*` columns of runs/rageval/human_labeling.csv, then run:

    python -m arretrieval.rageval.agreement                       # default paths
    python -m arretrieval.rageval.agreement --csv my_labels.csv

Label values accepted: 1/0, yes/no, y/n, true/false, good/bad (blank = skipped, so partial files work).
  human_faithful    1 = every claim in the answer is stated in the documents shown
  human_correct     1 = the answer gives the reference answer's fact(s) with nothing contradicting them
  human_acceptable  1 = faithful AND correct AND same language as the question AND cites a document that really
                    supports it AND (abstains exactly when no document contains the answer)
Prints, per judge: n, raw agreement, Cohen's kappa, confusion matrix (rows = human, cols = judge, order [bad, good]),
plus human-vs-planted-truth on the planted items, and writes runs/rageval/agreement_report.md.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from .paths import RUNS
from .stats import cohen_kappa, confusion, raw_agreement

TRUE = {"1", "yes", "y", "true", "good", "t"}
FALSE = {"0", "no", "n", "false", "bad", "f"}


def parse(v: str):
    v = (v or "").strip().lower()
    return 1 if v in TRUE else 0 if v in FALSE else None


def fmt_cm(cm: np.ndarray) -> str:
    return f"[[{cm[0, 0]:3d} {cm[0, 1]:3d}]\n         [{cm[1, 0]:3d} {cm[1, 1]:3d}]]"


def compare(name, human: dict, other: dict, ids) -> dict | None:
    pairs = [(human[i], other.get(i)) for i in ids if human.get(i) is not None and other.get(i) is not None]
    if len(pairs) < 3:
        return None
    h, o = zip(*pairs)
    return {"name": name, "n": len(pairs), "raw": raw_agreement(h, o), "kappa": cohen_kappa(h, o), "cm": confusion(h, o)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RUNS / "human_labeling.csv"))
    ap.add_argument("--key", default=str(RUNS / "human_labeling_key.json"))
    a = ap.parse_args()
    key = json.load(open(a.key))
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    axes = {"faithful": "human_faithful", "correct": "human_correct", "acceptable": "human_acceptable"}
    human = {ax: {r["item_id"]: parse(r[col]) for r in rows} for ax, col in axes.items()}
    filled = {ax: sum(v is not None for v in d.values()) for ax, d in human.items()}
    print(f"rows: {len(rows)} | labelled per axis: {filled}")
    if not any(filled.values()):
        print("No human labels found yet - fill the human_* columns and re-run.")
        return
    out = ["# Human vs judge agreement", "", f"labelled items per axis: {filled}", ""]
    ids_all = [r["item_id"] for r in rows]
    planted = [i for i in ids_all if key["items"][i]["source"] == "planted"]
    # 1. human vs planted ground truth (sanity check of the human labels themselves)
    out.append("## Human vs planted ground truth (planted items only)")
    for ax in axes:
        truth = {i: key["items"][i]["truth"][ax] for i in planted if key["items"][i]["truth"].get(ax) is not None}
        r = compare(f"human_{ax} vs truth", human[ax], truth, planted)
        if r:
            out.append(f"- {r['name']}: n={r['n']} raw={r['raw']:.2f} kappa={r['kappa']:.2f}\n```\n{fmt_cm(r['cm'])}\n```")
    # 2. human vs each judge, on the judge's own axis (and the all-items / planted-only / real-only splits)
    out += ["", "## Human vs judge (rows = human [bad, good], cols = judge [bad, good])"]
    for jname, jd in sorted(key["judges"].items()):
        ax = jd["axis"]
        if ax not in human:
            continue
        for label, ids in (("all", ids_all), ("real-only", [i for i in ids_all if i not in planted]), ("planted-only", planted)):
            r = compare(f"{jname} [{ax}] ({label})", human[ax], jd["verdicts"], ids)
            if r:
                out.append(f"- **{r['name']}**: n={r['n']} raw={r['raw']:.2f} kappa={r['kappa']:.2f}\n```\n{fmt_cm(r['cm'])}\n```")
    # 3. where did human and the best end-to-end judge disagree?
    j = key["judges"].get("gemma:overall_pipeline") or key["judges"].get("gemma:overall_v0")
    if j:
        out += ["", "## Disagreements: human_acceptable vs end-to-end judge (look at these first)"]
        for r in rows:
            h, v = human["acceptable"].get(r["item_id"]), j["verdicts"].get(r["item_id"])
            if h is not None and v is not None and h != v:
                out.append(
                    f"- {r['item_id']} ({key['items'][r['item_id']]['etype']}): human={h} judge={v} | Q: {r['question'][:70]} | A: {r['system_answer'][:90]}"
                )
    text = "\n".join(out)
    print(text)
    (RUNS / "agreement_report.md").write_text(text)


if __name__ == "__main__":
    main()
