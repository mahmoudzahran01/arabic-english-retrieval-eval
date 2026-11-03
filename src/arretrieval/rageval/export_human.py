"""Export the 40-item human-labelling sample (20 real RAG answers + 20 planted-error items, stratified, shuffled).

  python -m arretrieval.rageval.export_human
Writes  runs/rageval/human_labeling.csv       <- give this to the human; label columns are BLANK
        runs/rageval/human_labeling_key.json  <- planted truth + judge verdicts; DO NOT open before labelling
The CSV deliberately does not say which items are planted.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict

from .paths import DATA, RUNS

PLANTED_QUOTA = {
    "clean": 2,
    "clean_numeral": 1,
    "wrong_number": 2,
    "wrong_entity": 2,
    "unsupported_claim": 2,
    "omitted_fact": 2,
    "wrong_language": 2,
    "wrong_citation": 2,
    "correct_abstention": 1,
    "wrong_abstention": 2,
    "halluc_unanswerable": 2,
}
REAL_QUOTA = {("en", "mono"): 4, ("en", "cross"): 4, ("ar", "mono"): 4, ("ar", "cross"): 4, ("unans", "-"): 4}
JUDGE_AXIS = {
    "faith_v0": "faithful",
    "faith_v1": "faithful",
    "faith_v2": "faithful",
    "faith_v3": "faithful",
    "cite_v1": "citation_ok",
    "correct_v0": "correct",
    "correct_v1": "correct",
    "overall_v0": "acceptable",
    "overall_pipeline": "acceptable",
    "overall_pipeline_v3": "acceptable",
}


def load_verdicts(dirpath) -> dict:
    out = {}
    for p in dirpath.glob("*.jsonl") if dirpath.exists() else []:
        out[p.stem] = {json.loads(l)["id"]: json.loads(l)["pass"] for l in open(p)}
    return out


def main(config: str = "A_baseline", seed: int = 2026):
    rng = random.Random(seed)
    planted = [i for i in (json.loads(l) for l in open(DATA / "meta_items.jsonl")) if not i.get("excluded")]
    real = [json.loads(l) for l in open(RUNS / f"answers_{config}.jsonl")]
    sel = []  # (source, item)
    by_type = defaultdict(list)
    for it in planted:
        by_type[it["etype"]].append(it)
    for et, n in PLANTED_QUOTA.items():
        pool = [i for i in by_type[et] if i["split"] == "test"] or by_type[et]
        rng.shuffle(pool)
        sel += [("planted", i) for i in pool[:n]]
    cells = defaultdict(list)
    for r in real:
        cells[("unans", "-") if not r["answerable"] else (r["q_lang"], r["kind"])].append(r)
    for c, n in REAL_QUOTA.items():
        rng.shuffle(cells[c])
        sel += [("real", r) for r in cells[c][:n]]
    rng.shuffle(sel)

    gem_meta, qw_meta = load_verdicts(RUNS / "judge_gemma"), load_verdicts(RUNS / "judge_qwen")
    gem_real = load_verdicts(RUNS / f"judge_rag_{config}_gemma")
    for j, v in load_verdicts(RUNS / f"judge_rag_{config}_gemma_humanrows").items():  # end-to-end judges run on the 20 real rows only
        gem_real.setdefault(j, {}).update(v)
    qw_real = load_verdicts(RUNS / f"judge_rag_{config}_qwen")
    rows, key = [], {"items": {}, "judges": defaultdict(lambda: {"axis": None, "verdicts": {}}), "config": config}
    for n, (src, it) in enumerate(sel, 1):
        hid = f"H{n:02d}"
        if src == "planted":
            ctx, ans, ref, q, ql = it["contexts"], it["answer"], it["reference"], it["question"], it["q_lang"]
            truth = {
                "faithful": it["labels"]["faithful"],
                "correct": it["labels"]["correct"],
                "acceptable": it["acceptable"],
                "citation_ok": it["labels"]["citation_ok"],
            }
            key["items"][hid] = {"source": "planted", "src_id": it["id"], "etype": it["etype"], "truth": truth}
            for j, v in gem_meta.items():
                key["judges"][f"gemma:{j}"]["verdicts"][hid] = v.get(it["id"])
            for j, v in qw_meta.items():
                key["judges"][f"qwen:{j}"]["verdicts"][hid] = v.get(it["id"])
        else:
            ctx = [{"id": p["id"], "text": p["text"]} for p in it["retrieved"]]
            ans, q, ql = it["answer"], it["question"], it["q_lang"]
            ref = it["reference"] if it["answerable"] else "(no answer exists in the documents)"
            key["items"][hid] = {"source": "real", "src_id": it["qid"], "etype": f"real:{it['q_lang']}|{it['kind']}", "truth": None}
            for j, v in gem_real.items():
                key["judges"][f"gemma:{j}"]["verdicts"][hid] = v.get(it["qid"])
            for j, v in qw_real.items():
                key["judges"][f"qwen:{j}"]["verdicts"][hid] = v.get(it["qid"])
        row = {"item_id": hid, "question": q, "question_language": ql, "reference_answer": ref}
        for k in range(3):
            row[f"document_{k + 1}"] = f"[{ctx[k]['id']}] {ctx[k]['text']}" if k < len(ctx) else ""
        row.update({"system_answer": ans, "human_faithful": "", "human_correct": "", "human_acceptable": "", "human_notes": ""})
        rows.append(row)
    for name, d in key["judges"].items():
        d["axis"] = JUDGE_AXIS[name.split(":")[1]]
    key["judges"] = dict(key["judges"])
    with open(RUNS / "human_labeling.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(key, open(RUNS / "human_labeling_key.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rows)} rows ({sum(1 for s, _ in sel if s == 'planted')} planted / {sum(1 for s, _ in sel if s == 'real')} real)")


if __name__ == "__main__":
    main()
