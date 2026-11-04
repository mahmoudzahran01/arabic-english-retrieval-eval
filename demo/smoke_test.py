"""Smoke test of the demo: 5 Arabic + 5 English queries through the same code path as the UI, then one call through the running Gradio app.

    python demo/smoke_test.py --out demo/smoke_test.md [--no-ui]

Queries marked (gold) come from the training pool and have a known answer paragraph in the corpus, so the rank of that paragraph is
checked; the others are free-text questions whose top results are listed for a manual look. Because the corpus consists of training-pool
paragraphs, the gold checks show that the pipeline works end to end, not how well the models generalise.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import app as demo_app  # noqa: E402

GOLD_QUERIES = {
    "ar": ['من مؤلف رواية " كوخ العم توم " ؟', "اين توجد حدائق بابل المعلقة ؟", "كم عاش ابن خلدون وأين دفن؟"],
    "en": ["In what region of France is Dieppe located?", "When was economist James Tobin born?", "When was Spacetoon TV channel founded?"],
}
FREE_QUERIES = {
    "ar": ["ما هي عاصمة المملكة العربية السعودية؟", "من اخترع المصباح الكهربائي؟"],
    "en": ["Who was the first president of the United States?", "What is the largest planet in the solar system?"],
}


def run_queries(demo, k: int = 5) -> list[dict]:
    gold = {e["query"]: e for e in json.loads((demo_app.INDEX / "examples.json").read_text())}
    models = ("v2", *[demo_app.BASELINES[b][0] for b in demo.available_baselines()])
    demo.search("warm-up", 1, None, models)  # the first call of a model loads it; keep that out of the timings
    rows = []
    for lang in ("ar", "en"):
        for q in [*GOLD_QUERIES[lang], *FREE_QUERIES[lang]]:
            res = demo.search(q, k, None, models)
            g = gold.get(q)
            row = {"lang": lang, "query": q, "gold_pid": g["gold_pid"] if g else None, "slice": g["slice"] if g else None, "models": {}}
            for m in models:
                pids = [p.pid for p, _ in res[m]["hits"]]
                row["models"][m] = {
                    "encode_ms": round(res[m]["encode_ms"]),
                    "gold_rank": (pids.index(g["gold_pid"]) + 1) if g and g["gold_pid"] in pids else None,
                    "top": [(p.lang, str(p.title), round(s, 3), p.text[:110].replace("\n", " ")) for p, s in res[m]["hits"][:3]],
                }
            rows.append(row)
    return rows


def ui_roundtrip(demo) -> str:
    """Launch the real Gradio app in-process and call it through gradio_client, as a browser would."""
    from gradio_client import Client

    ui = demo_app.build_ui(demo)
    _app, url, _ = ui.launch(prevent_thread_lock=True, server_name="127.0.0.1", server_port=None, quiet=True)
    try:
        t0 = time.perf_counter()
        left, right, note = Client(url, verbose=False).predict(
            "When was economist James Tobin born?", 3, "both languages", demo.available_baselines()[0], api_name="/search"
        )
        ok = "Tobin" in left or "tobin" in left.lower() or "AR" in left or "EN" in left
        return f"UI round trip via gradio_client at {url}: {time.perf_counter() - t0:.2f} s, left panel {len(left)} chars, right panel {len(right)} chars, '{note}', rendered={ok}"
    finally:
        ui.close()


def to_markdown(rows: list[dict], header: str, ui_line: str) -> str:
    out = [f"# Demo smoke test\n\n{header}\n\n{ui_line}\n"]
    for lang in ("ar", "en"):
        out.append(f"\n## {'Arabic' if lang == 'ar' else 'English'} queries\n")
        for r in (r for r in rows if r["lang"] == lang):
            tag = f" (gold: {r['slice']})" if r["gold_pid"] else " (free text)"
            out.append(f"\n**{r['query']}**{tag}\n")
            for m, v in r["models"].items():
                gr_ = "" if r["gold_pid"] is None else (f", gold at rank {v['gold_rank']}" if v["gold_rank"] else ", gold NOT in top-5")
                out.append(f"- `{m}` ({v['encode_ms']} ms{gr_})")
                out.extend(f"    {i}. [{lg.upper()}] {t} ({s}) {x}…" for i, (lg, t, s, x) in enumerate(v["top"], 1))
    gold_rows = [r for r in rows if r["gold_pid"]]
    out.append("\n## Summary\n\n| model | gold in top-5 | gold at rank 1 | median query encode (ms) |\n|---|---:|---:|---:|")
    for m in rows[0]["models"]:
        hit5 = sum(r["models"][m]["gold_rank"] is not None for r in gold_rows)
        hit1 = sum(r["models"][m]["gold_rank"] == 1 for r in gold_rows)
        ms = sorted(r["models"][m]["encode_ms"] for r in rows)[len(rows) // 2]
        out.append(f"| {m} | {hit5}/{len(gold_rows)} | {hit1}/{len(gold_rows)} | {ms} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "smoke_test.md"))
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--threads", type=int, default=2, help="2 = the CPU of a free Hugging Face Space")
    args = ap.parse_args()
    demo = demo_app.Demo(threads=args.threads)
    rows = run_queries(demo)
    header = (
        f"Run on {platform.machine()} {platform.system()} CPU, {args.threads} threads; v2 queries use the `{demo.v2_variant}` ONNX graph, "
        f"corpus of {len(demo.passages)} passages (precomputed embeddings), top-5 per model. "
        f"Load average of the (shared) machine during the run: {os.getloadavg()[0]:.1f}, so the latencies are indicative."
    )
    ui_line = "UI round trip skipped." if args.no_ui else ui_roundtrip(demo)
    Path(args.out).write_text(to_markdown(rows, header, ui_line))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
