"""Bilingual Arabic/English search demo: v2 (ONNX, CPU) side by side with a baseline.

Type an Arabic or an English question; the app encodes it on the CPU and shows the top-k passages of a small mixed Arabic + English
corpus for both models, with the language of every passage marked. Corpus embeddings are precomputed (demo/build_index.py), so a query
costs one forward pass of a short text.

    python demo/app.py                 # http://localhost:7860
Environment: DEMO_INDEX (default demo/index), V2_ONNX_DIR (default demo/models/v2-base), V2_ONNX_VARIANT (default fp32).
"""

from __future__ import annotations

import html
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engines import E5_PREFIXES, NO_PREFIXES, HFEmbedder, OnnxEmbedder, top_k  # noqa: E402

HERE = Path(__file__).resolve().parent
INDEX = Path(os.environ.get("DEMO_INDEX", HERE / "index"))
V2_DIR = Path(os.environ.get("V2_ONNX_DIR", HERE / "models" / "v2-base"))
V2_VARIANT = os.environ.get("V2_ONNX_VARIANT", "fp32")
QUERY_MAX_LEN = 64
V2_NAME = "v2 (e5-base distilled, ONNX)"
BASELINES = {  # label -> (index key, HF id, pooling, prefixes)
    "bge-m3 (568M, PyTorch)": ("bge-m3", "BAAI/bge-m3", "cls", NO_PREFIXES),
    "v1 (e5-small fine-tune)": ("v1", "mzahran1/arabic-english-e5-small-retrieval", "mean", E5_PREFIXES),
}
NONE = "none"
LANG_FILTERS = {"both languages": None, "Arabic passages only": "ar", "English passages only": "en"}
ARABIC = re.compile(r"[؀-ۿ]")
EXAMPLES = [  # answerable from the demo corpus (questions of the training pool, see README for the caveat)
    'من مؤلف رواية " كوخ العم توم " ؟',
    "اين توجد حدائق بابل المعلقة ؟",
    "متى ولد المفكر طه حسين؟",
    "كم عاش ابن خلدون وأين دفن؟",
    'متى صُنع فيلم "Airport"؟',
    "In what region of France is Dieppe located?",
    "When was economist James Tobin born?",
    "How many episodes did Haven have?",
    "When was Spacetoon TV channel founded?",
    "How long is the border strip between Kuwait and Iraq?",
]


def query_language(text: str) -> str:
    """'ar' if at least a third of the letters are Arabic script, else 'en' (only used for the cross-lingual tag)."""
    letters = [c for c in text if c.isalpha()]
    return "ar" if letters and sum(bool(ARABIC.match(c)) for c in letters) / len(letters) >= 1 / 3 else "en"


class Demo:
    def __init__(self, index: Path = INDEX, v2_dir: Path = V2_DIR, v2_variant: str = V2_VARIANT, threads: int | None = None):
        self.passages = pd.read_parquet(index / "passages.parquet")
        self.lang = self.passages.lang.to_numpy()
        self.emb = {p.stem[4:]: np.load(p).astype(np.float32) for p in sorted(index.glob("emb_*.npy"))}
        self.v2_dir, self.v2_variant = v2_dir, v2_variant
        self.threads = threads or max(os.cpu_count() or 2, 1)
        self.encoders: dict[str, object] = {}

    def available_baselines(self) -> list[str]:
        return [label for label, (key, *_rest) in BASELINES.items() if key in self.emb]

    def encoder(self, key: str):
        if key not in self.encoders:
            if key == "v2":
                self.encoders[key] = OnnxEmbedder(self.v2_dir, self.v2_variant, E5_PREFIXES, QUERY_MAX_LEN, self.threads)
            else:
                label = next(lb for lb, v in BASELINES.items() if v[0] == key)
                _, hf_id, pooling, prefixes = BASELINES[label]
                self.encoders[key] = HFEmbedder(
                    os.environ.get(f"BASELINE_{key.upper().replace('-', '_')}", hf_id), pooling, prefixes, QUERY_MAX_LEN, self.threads
                )
        return self.encoders[key]

    def search(self, query: str, k: int = 5, lang: str | None = None, models: tuple[str, ...] = ("v2",)) -> dict[str, dict]:
        allowed = None if lang is None else (self.lang == lang)
        out = {}
        for key in models:
            t0 = time.perf_counter()
            qv = self.encoder(key).encode([query], "q")[0]
            t_enc = (time.perf_counter() - t0) * 1000
            hits = top_k(qv, self.emb[key], k, allowed)
            out[key] = {"encode_ms": t_enc, "hits": [(self.passages.iloc[i], s) for i, s in hits]}
        return out


# ------------------------------------------------------------------------------------------------ rendering
CARD = "border:1px solid #8884;border-radius:8px;padding:8px 10px;margin:6px 0;"
BADGE = "display:inline-block;font-size:11px;font-weight:700;border-radius:4px;padding:1px 6px;margin-right:6px;color:#fff;background:{c};"


def render_hits(title: str, res: dict, qlang: str, k_shown: int) -> str:
    head = f"<div style='font-weight:700;margin-bottom:2px'>{html.escape(title)}</div>"
    head += f"<div style='font-size:12px;opacity:.7'>query encoded in {res['encode_ms']:.0f} ms (CPU)</div>"
    cards = []
    for rank, (p, score) in enumerate(res["hits"][:k_shown], 1):
        ar = p.lang == "ar"
        text = p.text if len(p.text) <= 480 else p.text[:480].rsplit(" ", 1)[0] + " …"
        cross = "" if p.lang == qlang else "<span style='" + BADGE.format(c="#b8860b") + "'>cross-lingual</span>"
        direction = "rtl" if ar else "ltr"
        cards.append(
            f"<div style='{CARD}'><div><span style='font-size:12px;opacity:.7'>#{rank} · {score:.3f}</span> "
            f"<span style='{BADGE.format(c='#0b6e4f' if ar else '#1f4e9c')}'>{'AR' if ar else 'EN'}</span>{cross}"
            f"<b dir='{direction}'>{html.escape(str(p.title))}</b></div>"
            f"<div dir='{direction}' lang='{p.lang}' style='margin-top:4px;font-size:14px;line-height:1.5;text-align:{'right' if ar else 'left'}'>{html.escape(text)}</div></div>"
        )
    return head + "".join(cards)


def run(demo: Demo, query: str, k: int, lang_label: str, baseline_label: str) -> tuple[str, str, str]:
    query = (query or "").strip()
    if not query:
        return "<i>Type a question in Arabic or English.</i>", "", ""
    lang = LANG_FILTERS[lang_label]
    qlang = query_language(query)
    keys: tuple[str, ...] = ("v2",) if baseline_label == NONE else ("v2", BASELINES[baseline_label][0])
    res = demo.search(query, int(k), lang, keys)
    left = render_hits(V2_NAME, res["v2"], qlang, int(k))
    if baseline_label == NONE:
        return left, "", f"query language detected: {'Arabic' if qlang == 'ar' else 'English'}"
    other = res[keys[1]]
    right = render_hits(baseline_label, other, qlang, int(k))
    a = {p.pid for p, _ in res["v2"]["hits"]}
    b = {p.pid for p, _ in other["hits"]}
    note = f"query language detected: {'Arabic' if qlang == 'ar' else 'English'} · overlap of the two top-{int(k)} lists: {len(a & b)}/{int(k)}"
    return left, right, note


def build_ui(demo: Demo):
    import gradio as gr

    baselines = demo.available_baselines()
    intro = (
        "### Arabic ⇄ English retrieval: v2 vs a baseline\n"
        "Ask in **Arabic or English**; passages come from a small mixed corpus "
        f"({len(demo.passages):,} Wikipedia-based paragraphs, {int((demo.lang == 'ar').sum()):,} Arabic and {int((demo.lang == 'en').sum()):,} English). "
        "Cross-lingual hits (Arabic question → English passage and the reverse) carry a **cross-lingual** tag."
    )
    with gr.Blocks(title="Arabic-English retrieval: v2 demo") as ui:
        gr.Markdown(intro)
        with gr.Row():
            q = gr.Textbox(
                label="Question (Arabic or English)", placeholder="متى ولد طه حسين؟  /  When was James Tobin born?", scale=4, rtl=False
            )
            go = gr.Button("Search", variant="primary", scale=1)
        with gr.Row():
            k = gr.Slider(1, 10, value=5, step=1, label="Passages shown")
            lang = gr.Radio(list(LANG_FILTERS), value="both languages", label="Passages to search")
            base = gr.Radio([*baselines, NONE], value=baselines[0] if baselines else NONE, label="Compare v2 with")
        note = gr.Markdown()
        with gr.Row():
            left, right = gr.HTML(), gr.HTML()
        gr.Examples(EXAMPLES, inputs=[q], label="Examples (questions whose answer is in the corpus)")
        gr.Markdown(
            "**Read this as a demo, not a benchmark.** The corpus is made of *training-pool* paragraphs (TyDi QA, SQuAD v1.1, ARCD), which the "
            "fine-tuned models saw during training; the held-out numbers are in the technical report. "
            "Data licences: TyDi QA Apache-2.0, SQuAD CC BY-SA 4.0, ARCD MIT; paragraph text is Wikipedia (CC BY-SA)."
        )
        fn = lambda *a: run(demo, *a)  # noqa: E731
        go.click(fn, [q, k, lang, base], [left, right, note], api_name="search")
        q.submit(fn, [q, k, lang, base], [left, right, note], api_name=False)
    return ui


if __name__ == "__main__":
    build_ui(Demo()).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
