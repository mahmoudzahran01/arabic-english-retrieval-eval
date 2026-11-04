"""Turn the raw deployment results (deploy/results/{quality,bench}/*.json) into tables: markdown for deploy/README.md, rows for the report.

Pure numpy / json: runs in CI without torch or onnxruntime.

    python -m deploy.summarize            # prints the markdown tables
    python -m deploy.summarize --write    # also refreshes the block between the RESULTS markers of deploy/README.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
REFERENCE = "torch-fp32"
VARIANT_ORDER = [
    "torch-fp32",
    "onnx-fp32",
    "onnx-int8",
    "onnx-int8-mm",
    "onnx-int8-pc",
    "onnx-int8-pc-mm",
    "onnx-w8",
    "onnx-int8+q-only",
    "onnx-int8-pc+q-only",
    "onnx-w8+q-only",
]
VARIANT_LABEL = {
    "torch-fp32": "PyTorch fp32 (reference)",
    "onnx-fp32": "ONNX fp32",
    "onnx-int8": "ONNX int8 (dynamic)",
    "onnx-int8-mm": "ONNX int8, MatMul only",
    "onnx-int8-pc": "ONNX int8, per-channel",
    "onnx-int8-pc-mm": "ONNX int8, per-channel, MatMul only",
    "onnx-w8": "ONNX weight-only 8-bit (MatMulNBits)",
    "onnx-int8+q-only": "int8 queries vs fp32 index",
    "onnx-int8-pc+q-only": "int8 per-channel queries vs fp32 index",
    "onnx-w8+q-only": "w8 queries vs fp32 index",
}
DIRECTIONS = ("ar->ar", "en->en", "ar->en", "en->ar")
BOOT = 2000


def _direction_of(slice_name: str, dir_of: dict[str, str]) -> str:
    return dir_of[slice_name]


def default_direction_map() -> dict[str, str]:
    from arretrieval.retrieval.evalkit import DIRECTION_OF

    return dict(DIRECTION_OF)


def load_quality(results: Path = RESULTS) -> dict[tuple[str, str], dict]:
    """{(model, variant): {"json": ..., "perq": {slice: array}}} for the TEST split."""
    out = {}
    for p in sorted((results / "quality").glob("*.test.json")):
        j = json.loads(p.read_text())
        z = np.load(p.parent / (p.name[: -len(".json")] + ".perq.npz"))
        out[(j["model"], j["variant"])] = {"json": j, "perq": {k: z[k].astype(np.float64) for k in z.files}}
    return out


def load_bench(results: Path = RESULTS, sub: str = "bench") -> dict[tuple[str, str, int], dict]:
    out = {}
    for p in sorted((results / sub).glob("*.bench.json")):
        j = json.loads(p.read_text())
        out[(j["model"], j["variant"], j["threads"])] = j
    return out


def macro_bootstrap(perq_a: dict, perq_b: dict | None, dir_of: dict[str, str], n_boot: int = BOOT, seed: int = 13) -> dict[str, dict]:
    """Percentile bootstrap over queries (resampled within each slice, same draws for A and B) of mean(B) - mean(A) per direction and macro.

    With perq_b = None returns the CI of A itself. Values are fractions (MRR@10 in [0, 1])."""
    slices = sorted(perq_a)
    boot_a: dict[str, list] = {}
    boot_b: dict[str, list] = {}
    for i, sl in enumerate(slices):
        a = perq_a[sl]
        idx = np.random.default_rng([seed, i]).integers(0, len(a), size=(n_boot, len(a)))
        d = dir_of[sl]
        boot_a.setdefault(d, []).append(a[idx].mean(axis=1))
        if perq_b is not None:
            boot_b.setdefault(d, []).append(perq_b[sl][idx].mean(axis=1))

    def finish(per_dir: dict[str, list]) -> dict[str, np.ndarray]:
        out = {d: np.mean(v, axis=0) for d, v in per_dir.items()}
        out["macro"] = np.mean([out[d] for d in out], axis=0)
        return out

    fa = finish(boot_a)
    f = fa if perq_b is None else {k: finish(boot_b)[k] - fa[k] for k in fa}
    res = {}
    for k, v in f.items():
        res[k] = {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))}
    return res


def point_estimate(perq: dict, dir_of: dict[str, str]) -> dict[str, float]:
    by_dir: dict[str, list] = {}
    for sl, v in perq.items():
        by_dir.setdefault(dir_of[sl], []).append(float(v.mean()))
    out = {d: float(np.mean(v)) for d, v in by_dir.items()}
    out["macro"] = float(np.mean(list(out.values())))
    return out


def quality_rows(quality: dict, model: str, dir_of: dict[str, str]) -> list[dict]:
    ref = quality.get((model, REFERENCE))
    rows = []
    for variant in VARIANT_ORDER:
        q = quality.get((model, variant))
        if q is None:
            continue
        pe = point_estimate(q["perq"], dir_of)
        row = {"model": model, "variant": variant, "mrr": pe}
        if ref is not None and variant != REFERENCE:
            pe_ref = point_estimate(ref["perq"], dir_of)
            ci = macro_bootstrap(ref["perq"], q["perq"], dir_of)
            row["delta"] = {k: pe[k] - pe_ref[k] for k in pe}
            row["delta_ci"] = ci
        rows.append(row)
    return rows


def _f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def markdown_quality(quality: dict, model: str, dir_of: dict[str, str]) -> str:
    lines = [
        f"**{model}**: test MRR@10 x100 (CPU, max length 512). Delta = variant minus PyTorch fp32, with a 95% paired-bootstrap CI over queries.",
        "",
        "| variant | ar->ar | en->en | ar->en | en->ar | macro | delta macro [95% CI] | delta ar->en [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in quality_rows(quality, model, dir_of):
        m = r["mrr"]
        cells = [f"{100 * m[d]:.2f}" for d in (*DIRECTIONS, "macro")]
        if "delta" in r:
            d, ci = r["delta"], r["delta_ci"]
            cells.append(f"{100 * d['macro']:+.2f} [{100 * ci['macro']['lo']:+.2f}, {100 * ci['macro']['hi']:+.2f}]")
            cells.append(f"{100 * d['ar->en']:+.2f} [{100 * ci['ar->en']['lo']:+.2f}, {100 * ci['ar->en']['hi']:+.2f}]")
        else:
            cells += ["reference", "reference"]
        lines.append(f"| {VARIANT_LABEL[r['variant']]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_bench(bench: dict, model: str, threads: int, where: str = "") -> str:
    lines = [
        f"**{model}**{where}, {threads} CPU threads. Latency in ms per call (p50 / p95), throughput in sentences per second.",
        "",
        "| variant | size (MiB) | RSS after load / peak (MiB) | query b=1 p50 / p95 | query b=32 sent/s | passage b=1 p50 / p95 | passage b=32 p50 / p95 | passage b=32 sent/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        b = bench.get((model, variant, threads))
        if b is None:
            continue
        w = b["workloads"]
        lines.append(
            f"| {VARIANT_LABEL[variant]} | {b['size_mb']:.0f} | {b.get('rss_after_load_mb', float('nan')):.0f} / {b['peak_rss_mb']:.0f} | "
            f"{w['query_b1']['p50_ms']:.1f} / {w['query_b1']['p95_ms']:.1f} | {w['query_b32']['throughput_sent_per_s']:.0f} | "
            f"{w['passage_b1']['p50_ms']:.0f} / {w['passage_b1']['p95_ms']:.0f} | "
            f"{w['passage_b32']['p50_ms']:.0f} / {w['passage_b32']['p95_ms']:.0f} | {w['passage_b32']['throughput_sent_per_s']:.1f} |"
        )
    return "\n".join(lines)


def build_markdown(results: Path = RESULTS) -> str:
    dir_of = default_direction_map()
    quality, bench = load_quality(results), load_bench(results)
    models = sorted({m for m, _ in quality} | {m for m, _, _ in bench})
    parts = []
    for model in models:
        parts.append(markdown_quality(quality, model, dir_of))
        for t in sorted({t for (m, _, t) in bench if m == model}, reverse=True):
            parts.append(markdown_bench(bench, model, t, " (Spark, ARM server CPU)"))
    mac = load_bench(results, "bench_mac")
    for model in sorted({m for m, _, _ in mac}):
        for t in sorted({t for (m, _, t) in mac if m == model}):
            parts.append(markdown_bench(mac, model, t, " (Apple M3 Pro laptop CPU, the two 8-bit variants of the small model only)"))
    return "\n\n".join(parts)


MARK = re.compile(r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)", re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the results block of deploy/README.md")
    args = ap.parse_args()
    md = build_markdown()
    print(md)
    if args.write:
        readme = Path(__file__).resolve().parent / "README.md"
        text = readme.read_text()
        if not MARK.search(text):
            raise SystemExit("README.md has no RESULTS markers")
        readme.write_text(MARK.sub(lambda m: m.group(1) + md + m.group(2), text))


if __name__ == "__main__":
    main()
