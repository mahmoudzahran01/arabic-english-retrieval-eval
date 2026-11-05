"""Load model folders with the sentence-transformers that is installed here and record what it sees plus the embeddings of fixed texts (CPU only).

    python check_compat.py --label st6 --out OUT_DIR  NAME=MODEL_DIR [NAME=MODEL_DIR ...]

Writes OUT_DIR/<label>__<NAME>.json (loads or not, library versions, max_seq_length, pooling mode, dimension, module classes of modules.json)
and OUT_DIR/<label>__<NAME>.npz (embeddings). A folder that cannot be loaded is recorded with the error instead of aborting.
`compare_compat.py` compares the files of two environments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TEXTS = [
    "query: ما هي عاصمة السعودية؟",
    "passage: الرياض هي عاصمة المملكة العربية السعودية وأكبر مدنها.",
    "query: who wrote the quran tafsir",
    "passage: The Hijaz is a region in the west of Saudi Arabia that includes Jeddah, Mecca and Medina.",
    "query: كيف أجدد رخصة القيادة؟ how to renew a driving licence",
    "passage: بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
    "query: سعر الصرف 3.75 ريال/دولار في 2024-05-01؟",
    "مرحبا",
    "A short sentence without any prefix.",
    "query: 😀 رحلة إلى جدة",
    "passage: def f(x): return x**2  # دالة التربيع",
    "passage: " + "الكلمة " * 600,  # far beyond the maximum length: checks truncation
    "passage: " + "The quick brown fox jumps over the lazy dog. " * 120,
]
BATCH = 4


def pooling_mode(module) -> str:
    getter = getattr(module, "get_pooling_mode_str", None)
    if callable(getter):  # ST 5.x
        return str(getter())
    return str(getattr(module, "pooling_mode", "?"))  # ST 6.x


def check(path: Path) -> tuple[dict, np.ndarray | None]:
    import sentence_transformers as st
    import torch

    torch.set_num_threads(4)
    info: dict = {
        "path": str(path),
        "sentence_transformers": st.__version__,
        "torch": torch.__version__,
        "module_classes": [m["type"] for m in json.loads((path / "modules.json").read_text())],
    }
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(path), device="cpu")
        emb = model.encode(TEXTS, batch_size=BATCH, convert_to_numpy=True)
        info.update(
            loads=True,
            max_seq_length=int(model.max_seq_length),
            pooling=pooling_mode(model[1]),
            dimension=int(model.get_sentence_embedding_dimension()),
            norm_min=float(np.linalg.norm(emb, axis=1).min()),
            norm_max=float(np.linalg.norm(emb, axis=1).max()),
        )
        return info, emb.astype(np.float32)
    except Exception as e:  # recorded, not fatal: ST 5.x is expected to fail on ST 6 folders
        info.update(loads=False, error=f"{type(e).__name__}: {str(e)[:240]}")
        return info, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("models", nargs="+", help="NAME=MODEL_DIR")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.models:
        name, _, folder = spec.partition("=")
        info, emb = check(Path(folder))
        (a.out / f"{a.label}__{name}.json").write_text(json.dumps(info, indent=1))
        if emb is not None:
            np.savez(a.out / f"{a.label}__{name}.npz", emb=emb)
        print(f"[check] {a.label} {name}: " + ("loads" if info["loads"] else f"FAILS ({info['error'][:80]})"), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
