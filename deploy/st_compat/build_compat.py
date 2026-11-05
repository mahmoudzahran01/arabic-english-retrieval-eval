"""Rebuild a model saved with sentence-transformers 6 as a folder that sentence-transformers 5.x can load.

ST 6 writes `sentence_transformers.base.*` / `sentence_transformers.sentence_transformer.*` module paths into modules.json, which ST 5.x cannot import.
The weights and the tokenizer are plain Hugging Face files, so the fix is to rebuild the three modules with the ST 5 classes and save again:

    python build_compat.py --src MODEL_DIR --dst OUT_DIR --pooling mean|cls --max-seq-length 256

Run it with an ST 5.x environment (tested with 5.3.0). The pooling mode and the maximum length are the ones the ST 6 model reports; `check_compat.py` verifies
that both sides agree. The model card of the source folder is copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer, models


def build(src: Path, dst: Path, pooling: str, max_seq_length: int) -> None:
    dim = int(json.loads((src / "1_Pooling" / "config.json").read_text())["embedding_dimension"])
    transformer = models.Transformer(str(src), max_seq_length=max_seq_length)
    model = SentenceTransformer(modules=[transformer, models.Pooling(dim, pooling_mode=pooling), models.Normalize()])
    model.save(str(dst))
    if (src / "README.md").exists():
        shutil.copy(src / "README.md", dst / "README.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--pooling", choices=["mean", "cls"], required=True)
    ap.add_argument("--max-seq-length", type=int, required=True)
    a = ap.parse_args()
    build(a.src, a.dst, a.pooling, a.max_seq_length)
    print(f"[build] {a.src.name} -> {a.dst} (pooling {a.pooling}, max_seq_length {a.max_seq_length})")


if __name__ == "__main__":
    main()
