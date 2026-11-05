"""Export a sentence-transformers retriever to ONNX (fp32) and quantise it to dynamic int8 with ONNX Runtime.

The exported graph takes (input_ids, attention_mask) and returns the L2-normalised sentence embedding, i.e. pooling and
normalisation are inside the graph, so a deployment needs only `onnxruntime` + `tokenizers`.

    python -m deploy.export_onnx --model /path/to/model --out artifacts/v2-base [--variants fp32 int8 int8-mm]

Output layout (consumed by deploy.common.OnnxEncoder):
    OUT/fp32/model.onnx     exported graph
    OUT/int8/model.onnx     dynamic int8, MatMul weights *and* the embedding table (Gather) quantised
    OUT/int8-mm/model.onnx  dynamic int8, MatMul weights only (embedding table stays fp32; bigger, safer fallback)
    OUT/int8-pc/model.onnx  like int8 but with per-channel weight scales (usually the more accurate int8)
    OUT/int8-pc-mm/model.onnx  per-channel, MatMul weights only
    OUT/w8/model.onnx       weight-only 8-bit (block-wise, MatMulNBits) for the MatMuls + int8 embedding table; activations stay fp32
    OUT/tokenizer.json, OUT/config.json, OUT/export_meta.json

Exported with torch.onnx.export rather than optimum: optimum's ONNX exporter pins transformers<5 at the time of writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .common import MIN_MEM_GB, dir_size_mb, set_threads, wait_for_memory

OPSET = 17
# variant name -> (MatMul weights only?, per-channel scales?)
QUANT_VARIANTS = {"int8": (False, False), "int8-mm": (True, False), "int8-pc": (False, True), "int8-pc-mm": (True, True)}
PARITY_TEXTS = [
    "query: what is the capital of France?",
    "query: ما هي عاصمة المملكة العربية السعودية؟",
    "passage: Paris is the capital and most populous city of France, with an estimated population of 2,102,650 residents.",
    "passage: الرياض هي عاصمة المملكة العربية السعودية وأكبر مدنها من حيث عدد السكان، وتقع في وسط شبه الجزيرة العربية.",
    "short",
    "نص قصير",
]
# longer than anything in the tracing example, so that the dynamic axes are exercised in the parity check
PARITY_LONG = "passage: " + " ".join(["The quick brown fox jumps over the lazy dog while the cat sleeps by the window."] * 40)


def build_wrapper(st_model):
    """HF encoder + pooling + L2 normalisation as one torch module (pooling mode read from the sentence-transformers config)."""
    import torch
    import torch.nn.functional as F

    pooling = st_model[1].pooling_mode  # "mean" | "cls" in sentence-transformers >= 6
    if pooling not in ("mean", "cls"):
        raise ValueError(f"unsupported pooling mode {pooling!r}")

    class Wrapped(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if pooling == "cls":
                e = h[:, 0]
            else:
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return F.normalize(e, p=2, dim=-1)

    return Wrapped(st_model[0].auto_model).eval(), pooling


def export_fp32(model_path: str, out_dir: Path) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_path, device="cpu", model_kwargs={"attn_implementation": "eager"})
    wrapper, pooling = build_wrapper(st)
    enc = st.tokenizer(PARITY_TEXTS, padding=True, truncation=True, max_length=64, return_tensors="pt")
    dst = out_dir / "fp32"
    dst.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (enc["input_ids"], enc["attention_mask"]),
            str(dst / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
    for f in ("tokenizer.json", "config.json"):
        shutil.copy(Path(model_path) / f, out_dir / f)
    return {"pooling": pooling, "opset": OPSET, "hidden_size": int(st.get_embedding_dimension())}


def parity(model_path: str, onnx_path: Path, out_dir: Path) -> dict:
    """Cosine between the sentence-transformers embedding and the ONNX embedding on a few AR/EN strings."""
    from sentence_transformers import SentenceTransformer

    from .common import OnnxEncoder

    texts = [*PARITY_TEXTS, PARITY_LONG]
    st = SentenceTransformer(model_path, device="cpu")
    st.max_seq_length = 512  # the folder's default is the training length (256); the evaluation and the ONNX graph use 512
    ref = st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    got = OnnxEncoder(str(onnx_path), str(out_dir), prefix="none", threads=4).encode(texts, "p")
    cos = (ref * got).sum(1)
    return {"min_cosine_vs_torch": float(cos.min()), "mean_cosine_vs_torch": float(cos.mean())}


def quantize(src: Path, dst: Path, matmul_only: bool, per_channel: bool) -> None:
    """Dynamic int8 quantisation of the fp32 graph.

    ONNX Runtime's `quant_pre_process` (symbolic shape inference + optimisation) is deliberately NOT run: on these graphs the shape
    inference reports "Incomplete symbolic shape inference" and falls back anyway, and the attempt peaked at about 7 GB resident for the 278M model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        op_types_to_quantize=["MatMul"] if matmul_only else None,
    )


def quantize_weight_only(src: Path, dst: Path, bits: int = 8, block_size: int = 64) -> None:
    """Weight-only block-wise quantisation of the MatMul weights (MatMulNBits, symmetric) + int8 embedding table.

    Unlike dynamic int8 the activations are not quantised, which avoids the activation-outlier error that costs quality on the e5 models."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / "_mm.onnx"
    before = {f.name for f in Path.cwd().glob("*.data")}
    q = MatMulNBitsQuantizer(onnx.load(str(src)), bits=bits, block_size=block_size, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    for f in Path.cwd().glob(
        "*.data"
    ):  # the quantizer leaves one model-sized temporary "<uuid>.data" file per run in the working directory
        if f.name not in before:
            f.unlink()
    quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    tmp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="sentence-transformers folder (model.safetensors, tokenizer.json, 1_Pooling/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["fp32", *QUANT_VARIANTS, "w8"], choices=["fp32", *QUANT_VARIANTS, "w8"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-mem-gb", type=float, default=MIN_MEM_GB)
    args = ap.parse_args()
    set_threads(args.threads)
    wait_for_memory(args.min_mem_gb)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "export_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"source_model": str(args.model)})
    if "fp32" in args.variants or not (out / "fp32" / "model.onnx").exists():
        t0 = time.time()
        meta.update(export_fp32(args.model, out))
        meta["export_seconds"] = round(time.time() - t0, 1)
        meta["parity_fp32"] = parity(args.model, out / "fp32" / "model.onnx", out)
        print("[export] fp32", meta["parity_fp32"], flush=True)
    for name, (mm, pc) in QUANT_VARIANTS.items():
        if name in args.variants:
            quantize(out / "fp32" / "model.onnx", out / name / "model.onnx", matmul_only=mm, per_channel=pc)
            meta[f"size_mb_{name}"] = round(dir_size_mb(out / name), 1)
            print(f"[export] {name}: {meta[f'size_mb_{name}']} MiB", flush=True)
    if "w8" in args.variants:
        quantize_weight_only(out / "fp32" / "model.onnx", out / "w8" / "model.onnx")
        meta["size_mb_w8"] = round(dir_size_mb(out / "w8"), 1)
        print(f"[export] w8: {meta['size_mb_w8']} MiB", flush=True)
    meta["size_mb_fp32"] = round(dir_size_mb(out / "fp32"), 1)
    meta_path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
