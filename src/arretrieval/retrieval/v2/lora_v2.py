"""LoRA helpers for the sentence-transformers XLM-R encoder (used by train_v2 --lora) and a self-test.

Why a module: the first LoRA run (m3_lo_s13) trained for 56 min and then crashed at the merge because the code looked the PeftModel up again through the sentence-transformers
module (which returned the plain, in-place-injected XLMRobertaModel).  The handle returned by get_peft_model is now kept, the adapter is written to disk BEFORE the merge, and the
merge is verified (no lora_ parameter left).   Self-test (GPU, gated):  python -m arretrieval.retrieval.v2.lora_v2
"""

from __future__ import annotations

import sys
import tempfile


def transformer_of(model):
    m0 = model[0]
    return getattr(m0, "auto_model", None) or getattr(m0, "model")


def wrap(model, rank: int, alpha: int = 0):
    """Inject LoRA adapters (attention q/k/v + all dense layers: attention output, intermediate, output) IN PLACE and return the PeftModel handle."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=0.05, bias="none", target_modules=["query", "key", "value", "dense"]
    )
    handle = get_peft_model(transformer_of(model), cfg)
    handle.print_trainable_parameters()
    return handle


def merge(model, handle, adapter_dir: str | None = None) -> None:
    """Fold the adapters into the weights in place (the sentence-transformers module keeps referencing the same, now plain, encoder)."""
    if adapter_dir:
        handle.save_pretrained(adapter_dir)
    handle.merge_and_unload()
    left = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert not left, f"adapters still present after merge: {left[:3]}"


def selftest() -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    texts = [
        "ما هي عاصمة المملكة العربية السعودية؟",
        "What is the capital of Saudi Arabia?",
        "الرياض هي العاصمة",
        "Riyadh is the capital city.",
    ]
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.max_seq_length = 64
    base = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    handle = wrap(model, 4)
    with torch.no_grad():  # make the adapters non-trivial (B starts at zero)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.02)
    model.eval()
    with_adapter = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    assert float(np.min(np.sum(base * with_adapter, 1))) < 0.9999, "adapters had no effect - test is vacuous"
    merge(model, handle, adapter_dir=tempfile.mkdtemp())
    merged = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    cos = np.sum(with_adapter * merged, 1)
    print("cos(adapter-active, merged) =", np.round(cos, 5).tolist(), flush=True)
    assert float(cos.min()) > 0.999, "merge changed the function"
    d = tempfile.mkdtemp()
    model.save(d)
    reloaded = SentenceTransformer(d, device="cuda")
    reloaded.max_seq_length = 64
    again = reloaded.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    assert float(np.sum(merged * again, 1).min()) > 0.999, "save/reload changed the function"
    print("LORA SELFTEST OK", flush=True)


if __name__ == "__main__":
    selftest()
    sys.exit(0)
