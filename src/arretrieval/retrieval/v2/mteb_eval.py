"""Out-of-domain evaluation with the `mteb` package (2.26.33, private install in runs/retrieval/v2/pylibs).

Models are loaded as sentence-transformers (fp16, max_seq_length forced to 512 for every model so that nobody is truncated more than
the others; v1/v2 were saved with 256) and wrapped with the e5 prompts ("query: " / "passage: ") where the model needs them.

Run (GPU): python -m arretrieval.retrieval.v2.mteb_eval --models e5-small v1 v2=/lab/runs/.../model bge-m3 --suite core
Results: runs/retrieval/v2/mteb/{model}/{task}.json  (+ the mteb ResultCache under runs/retrieval/v2/mteb_cache)
"""

from __future__ import annotations

import argparse
import json

import torch

from .common_v2 import V2

OUT = V2 / "mteb"
E5P = {"query": "query: ", "document": "passage: "}
KNOWN = {  # tag -> (path / hf id, prompts)
    "e5-small": ("intfloat/multilingual-e5-small", E5P),
    "e5-base": ("intfloat/multilingual-e5-base", E5P),
    "bge-m3": ("BAAI/bge-m3", None),
    "v1": (
        "mzahran1/arabic-english-e5-small-retrieval",
        E5P,
    ),  # downloaded from the Hub (read-only), same weights as release-models/embedder
}
# suite -> list of (task, hf_subsets or None, eval_splits)
SUITES = {
    "core": [
        ("XPQARetrieval", ["ara-eng", "eng-ara", "ara-ara"], ["test"]),
        ("MLQARetrieval", ["ara-eng", "eng-ara", "ara-ara"], ["test"]),
        ("BelebeleRetrieval", ["arb_Arab-eng_Latn", "eng_Latn-arb_Arab", "arb_Arab-arb_Arab"], ["test"]),
        ("MintakaRetrieval", ["ar"], ["test"]),
        ("MKQARetrieval", ["ar"], ["train"]),
        ("XQuADRetrieval", ["ar"], ["validation"]),
    ],
    "big": [("MIRACLRetrievalHardNegatives.v2", ["ar"], ["dev"])],
    # regression GUARDS requested by the coordinator (v1 loses to base e5-small on these Arabic-only tasks): reported on final candidates only, never used for selection
    "guard": [
        ("MIRACLRetrievalHardNegatives.v2", ["ar"], ["dev"]),
        ("NamaaMrTydiReranking", None, ["test"]),
        ("SemRel24STS", ["arb"], ["test"]),
    ],
    "oodev": [
        ("MLQARetrieval", ["ara-eng", "eng-ara", "ara-ara"], ["validation"])
    ],  # OOD *development* set (used to choose the merge weight; MLQA test stays held out)
    "align": [
        ("STS17", ["en-ar", "ar-ar"], ["test"]),
        ("Tatoeba", ["ara-eng"], ["test"]),
        ("NTREXBitextMining", ["arb_Arab-eng_Latn"], ["test"]),
        ("FloresBitextMining", ["arb_Arab-eng_Latn", "eng_Latn-arb_Arab"], ["devtest"]),
    ],
}


def load(tag: str, spec: str | None):
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
    from sentence_transformers import SentenceTransformer

    path, prompts = (
        (spec, None if tag.startswith(("bge", "m3")) else E5P) if spec else KNOWN[tag]
    )  # every local model here is an e5 derivative
    st = SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    st.max_seq_length = 512
    w = SentenceTransformerEncoderWrapper(st, model_prompts=prompts)
    w.mteb_model_meta = w.mteb_model_meta.model_copy(update={"name": f"local/{tag}", "revision": "v2"})
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tag or tag=path")
    ap.add_argument("--suite", nargs="+", default=["core"], choices=list(SUITES))
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    import mteb

    cache = mteb.ResultCache(cache_path=str(V2 / "mteb_cache"))
    for m in args.models:
        tag, _, spec = m.partition("=")
        model = load(tag, spec or None)
        (OUT / tag).mkdir(parents=True, exist_ok=True)
        for suite in args.suite:
            for name, subsets, splits in SUITES[suite]:
                out = OUT / tag / (f"{name}.json" if splits[0] != "validation" or name == "XQuADRetrieval" else f"{name}.validation.json")
                if out.exists():
                    print(f"[{tag}] {name}: cached", flush=True)
                    continue
                task = mteb.get_task(name, hf_subsets=subsets, eval_splits=splits)
                res = mteb.evaluate(
                    model,
                    task,
                    cache=cache,
                    encode_kwargs={"batch_size": args.bs},
                    overwrite_strategy="only-missing",
                    show_progress_bar=False,
                )
                tr = res.task_results[0]
                scores = {
                    sp: {s["hf_subset"]: {k: v for k, v in s.items() if isinstance(v, (int, float))} for s in lst}
                    for sp, lst in tr.scores.items()
                }
                out.write_text(
                    json.dumps(
                        {
                            "task": name,
                            "model": tag,
                            "main_score": tr.task.metadata.main_score if hasattr(tr, "task") else None,
                            "scores": scores,
                        },
                        indent=1,
                    )
                )
                brief = {
                    sp: {s: round(v.get("ndcg_at_10", v.get("main_score", 0)) * 100, 1) for s, v in d.items()} for sp, d in scores.items()
                }
                print(f"[{tag}] {name}: ndcg@10 {brief}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
