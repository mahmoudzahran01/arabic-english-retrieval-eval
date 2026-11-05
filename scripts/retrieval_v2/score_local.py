"""Score a LOCAL v2 checkpoint on MTEB(ara, v1) - variant of mteb-arabic/scripts/run_eval_local.py that handles the bge-m3 family correctly.

Why a variant: run_eval_local.py always applies the e5 prompts ("query: " / "passage: "). bge-m3 and the checkpoints fine-tuned from it were trained WITHOUT prefixes, so they
must be scored without them; and the ModelMeta must carry the real parameter count / dimension / training datasets of the family member.
Usage (inside the mteb-arabic container):  score_local.py LOCAL_DIR RESULT_NAME BATCH META_BASE_ID
  META_BASE_ID: BAAI/bge-m3 | intfloat/multilingual-e5-base | intfloat/multilingual-e5-small   (its registered ModelMeta is copied; prompts follow it)
Same conditions as the other models: fp16, batch <= 32, max_seq_length 512, results in /work/results_cache_v2 under RESULT_NAME.
training_datasets (honest, conservative): the base model's own list + {MrTidyRetrieval, NamaaMrTydiReranking, XQuADRetrieval, MIRACLRetrieval}
(TyDi-QA train = Mr.TyDi train questions, SQuAD train, MIRACL replay / teacher); KD students additionally inherit their teacher's (bge-m3) list.
"""
import json
import os
import sys
import time
import traceback

import mteb
import torch
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from sentence_transformers import SentenceTransformer

TRAINING = {"MrTidyRetrieval", "NamaaMrTydiReranking", "XQuADRetrieval", "MIRACLRetrieval"}

local_dir, result_name = sys.argv[1], sys.argv[2]
bs = min(int(sys.argv[3]) if len(sys.argv) > 3 else 32, 32)
base_id = sys.argv[4] if len(sys.argv) > 4 else "BAAI/bge-m3"
only = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else None
revision = "local-" + os.path.basename(local_dir.rstrip("/"))
prompts = None if "bge" in base_id else {"query": "query: ", "document": "passage: "}

base = mteb.get_model_meta(base_id)
inherited = base.training_datasets or {}
extra = set(TRAINING) | (set(mteb.get_model_meta("BAAI/bge-m3").training_datasets or {}) if "bge" not in base_id else set())
if isinstance(inherited, dict):
    td = {**inherited, **{k: [] for k in extra if k not in inherited}}
else:
    td = set(inherited) | extra
meta = base.model_copy(update=dict(name=result_name, revision=revision, release_date="2026-11-04", training_datasets=td, adapted_from=base_id, license="mit",
                                   public_training_code="https://github.com/mahmoudzahran01/arabic-english-retrieval-eval", public_training_data=None,
                                   reference="https://huggingface.co/mzahran1 (unreleased local checkpoint)"))

st = SentenceTransformer(local_dir, device="cuda", model_kwargs={"torch_dtype": torch.float16})
st.max_seq_length = 512
model = SentenceTransformerEncoderWrapper(st, model_prompts=prompts)
model.mteb_model_meta = meta

bench = mteb.get_benchmark("MTEB(ara, v1)")
tasks = [t for t in bench.tasks if only is None or t.metadata.name in only]
cache = mteb.ResultCache(cache_path="/work/results_cache_v2")
os.makedirs("/work/logs", exist_ok=True)
t0 = time.time()
for task in tasks:
    ts = time.time()
    status = "ok"
    try:
        mteb.evaluate(model, task, cache=cache, encode_kwargs={"batch_size": bs}, raise_error=True, show_progress_bar=False)
    except Exception as e:
        status = f"FAILED {type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    dt = time.time() - ts
    print(f"[score_local] {result_name} {task.metadata.name} {status} {dt:.0f}s", flush=True)
    with open("/work/logs/times_v2.jsonl", "a") as f:
        f.write(json.dumps({"model": result_name, "task": task.metadata.name, "status": status, "wall_s": round(dt, 1), "bs": bs, "fp16": True, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
print(f"[score_local] DONE {result_name} total {time.time() - t0:.0f}s", flush=True)
