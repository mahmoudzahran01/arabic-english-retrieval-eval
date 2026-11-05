"""Weight-space merging of sentence-transformers models that share an architecture (no training).

  --base B --ft F --alpha a     WiSE-FT style interpolation  W = (1-a) * W_base + a * W_ft      (Wortsman et al. 2022)
  --soup F1 F2 F3               uniform average of fine-tuned runs from the same initialisation   (model soup, Wortsman et al. 2022)
  (both can be combined: soup first, then interpolation towards --base)

Why: fine-tuning on a narrow factoid-QA set lowers out-of-domain Arabic scores (v1: MIRACL-ar -6.6 nDCG vs the base model, which saw MIRACL
train during its own supervised stage); interpolating back towards the base weights is the cheapest known remedy, and averaging seeds
is the cheapest known variance reducer.   Run (GPU container, loads on CPU): python -m arretrieval.retrieval.v2.merge_v2 --base intfloat/multilingual-e5-small --ft DIR --alpha 0.6 --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--ft", nargs="*", default=[])
    ap.add_argument("--soup", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    members = args.soup or args.ft
    assert members, "give --ft DIR or --soup DIR DIR ..."
    models = [SentenceTransformer(m, device="cpu") for m in members]
    sds = [m.state_dict() for m in models]
    avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    if args.base and args.alpha < 1.0:
        sd_b = SentenceTransformer(args.base, device="cpu").state_dict()
        avg = {k: ((1 - args.alpha) * sd_b[k].float() + args.alpha * v) if v.is_floating_point() else v for k, v in avg.items()}
    out = models[0]
    out.load_state_dict({k: v.to(sds[0][k].dtype) for k, v in avg.items()})
    out.save(args.out)
    Path(args.out, "merge.json").write_text(json.dumps({"base": args.base, "members": members, "alpha": args.alpha}))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
