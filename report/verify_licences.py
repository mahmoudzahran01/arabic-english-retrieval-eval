"""Look up the licence of every dataset and model used by the v2 study on the Hugging Face Hub and write the data card.

    python report/verify_licences.py            # needs network; writes results/retrieval_v2/data/licences.json

`make_tables.py` renders that file (report/tables/tab_licences.tex); the report never states a licence that is not in it. Sources without a licence tag
on their Hub card are recorded as such (`hub_licence: null`) rather than guessed. The `role` field says how the source entered the study.
"""

from __future__ import annotations

import datetime
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "retrieval_v2" / "data" / "licences.json"

# role: train = in the training data of the final recipe; train-exp = training data of an ablation only; eval = evaluation / selection only;
#       translator = produced the machine-translated training queries; base = initial weights
SOURCES = [
    {
        "name": "TyDi QA GoldP (Arabic, English)",
        "kind": "datasets",
        "id": "google-research-datasets/tydiqa",
        "role": "train",
        "note": "also the source of the TyDi test slices",
    },
    {
        "name": "SQuAD v1.1 train",
        "kind": "datasets",
        "id": "rajpurkar/squad",
        "role": "train",
        "note": "share-alike: see the weights-licence question",
    },
    {
        "name": "ARCD",
        "kind": "datasets",
        "id": "hsseinmz/arcd",
        "role": "train",
        "note": "train and validation articles of the Arabic reading-comprehension set",
    },
    {"name": "XQuAD", "kind": "datasets", "id": "google/xquad", "role": "eval", "note": "test pool only; never trained on"},
    {
        "name": "MIRACL Arabic (queries, qrels, corpus)",
        "kind": "datasets",
        "id": "miracl/miracl",
        "role": "train-exp",
        "note": "replay ablation only; corpus: miracl/miracl-corpus",
    },
    {
        "name": "Gemma-4-26B-A4B (query translator)",
        "kind": "models",
        "id": "google/gemma-4-26B-A4B-it",
        "role": "translator",
        "note": "machine-translated questions only; the base repo google/gemma-4-26B-A4B carries the same licence tag",
    },
    {"name": "bge-m3 (base weights of v2-m3, teacher baseline)", "kind": "models", "id": "BAAI/bge-m3", "role": "base", "note": ""},
    {"name": "multilingual-E5-base (base of v2-base)", "kind": "models", "id": "intfloat/multilingual-e5-base", "role": "base", "note": ""},
    {
        "name": "multilingual-E5-small (base of v2-small)",
        "kind": "models",
        "id": "intfloat/multilingual-e5-small",
        "role": "base",
        "note": "",
    },
    {
        "name": "ArSarcasm (selection suite)",
        "kind": "datasets",
        "id": "iabufarha/ar_sarcasm",
        "role": "eval",
        "note": "train split used only for a validation probe",
    },
    {
        "name": "XNLI (selection suite)",
        "kind": "datasets",
        "id": "facebook/xnli",
        "role": "eval",
        "note": "Arabic train subsample used only for validation",
    },
    {
        "name": "Arabic STS-B (selection suite)",
        "kind": "datasets",
        "id": "Omartificial-Intelligence-Space/Arabic-stsb",
        "role": "eval",
        "note": "validation split used only for selection",
    },
    {
        "name": "mMARCO Arabic (selection proxy)",
        "kind": "datasets",
        "id": "unicamp-dl/mmarco",
        "role": "eval",
        "note": "1,000-query dev proxy, selection only",
    },
]


def hub_licence(kind: str, ident: str) -> dict:
    d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/{kind}/{ident}", timeout=30))
    tags = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")]
    card = (d.get("cardData") or {}).get("license")
    lic = tags[0] if tags else (card[0] if isinstance(card, list) and card else card)
    return {"hub_licence": lic, "hub_revision": d.get("sha", "")[:8], "hub_last_modified": d.get("lastModified", "")[:10]}


def main() -> None:
    rows = []
    for s in SOURCES:
        try:
            info = hub_licence(s["kind"], s["id"])
        except Exception as e:  # keep the row, say that the lookup failed
            info = {"hub_licence": None, "lookup_error": f"{type(e).__name__}: {e}"}
        rows.append({**s, **info})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"checked": datetime.date.today().isoformat(), "source": "https://huggingface.co/api/{datasets,models}/<id>", "sources": rows},
            indent=1,
        )
    )
    for r in rows:
        print(f"{r['id']:55s} {r['role']:10s} {r.get('hub_licence')}")


if __name__ == "__main__":
    main()
