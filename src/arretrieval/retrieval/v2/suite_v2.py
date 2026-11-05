"""Validation-suite components 4-6 (non-retrieval, non-MTEB-split), PRE-REGISTERED in runs/retrieval/v2/PROGRESS.md (2026-11-04 ~19:05).

  xnli : average precision of the cosine similarity on 3,000 XNLI-ar TRAIN pairs (seed 13; entailment=1, contradiction=0, neutral dropped, as the mteb task does).
         (mteb's XNLI scores BOTH the test and the validation split, so neither is used here.)
  cls  : mean of two 8-shot-per-label logistic-regression probes (mteb protocol: 10 experiments, samples_per_label=8, max_iter=100) on the ArSarcasm TRAIN split
         (iabufarha/ar_sarcasm, MIT): sarcasm (2 classes) and sentiment (3 classes).  Fit rows: TRAIN minus a seeded 2,000-row hold-out; scored on the hold-out.
  sts  : Spearman of the cosine similarity on the Arabic STSb VALIDATION split (Omartificial-Intelligence-Space/Arabic-stsb, apache-2.0; machine-translated STS-B, 1,500 pairs).

Run (GPU, behind the gate):  python -m arretrieval.retrieval.v2.suite_v2 --models NAME=PATH:e5|none ...      results: runs/retrieval/v2/results/suite/NAME.json (cached per model)
Nothing here is used for training; no MTEB split is read.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from arretrieval.retrieval import evalkit as E

from .common_v2 import V2DATA, V2RES

SUITE_DATA = V2DATA / "suite"
OUT = V2RES / "suite"
SEED, N_XNLI, N_HOLD, N_EXP, SHOTS = 13, 3000, 2000, 10, 8


def load_xnli() -> tuple[list[str], list[str], np.ndarray]:
    d = pd.read_parquet(SUITE_DATA / "xnli_ar_train.parquet")
    d = d[d["label"].isin([0, 2])].sample(n=N_XNLI, random_state=SEED)
    return d["premise"].tolist(), d["hypothesis"].tolist(), (d["label"] == 0).astype(int).to_numpy()


def load_sarcasm() -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    d = pd.read_parquet(SUITE_DATA / "ar_sarcasm_train.parquet").dropna(subset=["tweet"]).reset_index(drop=True)
    d = d.iloc[np.random.RandomState(SEED).permutation(len(d))].reset_index(drop=True)
    labels = {"sarcasm": pd.factorize(d["sarcasm"].astype(str))[0], "sentiment": pd.factorize(d["sentiment"].astype(str))[0]}
    hold = np.zeros(len(d), bool)
    hold[:N_HOLD] = True
    return d["tweet"].tolist(), labels, hold


def load_sts() -> tuple[list[str], list[str], np.ndarray]:
    d = pd.read_csv(SUITE_DATA / "stsb_ar_validation.csv").dropna()
    return d["sentence1"].astype(str).tolist(), d["sentence2"].astype(str).tolist(), d["score"].to_numpy(float)


def probe(emb: np.ndarray, y: np.ndarray, hold: np.ndarray) -> float:
    """mteb-style few-shot probe: SHOTS examples per label, logistic regression, accuracy on the hold-out; mean over N_EXP draws."""
    from sklearn.linear_model import LogisticRegression

    fit_idx = np.where(~hold)[0]
    hold_idx = np.where(hold)[0]
    accs = []
    for i in range(N_EXP):
        rng = np.random.RandomState(SEED + i)
        pick = np.concatenate([rng.choice(fit_idx[y[fit_idx] == c], SHOTS, replace=False) for c in np.unique(y)])
        clf = LogisticRegression(max_iter=100).fit(emb[pick], y[pick])
        accs.append(float((clf.predict(emb[hold_idx]) == y[hold_idx]).mean()))
    return float(np.mean(accs))


def run_model(m: E.Dense) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    a, b, lab = load_xnli()
    ea, eb = m.encode(a, "q"), m.encode(b, "q")
    xnli = float(average_precision_score(lab, (ea * eb).sum(1)))
    s1, s2, gold = load_sts()
    cos = (m.encode(s1, "q") * m.encode(s2, "q")).sum(1)
    sts = float(spearmanr(cos, gold).statistic)
    tw, labels, hold = load_sarcasm()
    et = m.encode(tw, "q")
    sarc, sent = probe(et, labels["sarcasm"], hold), probe(et, labels["sentiment"], hold)
    return {
        "xnli_ap": xnli,
        "sarcasm_acc": sarc,
        "sentiment_acc": sent,
        "cls": (sarc + sent) / 2,
        "sts_spearman": sts,
        "n": {"xnli": len(lab), "sts": len(gold), "tweets": len(tw), "holdout": int(hold.sum())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in a.models:
        name, _, rest = spec.partition("=")
        path, _, pre = rest.rpartition(":")
        if (OUT / f"{name}.json").exists():
            print("cached", name, flush=True)
            continue
        qp = "query: " if pre == "e5" else ""
        m = E.Dense(path, name, qp, qp, max_len=256, bs=64 if pre == "e5" else 32)
        r = run_model(m)
        (OUT / f"{name}.json").write_text(json.dumps(r, indent=1))
        print(
            f"[{name}] XNLI-ar(train) AP {r['xnli_ap'] * 100:.2f} | sarcasm {r['sarcasm_acc'] * 100:.1f} sentiment {r['sentiment_acc'] * 100:.1f} | STS-ar(dev) rho {r['sts_spearman'] * 100:.1f}",
            flush=True,
        )
        del m


if __name__ == "__main__":
    main()
