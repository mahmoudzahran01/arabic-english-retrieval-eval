"""Compare the files written by check_compat.py in two environments (ST 6 on the original folders, ST 5 on the compatibility folders).

    python compare_compat.py DIR NAME [NAME ...]  [--json OUT.json] [--tol 1e-5]

For every NAME it expects DIR/st6__NAME.orig, st6__NAME.compat (optional), st5__NAME.orig (expected to fail) and st5__NAME.compat, and prints a markdown table.
The pass criteria: the ST 5 compat folder loads, max abs difference of the embeddings (ST 6 original vs ST 5 compat) < tol, and the same max_seq_length and pooling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path, label: str, name: str, kind: str):
    j = d / f"{label}__{name}.{kind}.json"
    info = json.loads(j.read_text()) if j.exists() else None
    z = d / f"{label}__{name}.{kind}.npz"
    return info, (np.load(z)["emb"] if z.exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()
    rows, out = [], {}
    for n in a.names:
        o6, e6 = load(a.dir, "st6", n, "orig")
        c6, f6 = load(a.dir, "st6", n, "compat")
        o5, _ = load(a.dir, "st5", n, "orig")
        c5, e5 = load(a.dir, "st5", n, "compat")
        diff = float(np.abs(e6 - e5).max()) if e6 is not None and e5 is not None else None
        diff_cross = float(np.abs(e6 - f6).max()) if e6 is not None and f6 is not None else None
        cos = float((e6 * e5).sum(axis=1).min()) if e6 is not None and e5 is not None else None
        same_len = o6 and c5 and o6.get("max_seq_length") == c5.get("max_seq_length")
        same_pool = o6 and c5 and o6.get("pooling") == c5.get("pooling")
        ok = bool(c5 and c5["loads"] and diff is not None and diff < a.tol and same_len and same_pool)
        out[n] = dict(
            st6_orig=o6,
            st6_compat=c6,
            st5_orig=o5,
            st5_compat=c5,
            max_abs_diff_st6orig_vs_st5compat=diff,
            max_abs_diff_st6orig_vs_st6compat=diff_cross,
            min_cosine=cos,
            passes=ok,
        )
        yn = lambda i: "-" if i is None else ("yes" if i["loads"] else "no")  # noqa: E731
        rows.append(
            f"| {n} | {yn(o6)} | {yn(o5)} | {yn(c5)} | {yn(c6)} | "
            f"{o6['max_seq_length'] if o6 else '-'} / {c5['max_seq_length'] if c5 and c5['loads'] else '-'} | "
            f"{o6['pooling'] if o6 else '-'} / {c5['pooling'] if c5 and c5['loads'] else '-'} | "
            f"{'-' if diff is None else f'{diff:.2e}'} | {'-' if cos is None else f'{cos:.8f}'} | {'PASS' if ok else 'FAIL'} |"
        )
    print(
        "| model | ST 6.2 loads original | ST 5.3 loads original | ST 5.3 loads compat | ST 6.2 loads compat | max_seq_length (6 orig / 5 compat) | pooling (6 orig / 5 compat) | max abs diff (6 orig vs 5 compat) | min cosine | check |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
