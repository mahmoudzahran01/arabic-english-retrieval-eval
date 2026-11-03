"""Download XQuAD (ar + en) from its public GitHub repository into runs/rageval/data_raw and verify the checksums.

XQuAD: Artetxe, Ruder, Yogatama (2020), https://github.com/google-deepmind/xquad, licence CC BY-SA 4.0.
Run: python -m arretrieval.rageval.download_data
"""

from __future__ import annotations

import hashlib
import sys

import httpx

from .paths import RAW

BASE = "https://raw.githubusercontent.com/google-deepmind/xquad/master"
SHA256 = {
    "xquad.ar.json": "abdabd7afed5c635d99cca0f3f0d0c9d9ed0bc77451e963c2e4e0638c29e486d",
    "xquad.en.json": "e4c57d1c9143aaa1c5d265ba5987a65f4e69528d2a98f29d6e75019b10344f29",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, want in SHA256.items():
        dest = RAW / name
        if not dest.exists():
            r = httpx.get(f"{BASE}/{name}", follow_redirects=True, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"{dest}: checksum mismatch (got {got}, expected {want}); the upstream file changed, results will differ")
        print(f"ok  {dest}  ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
