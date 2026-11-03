"""Stage 1: build the bilingual retrieval benchmark (monolingual queries + passages + split bookkeeping).

Sources (all load as plain parquet from the HF Hub, no loading scripts):
  TyDi QA GoldP (ar, en)   Apache-2.0   google-research-datasets/tydiqa  (secondary_task)
  ARCD                     MIT          hsseinmz/arcd
  SQuAD v1.1               CC BY-SA 4.0 rajpurkar/squad
  XQuAD (ar, en parallel)  CC BY-SA 4.0 google/xquad   (human translations of 1190 SQuAD-dev questions)

Split design (the point of this file is honest splits):
  * Passages live in exactly one *pool* (train / dev / test). Evaluating split S searches ONLY pool S, so a
    train passage can never be retrieved (or memorised) during test, and test passages never appear as
    training negatives.
  * Whole articles (titles) are assigned to a pool together (hash of title), so paragraphs of one article
    never straddle pools. Any article that appears in a test source is removed from train/dev everywhere.
  * Test = official TyDi dev (ar, en) + ARCD validation + XQuAD (SQuAD-dev articles, parallel ar/en).
    Test background = the rest of SQuAD-dev + hash-assigned articles from the train sources.
  * Duplicate passages / questions across sources are resolved by priority test > dev > train:
    a lower-priority item that duplicates a higher-priority one is dropped.
  * Cross-lingual (ar query -> en passage and vice versa) test queries come from XQuAD's parallel data.
    Cross-lingual TRAIN/DEV queries are LLM-translated copies of real questions (translate.py + finalize_data.py).

Run: python -m arretrieval.retrieval.build_data
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from .paths import DATA, SEED
from .textnorm import dedupe_key

# fraction of (hash-ordered) article titles per pool: train / dev / dev_bg / test_bg (rest unused)
TITLE_SPLITS = {
    "tydi_ar_train": (0.50, 0.08, 0.12, 0.30),
    "tydi_en_train": (0.55, 0.10, 0.10, 0.25),
    "arcd_train": (0.75, 0.25, 0.0, 0.0),
    "squad_train": (0.20, 0.04, 0.04, 0.10),
}
MAX_Q_PER_CTX = {"squad_train": 2}  # SQuAD has ~5 questions per paragraph; cap to keep the mix balanced
LICENSES = {
    "tydi": "Apache-2.0 (google-research-datasets/tydiqa, secondary_task = GoldP)",
    "arcd": "MIT (hsseinmz/arcd)",
    "squad": "CC BY-SA 4.0 (rajpurkar/squad v1.1)",
    "xquad": "CC BY-SA 4.0 (google/xquad)",
}


# Dataset revisions the published numbers were built from (the repositories have not changed since early 2024).
REVISIONS = {
    "google-research-datasets/tydiqa": "da78f23f9119363459acbaf46bf89426ff26c259",
    "hsseinmz/arcd": "cc6906b6eda547e4ffc63b8d88ccca7e0515187a",
    "rajpurkar/squad": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "google/xquad": "51adfef1c1287aab1d2d91b5bead9bcfb9c68583",
}


def pq(repo: str, path: str) -> pd.DataFrame:
    """Download one parquet file of a public Hugging Face dataset (pinned revision, cached locally) and load it."""
    root = snapshot_download(repo, repo_type="dataset", revision=REVISIONS[repo], allow_patterns=[path])
    return pd.read_parquet(Path(root) / path)


def u01(*parts: str) -> float:
    h = hashlib.sha1("|".join((str(SEED), *parts)).encode()).hexdigest()
    return int(h[:8], 16) / 2**32


def pid_of(lang: str, text: str) -> str:
    return f"{lang}-{hashlib.sha1(dedupe_key(text).encode()).hexdigest()[:12]}"


class Registry:
    """Passages + queries with priority-aware de-duplication."""

    def __init__(self) -> None:
        self.passages: dict[str, dict] = {}
        self.queries: list[dict] = []
        self.qkeys: dict[tuple[str, str], str] = {}  # (lang, question key) -> split that owns it
        self.dropped: Counter = Counter()

    def add_passage(self, lang: str, text: str, title: str, source: str, pool: str) -> str | None:
        pid = pid_of(lang, text)
        cur = self.passages.get(pid)
        if cur is None:
            self.passages[pid] = {"pid": pid, "lang": lang, "text": text.strip(), "title": title, "source": source, "pool": pool}
            return pid
        return pid if cur["pool"] == pool else None  # duplicate living in another pool -> caller drops its item

    def add_query(
        self,
        *,
        qid: str,
        text: str,
        qlang: str,
        pos_pid: str,
        plang: str,
        split: str,
        slice_: str,
        source: str,
        title: str,
        src_qid: str = "",
    ) -> bool:
        key = (qlang, dedupe_key(text))
        owner = self.qkeys.get(key)
        if owner is not None and owner != split:
            self.dropped[f"dup_question_{slice_}"] += 1
            return False
        self.qkeys[key] = split
        self.queries.append(
            {
                "qid": qid,
                "text": text.strip(),
                "qlang": qlang,
                "pos_pid": pos_pid,
                "plang": plang,
                "split": split,
                "slice": slice_,
                "source": source,
                "title": title,
                "src_qid": src_qid,
            }
        )
        return True


def load_sources() -> dict[str, pd.DataFrame]:
    src: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation"):
        df = pq("google-research-datasets/tydiqa", f"secondary_task/{split}-00000-of-00001.parquet")
        df["lang"] = df["id"].str.split("-").str[0]
        for lang, short in (("arabic", "ar"), ("english", "en")):
            src[f"tydi_{short}_{'train' if split == 'train' else 'val'}"] = df[df.lang == lang].reset_index(drop=True)
    src["arcd_train"] = pq("hsseinmz/arcd", "plain_text/train-00000-of-00001.parquet")
    src["arcd_val"] = pq("hsseinmz/arcd", "plain_text/validation-00000-of-00001.parquet")
    src["squad_train"] = pq("rajpurkar/squad", "plain_text/train-00000-of-00001.parquet")
    src["squad_val"] = pq("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet")
    src["xquad_ar"] = pq("google/xquad", "xquad.ar/validation-00000-of-00001.parquet")
    src["xquad_en"] = pq("google/xquad", "xquad.en/validation-00000-of-00001.parquet")
    return src


def build_test(reg: Registry, src: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    """Test pool; returns the set of test article-title keys per language (used to purge train/dev)."""
    test_titles: dict[str, set[str]] = {"ar": set(), "en": set()}
    for short, key, slice_ in (("ar", "tydi_ar_val", "tydi_ar"), ("en", "tydi_en_val", "tydi_en"), ("ar", "arcd_val", "arcd_ar")):
        for r in src[key].itertuples():
            test_titles[short].add(dedupe_key(r.title))
            pid = reg.add_passage(short, r.context, r.title, key, "test")
            assert pid is not None
            reg.add_query(
                qid=f"{slice_}-{r.id}",
                text=r.question,
                qlang=short,
                pos_pid=pid,
                plang=short,
                split="test",
                slice_=slice_,
                source=key,
                title=r.title,
            )
    # XQuAD: parallel ar/en, ids identical across languages (and identical to SQuAD-dev ids)
    ar = src["xquad_ar"].set_index("id")
    en = src["xquad_en"].set_index("id")
    assert list(ar.index) == list(en.index), "XQuAD ar/en rows are not aligned"
    squad_val_title = {r.context: r.title for r in src["squad_val"].itertuples()}
    for qid in ar.index:
        title = squad_val_title.get(en.loc[qid, "context"], "")
        pid_ar = reg.add_passage("ar", ar.loc[qid, "context"], title, "xquad_ar", "test")
        pid_en = reg.add_passage("en", en.loc[qid, "context"], title, "xquad_en", "test")
        qa, qe = ar.loc[qid, "question"], en.loc[qid, "question"]
        for q_lang, q_text, p_lang, p_pid in (
            ("ar", qa, "ar", pid_ar),
            ("en", qe, "en", pid_en),
            ("ar", qa, "en", pid_en),
            ("en", qe, "ar", pid_ar),
        ):
            sl = f"xquad_{q_lang}2{p_lang}"
            # same question text appears in two directions: allowed (same split), identical text is intended here
            reg.add_query(
                qid=f"{sl}-{qid}",
                text=q_text,
                qlang=q_lang,
                pos_pid=p_pid,
                plang=p_lang,
                split="test",
                slice_=sl,
                source="xquad",
                title=title,
                src_qid=qid,
            )
    # rest of SQuAD-dev = English test background; its 48 articles are all "test titles"
    for r in src["squad_val"].itertuples():
        test_titles["en"].add(dedupe_key(r.title))
        reg.add_passage("en", r.context, r.title, "squad_val", "test")
    return test_titles


def title_pool(source: str, title: str) -> str:
    u = u01(source, dedupe_key(title))
    tr, dv, dvb, tsb = TITLE_SPLITS[source]
    if u < tr:
        return "train"
    if u < tr + dv:
        return "dev"
    if u < tr + dv + dvb:
        return "dev_bg"
    if u < tr + dv + dvb + tsb:
        return "test_bg"
    return "unused"


def build_train_sources(reg: Registry, src: dict[str, pd.DataFrame], test_titles: dict[str, set[str]]) -> None:
    rng = random.Random(SEED)
    plan = (
        ("tydi_ar_train", "ar", "tydi_ar"),
        ("tydi_en_train", "en", "tydi_en"),
        ("arcd_train", "ar", "arcd_ar"),
        ("squad_train", "en", "squad_en"),
    )
    # priority order: eval pools first, train last, so duplicates always lose against dev/test
    for want_pools in (("test_bg", "dev", "dev_bg"), ("train",)):
        for source, lang, slice_ in plan:
            df = src[source]
            per_ctx: dict[str, int] = defaultdict(int)
            rows = list(df.itertuples())
            rng.shuffle(rows)
            for r in rows:
                if dedupe_key(r.title) in test_titles[lang]:
                    reg.dropped[f"title_in_test_{source}"] += 1
                    continue
                pool = title_pool(source, r.title)
                if pool not in want_pools:
                    continue
                eff_pool = {"dev_bg": "dev", "test_bg": "test"}.get(pool, pool)
                pid = reg.add_passage(lang, r.context, r.title, source, eff_pool)
                if pid is None:
                    reg.dropped[f"passage_in_other_pool_{source}"] += 1
                    continue
                if pool in ("test_bg", "dev_bg"):
                    continue  # background: passage only
                if per_ctx[pid] >= MAX_Q_PER_CTX.get(source, 10**9):
                    continue
                if reg.add_query(
                    qid=f"{slice_}-{eff_pool}-{r.id}",
                    text=r.question,
                    qlang=lang,
                    pos_pid=pid,
                    plang=lang,
                    split=eff_pool,
                    slice_=f"{eff_pool}_{slice_}" if eff_pool == "dev" else f"train_{slice_}",
                    source=source,
                    title=r.title,
                ):
                    per_ctx[pid] += 1


def write_xling_candidates(queries: pd.DataFrame) -> pd.DataFrame:
    """Pick real mono-lingual questions to machine-translate into the other language."""
    rng = random.Random(SEED + 1)
    picks = []

    def take(df: pd.DataFrame, n: int, direction: str, split: str) -> None:
        ids = list(df.qid)
        rng.shuffle(ids)
        for q in ids[:n]:
            picks.append({"src_qid": q, "direction": direction, "split": split})

    tr, dv = queries[queries.split == "train"], queries[queries.split == "dev"]
    en_tr, ar_tr = tr[tr.qlang == "en"], tr[tr.qlang == "ar"]
    take(en_tr[en_tr.slice == "train_tydi_en"], 1200, "en2ar", "train")  # ar query -> en passage
    take(en_tr[en_tr.slice == "train_squad_en"], 1800, "en2ar", "train")
    take(ar_tr, 1500, "ar2en", "train")  # en query -> ar passage
    take(dv[dv.qlang == "en"], 300, "en2ar", "dev")
    take(dv[dv.qlang == "ar"], 200, "ar2en", "dev")
    cand = (
        pd.DataFrame(picks)
        .merge(queries[["qid", "text", "qlang", "pos_pid", "plang", "slice"]], left_on="src_qid", right_on="qid")
        .drop(columns="qid")
    )
    cand.to_json(DATA / "xling_candidates.jsonl", orient="records", lines=True, force_ascii=False)
    return cand


def main() -> None:
    src = load_sources()
    reg = Registry()
    test_titles = build_test(reg, src)
    build_train_sources(reg, src, test_titles)

    passages = pd.DataFrame(list(reg.passages.values()))
    queries = pd.DataFrame(reg.queries)
    # drop queries whose passage vanished / pool-mismatch (defensive; the registry already guarantees this)
    pool_of = dict(zip(passages.pid, passages.pool))
    bad = queries[queries.apply(lambda q: pool_of.get(q.pos_pid) != q.split, axis=1)]
    assert len(bad) == 0, f"{len(bad)} queries whose gold passage is outside their split's pool"
    gold = set(queries.pos_pid)
    passages["kind"] = ["gold" if p in gold else "background" for p in passages.pid]
    passages["order"] = [u01("order", p) for p in passages.pid]  # deterministic shuffle (BM25 tie-breaks etc.)
    passages = passages.sort_values("order").drop(columns="order").reset_index(drop=True)
    passages.to_parquet(DATA / "passages.parquet", index=False)
    queries.to_parquet(DATA / "queries_mono.parquet", index=False)
    cand = write_xling_candidates(queries)

    manifest = {
        "seed": SEED,
        "title_splits(train,dev,dev_bg,test_bg)": TITLE_SPLITS,
        "max_q_per_ctx": MAX_Q_PER_CTX,
        "licenses": LICENSES,
        "dropped": dict(reg.dropped),
        "passages_by_pool_lang": {f"{k[0]}/{k[1]}": int(v) for k, v in passages.groupby(["pool", "lang"]).size().items()},
        "queries_by_slice": {k: int(v) for k, v in queries.groupby(["split", "slice"]).size().items()}
        if False
        else {f"{a}/{b}": int(v) for (a, b), v in queries.groupby(["split", "slice"]).size().items()},
        "xling_candidates": {f"{a}/{b}": int(v) for (a, b), v in cand.groupby(["split", "direction"]).size().items()},
    }
    (DATA / "manifest_mono.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
