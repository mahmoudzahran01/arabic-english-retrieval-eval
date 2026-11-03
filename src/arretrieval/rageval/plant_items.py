"""Stage 2: turn verified-correct bases into the labelled judge-calibration set by PLANTING known errors.

Every label is true by construction (see LABELS below). Run:  python -m arretrieval.rageval.plant_items
Outputs runs/rageval/data/meta_items.jsonl (+ meta_card.json with counts and generation-time rejections).

Error types (all edits are deterministic except the two that need Gemma: invented sentence, translation; those
are validated programmatically and rejected if the validation fails):
  clean               real Gemma answer, verified correct against the XQuAD reference
  clean_numeral       same answer with the number re-written in the other digit script / separator style (STILL correct)
  wrong_number        reference number replaced (near=+-small, scale=x10, far=random)
  wrong_number_arindic the SAME wrong value as an existing wrong_number item, written in Arabic-Indic digits (paired twin:
                      isolates the digit-script effect on detection)
  wrong_entity        reference entity replaced by another entity of the same kind (never present in the context)
  unsupported_claim   one invented sentence with a fabricated number appended (number absent from every context)
  omitted_fact        compound 2-part question, answer contains only the first part
  wrong_language      correct answer translated to the other language
  wrong_citation      correct answer, citation switched to a context passage that does NOT contain the answer
  correct_abstention  gold passage removed from the context; answer says "not in the documents"
  wrong_abstention    gold passage present; answer says "not in the documents"
  halluc_unanswerable gold passage removed; answer confidently states the fact (citing a passage that lacks it)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict

from .llm import get_llm
from .paths import DATA
from .textnorm import cited_ids, contains_ref, lang_of, norm, numbers, to_arabic_indic, to_ascii_digits

SEED = 11
TARGET = {
    "clean": 24,
    "clean_numeral": 10,
    "wrong_number": 18,
    "wrong_entity": 18,
    "unsupported_claim": 18,
    "omitted_fact": 14,
    "wrong_language": 14,
    "wrong_citation": 16,
    "correct_abstention": 14,
    "wrong_abstention": 14,
    "halluc_unanswerable": 14,
    "wrong_number_arindic": 12,
}

# label semantics: faithful (claims supported by context), correct (matches reference), citation_ok, language_ok,
# abstention_ok.  None = not defined for that item.  acceptable = AND of the defined ones.
LABELS = {
    "clean": dict(faithful=1, correct=1, citation_ok=1, language_ok=1, abstention_ok=None),
    "clean_numeral": dict(faithful=1, correct=1, citation_ok=1, language_ok=1, abstention_ok=None),
    "wrong_number": dict(faithful=0, correct=0, citation_ok=1, language_ok=1, abstention_ok=None),
    "wrong_number_arindic": dict(faithful=0, correct=0, citation_ok=1, language_ok=1, abstention_ok=None),
    "wrong_entity": dict(faithful=0, correct=0, citation_ok=1, language_ok=1, abstention_ok=None),
    "unsupported_claim": dict(faithful=0, correct=1, citation_ok=1, language_ok=1, abstention_ok=None),
    "omitted_fact": dict(faithful=1, correct=0, citation_ok=1, language_ok=1, abstention_ok=None),
    "wrong_language": dict(faithful=1, correct=1, citation_ok=1, language_ok=0, abstention_ok=None),
    "wrong_citation": dict(faithful=1, correct=1, citation_ok=0, language_ok=1, abstention_ok=None),
    "correct_abstention": dict(faithful=1, correct=1, citation_ok=1, language_ok=1, abstention_ok=1),
    "wrong_abstention": dict(faithful=None, correct=0, citation_ok=1, language_ok=1, abstention_ok=0),
    "halluc_unanswerable": dict(faithful=0, correct=0, citation_ok=0, language_ok=1, abstention_ok=0),
}

ABST = {
    "en": [
        "Not in the documents.",
        "The documents do not contain the answer to this question.",
        "This information is not in the documents.",
    ],
    "ar": ["غير موجود في المستندات.", "لا تحتوي المستندات على إجابة لهذا السؤال.", "هذه المعلومة غير موجودة في المستندات."],
}
NUM_TOKEN = re.compile(r"[0-9٠-٩۰-۹][0-9٠-٩۰-۹,.٬٫،]*")


def acceptable(lbl: dict) -> int:
    return int(all(v == 1 for v in lbl.values() if v is not None))


def split_of(para: int) -> str:
    return "dev" if int(hashlib.sha1(f"split{para}".encode()).hexdigest(), 16) % 10 < 4 else "test"


# ----------------------------------------------------------------------------------------------- number edits
def find_num_token(answer: str, target: str):
    for m in NUM_TOKEN.finditer(answer):
        tok = m.group(0).rstrip(".,،٫")
        digits = re.sub(r"[,٬،]", "", to_ascii_digits(tok))
        if digits == target:
            return m.start(), m.start() + len(tok), tok
    return None


def render_like(value: int, template: str) -> str:
    s = f"{value:,}" if "," in template or "٬" in template else str(value)
    return to_arabic_indic(s) if re.search("[٠-٩]", template) else s


def wrong_value(n: int, rng: random.Random) -> tuple[int, str]:
    sub = rng.choices(["near", "scale", "far"], [0.4, 0.2, 0.4])[0]
    if sub == "near":
        if 1000 <= n <= 2100:
            d = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])
        elif n < 100:
            d = rng.choice([-2, -1, 1, 2])
        else:
            d = rng.choice([-1, 1]) * max(1, round(n * rng.choice([0.01, 0.02, 0.05])))
        return max(0, n + d), sub
    if sub == "scale":
        return (n * 10 if n < 10**6 else n // 10), sub
    lo, hi = max(1, int(n * 0.3)), max(3, int(n * 3))
    for _ in range(50):
        v = rng.randint(lo, hi)
        if abs(v - n) > 0.3 * max(n, 1) and v != n:
            return v, sub
    return n * 2 + 7, sub


def plant_wrong_number(b: dict, rng):
    nums = numbers(b["reference"])
    if len(nums) != 1 or not nums[0].isdigit():
        return None
    hit = find_num_token(b["answer"], nums[0])
    if not hit:
        return None
    s, e, tok = hit
    new, sub = wrong_value(int(nums[0]), rng)
    if new == int(nums[0]):
        return None
    ans = b["answer"][:s] + render_like(new, tok) + b["answer"][e:]
    return ans, {"subtype": sub, "orig": nums[0], "new": str(new), "script": "ar" if re.search("[٠-٩]", tok) else "ascii"}


def plant_numeral_variant(b: dict, rng):
    nums = numbers(b["reference"])
    if len(nums) != 1 or not nums[0].isdigit():
        return None
    hit = find_num_token(b["answer"], nums[0])
    if not hit:
        return None
    s, e, tok = hit
    v = int(nums[0])
    opts = []
    if not re.search("[٠-٩]", tok):
        opts.append(("ascii->arabic_indic", to_arabic_indic(tok)))
    else:
        opts.append(("arabic_indic->ascii", to_ascii_digits(tok)))
    if v >= 1000 and "," not in tok and "٬" not in tok:
        opts.append(("add_separator", render_like(v, tok.replace(tok, f"{v:,}"))))
    sub, new = rng.choice(opts)
    if new == tok:
        return None
    return b["answer"][:s] + new + b["answer"][e:], {"subtype": sub, "orig": tok, "new": new}


# ----------------------------------------------------------------------------------------------- entity swap
def plant_wrong_entity(b: dict, distractors: tuple[list[str], list[str]], rng):
    """Swap the reference entity for another entity. Preference order, to keep the swap *plausible* (a judge should not be
    able to catch it just because the new entity is absurd): same Wikipedia article > same word count > anything."""
    ref = b["reference"].strip()
    m = re.search(re.escape(ref), b["answer"], re.I)
    if not m or b["kind"] != "ent":
        return None
    ctx_norm = norm(" ".join(c["text"] for c in b["contexts"]))
    same, other = distractors
    ok = lambda d: (
        norm(d) != norm(ref)
        and norm(d) not in ctx_norm
        and not re.search(r"\d", d)
        and re.fullmatch(r"[\w\s'.\-]+", d, re.U)
        and abs(len(d.split()) - len(ref.split())) <= 1
    )
    pool = [d for d in same if ok(d)] or [d for d in other if ok(d) and len(d.split()) == len(ref.split())] or [d for d in other if ok(d)]
    if not pool:
        return None
    new = rng.choice(pool)
    return b["answer"][: m.start()] + new + b["answer"][m.end() :], {"orig": ref, "new": new}


# ----------------------------------------------------------------------------------------------- Gemma-assisted
_HALL_P = """Write ONE sentence in {lang} that adds a specific, plausible-sounding extra detail about the topic of this question.
Requirements: it MUST contain a specific number (a year, a count or a percentage) and a specific named person, place or organisation; it must NOT repeat anything stated in the passage below; keep it under 30 words.
Question: {q}
Passage: {p}
Output only the sentence."""

_TRANS_P = """Translate the following answer into {lang}. Keep every number and every bracketed document id such as [P012] exactly as written. Output only the translation.
Answer: {a}"""


def plant_unsupported(b: dict, llm):
    lang = "Arabic" if b["q_lang"] == "ar" else "English"
    gold = next(c for c in b["contexts"] if c["id"] == b["gold_id"])
    for attempt in range(3):
        s = (
            llm.chat(
                _HALL_P.format(lang=lang, q=b["question"], p=gold["text"]),
                max_tokens=100,
                temperature=0.7 if attempt else 0.2,
                seed=attempt,
            )
            .strip()
            .split("\n")[0]
            .strip()
        )
        if not s or len(s.split()) > 45 or lang_of(s) != b["q_lang"] or re.search(r"P\d{3}", s):
            continue
        ctx_nums = set(numbers(" ".join(c["text"] for c in b["contexts"]))) | set(numbers(b["question"]))
        s_nums = set(numbers(s))
        if not s_nums or (s_nums & ctx_nums):
            continue  # must contain numbers, none of them in the context
        return b["answer"].rstrip() + " " + s + f" [{b['gold_id']}]", {"sentence": s}
    return None


def plant_language(b: dict, llm):
    tgt = "en" if b["q_lang"] == "ar" else "ar"
    t = llm.chat(_TRANS_P.format(lang="English" if tgt == "en" else "Arabic", a=b["answer"]), max_tokens=300).strip()
    if lang_of(re.sub(r"\[P\d{3}\]", "", t)) != tgt:
        return None
    if set(cited_ids(t)) != set(cited_ids(b["answer"])):
        return None
    if not set(numbers(b["answer"])) <= set(numbers(t)):
        return None
    return t, {"target_lang": tgt}


# ----------------------------------------------------------------------------------------------- assembly
def main():
    rng = random.Random(SEED)
    llm = get_llm("gemma")
    bases = [json.loads(l) for l in open(DATA / "bases.jsonl")]
    ok = [b for b in bases if b["ok"]]
    allqa = [json.loads(l) for l in open(DATA / "all_qas.jsonl")]
    sent_of = {r["xquad_id"]: (r["sent_en"], r["sent_ar"]) for r in allqa}
    for b in bases:  # reference *sentence* (see ragas_eval: bare spans break FactualCorrectness)
        b.setdefault("ref_sentence", sent_of[b["xquad_id"]][0 if b["q_lang"] == "en" else 1])
    ok = [b for b in bases if b["ok"]]
    corpus = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "corpus.jsonl")}
    # distractor refs for entity swaps: other entity-like references, same language, preferring the same article
    ent_refs = {
        "en": [
            (r["para"], r["ref_en"])
            for r in allqa
            if r["ref_en"][:1].isupper() and not re.search(r"\d", r["ref_en"]) and len(r["ref_en"].split()) <= 5
        ],
        "ar": [(r["para"], r["ref_ar"]) for r in allqa if not re.search(r"[\d٠-٩]", r["ref_ar"]) and len(r["ref_ar"].split()) <= 5],
    }
    art = {c["pi"]: c["article"] for c in corpus.values()}

    def distractors(b):
        same = [x for p, x in ent_refs[b["q_lang"]] if p != b["para"] and art.get(p) == art.get(b["para"])]
        other = [x for p, x in ent_refs[b["q_lang"]] if art.get(p) != art.get(b["para"])]
        rng.shuffle(same), rng.shuffle(other)
        return same[:8], other[:16]

    items: list[dict] = []
    rejected = Counter()
    counts = Counter()
    used_roles: dict[str, set] = defaultdict(set)  # xquad_id -> error types already built from it

    def add(
        etype, b, answer, contexts, question=None, reference=None, answerable=True, extra=None, tag="", ref_sentence=None, excluded=None
    ):
        if counts[etype] >= TARGET[etype]:
            return False
        lbl = dict(LABELS[etype])
        items.append(
            {
                "id": f"M{len(items):03d}",
                "etype": etype,
                "base": b["xquad_id"] + tag,
                "para": b["para"],
                "split": split_of(b["para"]),
                "q_lang": b["q_lang"],
                "gold_lang": b["gold_lang"],
                "question": question or b["question"],
                "reference": reference or b["reference"],
                "answerable": answerable,
                "answer": answer,
                "ref_sentence": ref_sentence or b["ref_sentence"],
                "contexts": contexts,
                "cited": cited_ids(answer),
                "gold_id": b["gold_id"],
                "labels": lbl,
                "acceptable": acceptable(lbl),
                "extra": extra or {},
                "excluded": excluded,
            }
        )
        counts[etype] += 1
        used_roles[b["xquad_id"]].add(etype)
        return True

    def ctx_with_gold(b):  # the stored context (gold + 2 distractors, shuffled)
        return b["contexts"]

    def ctx_without_gold(b):
        spare = [c for c in (b["contexts"] + b["spare"]) if c["id"] != b["gold_id"]]
        out, seen = [], set()
        for c in spare:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
        ctx = out[:3]
        rng.shuffle(ctx)
        return ctx

    singles = [b for b in ok if b["role"] == "single"]
    rng.shuffle(singles)
    # interleave English and Arabic questions so every error type is filled ~50/50 (English bases are scarcer because
    # more of them are dropped for answering in the wrong language)
    by_l = {l: [b for b in singles if b["q_lang"] == l] for l in ("en", "ar")}
    singles = [b for pair in zip(by_l["en"], by_l["ar"]) for b in pair] + by_l["en"][len(by_l["ar"]) :] + by_l["ar"][len(by_l["en"]) :]

    # --- clean + numeral variants + wrong_number (numeric bases) --------------------------------------------------
    nums = [b for b in singles if b["kind"] == "num"]
    ents = [b for b in singles if b["kind"] == "ent"]
    for b in nums:
        if counts["wrong_number"] < TARGET["wrong_number"]:
            r = plant_wrong_number(b, rng)
            if r:
                add("wrong_number", b, r[0], ctx_with_gold(b), extra=r[1])
                if counts["clean"] < TARGET["clean"] // 2:  # twin clean item (same base) for paired comparison
                    add("clean", b, b["answer"], ctx_with_gold(b), tag="")
            else:
                rejected["wrong_number: number token not found"] += 1
    for b in nums:
        if "wrong_number" in used_roles[b["xquad_id"]]:
            continue
        if counts["clean_numeral"] < TARGET["clean_numeral"]:
            r = plant_numeral_variant(b, rng)
            if r:
                add("clean_numeral", b, r[0], ctx_with_gold(b), extra=r[1])
            else:
                rejected["clean_numeral: no variant"] += 1
    # --- wrong_entity ------------------------------------------------------------------------------------------------
    for b in ents:
        if counts["wrong_entity"] < TARGET["wrong_entity"]:
            r = plant_wrong_entity(b, distractors(b), rng)
            if r:
                add("wrong_entity", b, r[0], ctx_with_gold(b), extra=r[1])
                if counts["clean"] < TARGET["clean"] * 3 // 4:
                    add("clean", b, b["answer"], ctx_with_gold(b))
            else:
                rejected["wrong_entity: ref not in answer / no distractor"] += 1
    # --- unsupported_claim, wrong_language, wrong_citation (any single base not used above, then reuse) -----------------
    rest = [b for b in singles if not used_roles[b["xquad_id"]]] + [b for b in singles if used_roles[b["xquad_id"]]]
    for b in rest:
        if counts["unsupported_claim"] < TARGET["unsupported_claim"] and "unsupported_claim" not in used_roles[b["xquad_id"]]:
            r = plant_unsupported(b, llm)
            if r:
                add("unsupported_claim", b, r[0], ctx_with_gold(b), extra=r[1])
            else:
                rejected["unsupported_claim: validation failed"] += 1
    rest2 = [b for b in singles if not used_roles[b["xquad_id"]]] + [b for b in singles if used_roles[b["xquad_id"]]]
    for b in rest2:
        if counts["wrong_language"] < TARGET["wrong_language"] and "wrong_language" not in used_roles[b["xquad_id"]]:
            r = plant_language(b, llm)
            if r:
                add("wrong_language", b, r[0], ctx_with_gold(b), extra=r[1])
            else:
                rejected["wrong_language: translation failed validation"] += 1
    for b in rest2:
        if counts["wrong_citation"] < TARGET["wrong_citation"] and "wrong_citation" not in used_roles[b["xquad_id"]]:
            others = [c for c in b["contexts"] if c["id"] != b["gold_id"]]
            ans = re.sub(rf"\[?{b['gold_id']}\]?", lambda m: m.group(0).replace(b["gold_id"], others[0]["id"]), b["answer"])
            if ans != b["answer"] and others and norm(b["reference"]) not in norm(others[0]["text"]):
                add("wrong_citation", b, ans, ctx_with_gold(b), extra={"cited_instead": others[0]["id"]})
            else:
                rejected["wrong_citation: ref also in distractor"] += 1
    # --- abstention triple (same question, three variants) -----------------------------------------------------------
    for b in rest2:
        if counts["correct_abstention"] >= TARGET["correct_abstention"]:
            break
        nog = ctx_without_gold(b)
        nog_text = norm(" ".join(c["text"] for c in nog))
        if len(nog) < 3 or norm(b["ref_en"]) in nog_text or norm(b["ref_ar"]) in nog_text or norm(b["reference"]) in nog_text:
            rejected["abstention: answer string present in no-gold context"] += 1
            continue
        ph = ABST[b["q_lang"]][len(items) % 3]
        add("correct_abstention", b, ph, nog, reference=None, answerable=False, tag="|abst")
        fake = b["answer"].replace(b["gold_id"], nog[0]["id"])
        add("halluc_unanswerable", b, fake, nog, answerable=False, tag="|abst")
        add("wrong_abstention", b, ABST[b["q_lang"]][(len(items) + 1) % 3], ctx_with_gold(b), tag="|abst")
    # fix reference for unanswerable items
    for it in items:
        if not it["answerable"] or it["etype"] == "correct_abstention":
            it["answerable"] = False if it["etype"] in ("correct_abstention", "halluc_unanswerable") else it["answerable"]
    # --- omitted fact (compound questions from the paired bases) ----------------------------------------------------
    pairs = defaultdict(dict)
    for b in ok:
        if b["role"] in ("pair_a", "pair_b"):
            pairs[b["pair"]][b["role"]] = b
    for k, d in sorted(pairs.items()):
        if "pair_a" not in d or "pair_b" not in d:
            rejected["omitted_fact: pair incomplete"] += 1
            continue
        a, c = d["pair_a"], d["pair_b"]
        lang = a["q_lang"]
        join = " Also, " if lang == "en" else " وكذلك، "
        q = (
            a["question"].rstrip()
            + join
            + c["question"][:1].lower() * (lang == "en")
            + (c["question"][1:] if lang == "en" else c["question"])
        )
        ref = f"{a['reference']}; {c['reference']}"
        ctx = a["contexts"] if a["gold_id"] == c["gold_id"] else None
        if ctx is None:
            continue
        full = a["answer"].rstrip() + " " + c["answer"].lstrip()
        tag = f"|pair{k}"
        # label hygiene (found by inspecting judge misses): the omission is only real if the short answer lacks the second
        # fact. Items where the two references are identical or one answer already contains the other fact are kept in the
        # file (ids stay stable) but flagged `excluded` and ignored by every analysis.
        why = None
        if norm(a["reference"]) == norm(c["reference"]):
            why = "pair references identical"
        elif contains_ref(a["answer"], c["reference"]):
            why = "short answer already contains the other fact"
        elif not (contains_ref(a["answer"], a["reference"]) and contains_ref(c["answer"], c["reference"])):
            why = "an answer lacks its own reference (whole-number match)"
        rs = a["ref_sentence"].rstrip() + " " + c["ref_sentence"].lstrip()
        add("omitted_fact", a, a["answer"], ctx, question=q, reference=ref, tag=tag, ref_sentence=rs, excluded=why)
        if counts["clean"] < TARGET["clean"]:
            add("clean", a, full, ctx, question=q, reference=ref, tag=tag, ref_sentence=rs, excluded=why)
    # top up clean with plain bases
    for b in singles:
        if counts["clean"] >= TARGET["clean"]:
            break
        if "clean" not in used_roles[b["xquad_id"]]:
            add("clean", b, b["answer"], ctx_with_gold(b))
    # paired twins of wrong_number items with the wrong value rendered in Arabic-Indic digits (appended last, so the ids
    # of every earlier item are unchanged and cached judge verdicts stay valid)
    by_id_base = {b["xquad_id"]: b for b in ok}
    for it in [i for i in items if i["etype"] == "wrong_number" and i["extra"].get("script") == "ascii"]:
        if counts["wrong_number_arindic"] >= TARGET["wrong_number_arindic"]:
            break
        new = it["extra"]["new"]
        hit = find_num_token(it["answer"], new)
        if not hit:
            continue
        s0, e0, tok = hit
        b = by_id_base[it["base"].split("|")[0]]
        add(
            "wrong_number_arindic",
            b,
            it["answer"][:s0] + (to_arabic_indic(tok)) + it["answer"][e0:],
            ctx_with_gold(b),
            extra={**it["extra"], "script": "ar", "twin_of": it["id"]},
            tag="|arindic",
        )
    # unanswerable items have an explicit "no answer" reference
    for it in items:
        if it["etype"] in ("correct_abstention", "halluc_unanswerable"):
            it["reference"] = "(no answer exists in the documents)"
            it["ref_sentence"] = "The answer is not in the documents." if it["q_lang"] == "en" else "الإجابة غير موجودة في المستندات."

    # label hygiene #2: re-verify (with the whole-number-aware matcher) that every item meant to contain its reference does
    for it in items:
        if it["excluded"] or it["etype"] not in ("clean", "unsupported_claim", "wrong_citation"):
            continue
        parts = [p for p in it["reference"].split(";") if p.strip()]
        if not all(contains_ref(it["answer"], p) for p in parts):
            it["excluded"] = "reference not found as a whole token in the answer"
    with open(DATA / "meta_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    card = {
        "n_items": len(items),
        "n_excluded": sum(1 for i in items if i["excluded"]),
        "excluded": {i["id"]: i["excluded"] for i in items if i["excluded"]},
        "target": TARGET,
        "built": dict(counts),
        "by_split": dict(Counter(i["split"] for i in items)),
        "by_q_lang": dict(Counter(i["q_lang"] for i in items)),
        "acceptable_rate": sum(i["acceptable"] for i in items) / len(items),
        "bases_total": len(bases),
        "bases_kept": len(ok),
        "base_drop_reasons": dict(Counter(w for b in bases for w in b["drop_reasons"])),
        "plant_rejections": dict(rejected),
    }
    json.dump(card, open(DATA / "meta_card.json", "w"), indent=2)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
