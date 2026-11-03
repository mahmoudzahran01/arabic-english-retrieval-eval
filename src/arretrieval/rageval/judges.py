"""LLM-as-judge implementations. Every judge returns {"pass": 0|1|None, "raw": str, ...}  (pass=1 means "answer is OK
on the axis this judge targets"; None = unparsable reply, counted and reported, never silently dropped).

Item schema (a plain dict): question, q_lang, answer, reference, answerable, contexts=[{id,text,lang}], cited=[ids]

Versions:
  faith_v0  naive one-word YES/NO                           -> "before"
  faith_v1  rubric + step-by-step quotes, one call          -> improved (rubric + CoT)
  faith_v2  claim decomposition, then per-claim verdicts    -> improved (decomposition, like Ragas but our prompts)
  cite_v1   faith_v1 restricted to the CITED passages only  -> catches right-answer / wrong-citation
  correct_v0 naive YES/NO vs reference                      -> "before"
  correct_v1 reference-guided rubric + CoT + numeral/completeness/abstention rules
  overall_v0 one monolithic multi-criteria JSON verdict     -> "before" for the end-to-end acceptable label
  overall_pipeline  faith_v2 AND correct_v1 AND cite_v1 AND deterministic language/abstention checks -> "after"
  faith_v3          deterministic sentence split instead of LLM claim extraction (added after seeing v2 drop claims)
  overall_pipeline_v3  the pipeline with faith_v3
"""

from __future__ import annotations

import re

from .llm import LLM, parse_json
from .textnorm import cited_ids, is_abstention, lang_of

NO_ANSWER_REF = "(no answer exists in the documents; the correct response is to say the answer is not in the documents)"


def ctx_block(contexts: list[dict], only: list[str] | None = None) -> str:
    cs = [c for c in contexts if only is None or c["id"] in only]
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in cs)


def _verdict(text: str, pos: str, neg: str) -> int | None:
    m = re.findall(rf"VERDICT:?\**\s*({pos}|{neg})", text.upper().replace("NOT SUPPORTED", "NOT_SUPPORTED"))
    if not m:
        return None
    return 1 if m[-1] == pos else 0


def _yesno(text: str) -> int | None:
    t = text.strip().upper()
    if t.startswith("YES"):
        return 1
    if t.startswith("NO"):
        return 0
    return None


# ----------------------------------------------------------------------------- faithfulness
def faith_v0(it: dict, llm: LLM) -> dict:
    p = (
        f"Context:\n{ctx_block(it['contexts'])}\n\nQuestion: {it['question']}\nAnswer: {it['answer']}\n\n"
        "Is every statement in the Answer supported by the Context? Reply with one word: YES or NO."
    )
    raw = llm.chat(p, max_tokens=8)
    return {"pass": _yesno(raw), "raw": raw}


FAITH_RULES = """An answer is FAITHFUL only if every factual statement in it (each number, name, date, relation) is stated in, or directly entailed by, the Context.
- A detail that is plausible or true in the real world but NOT written in the Context makes the answer UNFAITHFUL.
- A number that differs from the Context in any digit is UNFAITHFUL. The same number written in different scripts or formats (3 = ٣, 1,000 = 1000 = ١٠٠٠) is the same number.
- Ignore citation markers such as [P012], and ignore the language the answer is written in.
- A reply saying the documents do not contain the answer is FAITHFUL only if the Context really does not contain it."""


def faith_v1(it: dict, llm: LLM, only: list[str] | None = None) -> dict:
    p = (
        f"Decide whether the Answer is FAITHFUL to the Context.\n{FAITH_RULES}\n\n"
        f"Context:\n{ctx_block(it['contexts'], only)}\n\nQuestion: {it['question']}\nAnswer: {it['answer']}\n\n"
        "Think step by step: for each statement in the Answer write a supporting quote from the Context (at most 12 words), or NONE. "
        "Be brief (one line per statement). Then write the last line exactly as 'VERDICT: FAITHFUL' or 'VERDICT: UNFAITHFUL'."
    )
    raw = llm.chat(p, max_tokens=350)
    return {"pass": _verdict(raw, "FAITHFUL", "UNFAITHFUL"), "raw": raw}


_CITE_P = """The Answer cites document ids in square brackets. Decide whether the CITED document(s) alone state the information given in the Answer.
- A number, name or fact must be written in a cited document; if it appears only in a document that is NOT cited, the citation is wrong.
- Digit scripts and formats are equivalent (3 = ٣, 1,000 = 1000). Ignore the language of the Answer.

Cited document(s):
{ctx}

Question: {q}
Answer: {a}

Reply with a quote of at most 12 words from the cited document(s) that supports the Answer, or NONE, then on the last line exactly 'VERDICT: SUPPORTED' or 'VERDICT: NOT_SUPPORTED'."""


def cite_v1(it: dict, llm: LLM) -> dict:
    """Is the answer supported by the passages it CITES (not merely by something in the context)?"""
    ids = it.get("cited") or cited_ids(it["answer"])
    if is_abstention(it["answer"]):
        return {"pass": 1, "raw": "abstention: no citation required"}
    valid = [i for i in ids if i in {c["id"] for c in it["contexts"]}]
    if not valid:
        return {"pass": 0, "raw": "no valid citation present"}
    raw = llm.chat(_CITE_P.format(ctx=ctx_block(it["contexts"], valid), q=it["question"], a=it["answer"]), max_tokens=120)
    return {"pass": _verdict(raw, "SUPPORTED", "NOT_SUPPORTED"), "raw": raw, "cited": valid}


_CLAIMS_P = """Split the Answer into atomic factual claims. Each claim must be a short, self-contained sentence in the same language as the Answer (replace pronouns with the names they refer to). Ignore citation markers like [P012]. If the Answer states no fact (for example it only says the information is not in the documents), return an empty list.
Question: {q}
Answer: {a}
Return ONLY a JSON list of strings."""

_VERIFY_P = """For each claim decide whether it is SUPPORTED by the Context, CONTRADICTED by it, or UNSUPPORTED (not stated in it).
Rules: a claim whose number differs in any digit is CONTRADICTED; the same number in different scripts or formats (3 = ٣, 1,000 = 1000) is identical; a detail that is not written in the Context is UNSUPPORTED even if it is true in the real world; names that differ only by translation or transliteration are the same.
Context:
{ctx}

Claims:
{claims}

Return ONLY a JSON list, one object per claim in order: {{"claim": "...", "evidence": "<quote of at most 12 words from the Context, or NONE>", "verdict": "SUPPORTED|CONTRADICTED|UNSUPPORTED"}}"""


def faith_v2(it: dict, llm: LLM) -> dict:
    raw1 = llm.chat(_CLAIMS_P.format(q=it["question"], a=it["answer"]), max_tokens=400)
    try:
        claims = [c for c in parse_json(raw1) if isinstance(c, str) and c.strip()]
    except Exception:  # noqa: BLE001
        return {"pass": None, "raw": raw1, "claims": None}
    if not claims:
        return {"pass": 1, "raw": raw1, "claims": []}
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    raw2 = llm.chat(_VERIFY_P.format(ctx=ctx_block(it["contexts"]), claims=numbered), max_tokens=700)
    try:
        v = parse_json(raw2)
        verdicts = [str(x.get("verdict", "")).upper() for x in v]
    except Exception:  # noqa: BLE001
        return {"pass": None, "raw": raw2, "claims": claims}
    if len(verdicts) != len(claims):
        verdicts = (verdicts + ["UNSUPPORTED"] * len(claims))[: len(claims)]
    sup = sum(1 for x in verdicts if x.startswith("SUPPORTED"))
    return {"pass": int(sup == len(claims)), "raw": raw2, "claims": claims, "verdicts": verdicts, "score": sup / len(claims)}


_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+")


def split_sentences(answer: str) -> list[str]:
    """Deterministic claim units: sentences of the answer with citation markers removed."""
    clean = re.sub(r"\[?P\d{3}(?:\.c\d+)?\]?", "", answer)
    return [x.strip() for x in _SENT_SPLIT.split(clean.strip()) if len(x.strip()) > 2]


def faith_v3(it: dict, llm: LLM) -> dict:
    """faith_v2 with the LLM claim-extraction step replaced by a deterministic sentence split (the LLM decomposer was
    observed to silently drop an appended hallucinated sentence). One call per item."""
    if is_abstention(it["answer"]):
        return {"pass": 1, "raw": "abstention: no factual claim", "claims": []}
    claims = split_sentences(it["answer"])
    if not claims:
        return {"pass": None, "raw": "no sentences", "claims": []}
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    raw = llm.chat(_VERIFY_P.format(ctx=ctx_block(it["contexts"]), claims=numbered), max_tokens=700)
    try:
        v = parse_json(raw)
        verdicts = [str(x.get("verdict", "")).upper() for x in v]
    except Exception:  # noqa: BLE001
        return {"pass": None, "raw": raw, "claims": claims}
    if len(verdicts) != len(claims):
        verdicts = (verdicts + ["UNSUPPORTED"] * len(claims))[: len(claims)]
    sup = sum(1 for x in verdicts if x.startswith("SUPPORTED"))
    return {"pass": int(sup == len(claims)), "raw": raw, "claims": claims, "verdicts": verdicts, "score": sup / len(claims)}


# ----------------------------------------------------------------------------- correctness vs reference
def _ref(it: dict) -> str:
    return it["reference"] if it.get("answerable", True) else NO_ANSWER_REF


def correct_v0(it: dict, llm: LLM) -> dict:
    p = (
        f"Question: {it['question']}\nReference answer: {_ref(it)}\nAnswer: {it['answer']}\n\n"
        "Is the Answer correct, judged against the Reference answer? Reply with one word: YES or NO."
    )
    raw = llm.chat(p, max_tokens=8)
    return {"pass": _yesno(raw), "raw": raw}


CORRECT_RULES = """Grade the Answer against the Reference answer.
- CORRECT: the Answer contains every key fact of the Reference and contradicts none. Wording, order, language (Arabic or English), extra harmless context and citation markers like [P012] do not matter.
- Different scripts or formats of the same number are EQUAL (308 = ٣٠٨, 1,000 = 1000 = ١٠٠٠). A DIFFERENT number is INCORRECT even if it differs by one digit (308 vs 309, 1995 vs 1996) or by a factor of ten.
- Names must refer to the same entity; translation or transliteration differences are fine, a different entity is INCORRECT.
- If the Reference lists several facts (separated by ';') and the Answer is missing any of them, it is INCORRECT.
- If the Reference says no answer exists in the documents, the Answer is CORRECT only if it says the answer is not in the documents; any specific answer is INCORRECT.
- If the Answer says the answer is not in the documents but the Reference gives a fact, it is INCORRECT.
- Judge ONLY agreement with the Reference. Do not judge whether the Answer is supported by any document."""


def correct_v1(it: dict, llm: LLM) -> dict:
    p = (
        f"{CORRECT_RULES}\n\nQuestion: {it['question']}\nReference answer: {_ref(it)}\nAnswer: {it['answer']}\n\n"
        "Think step by step in at most 3 short lines (the key facts of the Reference and whether each appears in the Answer), "
        "then write the last line exactly as 'VERDICT: CORRECT' or 'VERDICT: INCORRECT'."
    )
    raw = llm.chat(p, max_tokens=300)
    return {"pass": _verdict(raw, "CORRECT", "INCORRECT"), "raw": raw}


# ----------------------------------------------------------------------------- monolithic judge ("before") and pipeline ("after")
def overall_v0(it: dict, llm: LLM) -> dict:
    p = (
        f"You are grading an answer produced by a RAG system.\n\nDocuments:\n{ctx_block(it['contexts'])}\n\n"
        f"Question: {it['question']}\nReference answer: {_ref(it)}\nSystem answer: {it['answer']}\n\n"
        "Check: (1) faithful: every claim is supported by the documents; (2) correct: matches the reference; "
        "(3) language: the answer is in the same language as the question; (4) citation: the cited document ids support the answer; "
        "(5) abstention: if the documents lack the answer the system should say so, otherwise it should answer. "
        'Reply with ONLY JSON: {"faithful": 0|1, "correct": 0|1, "language": 0|1, "citation": 0|1, "abstention": 0|1, "acceptable": 0|1}'
    )
    raw = llm.chat(p, max_tokens=120)
    try:
        j = parse_json(raw)
        parts = {k: int(j[k]) for k in ("faithful", "correct", "language", "citation", "abstention", "acceptable")}
        return {"pass": parts["acceptable"], "raw": raw, "parts": parts}
    except Exception:  # noqa: BLE001
        return {"pass": None, "raw": raw}


def det_checks(it: dict) -> dict:
    """Deterministic checks no LLM should be trusted with: answer language and abstention-vs-answerability."""
    ab = is_abstention(it["answer"])
    lang_ok = int(lang_of(re.sub(r"\[P\d{3}\]", "", it["answer"])) == it["q_lang"])
    abst_ok = int(ab == (not it.get("answerable", True)))  # abstain iff the question is unanswerable
    return {"language": lang_ok, "abstention": abst_ok}


def _pipeline(it: dict, llm: LLM, faith_fn) -> dict:
    d = det_checks(it)
    f = faith_fn(it, llm)
    c = correct_v1(it, llm)
    ci = cite_v1(it, llm)
    parts = {"faithful": f["pass"], "correct": c["pass"], "citation": ci["pass"], **d}
    known = [v for v in parts.values() if v is not None]
    # any definite failure fails the item even if another part was unparsable; unparsable-only => None
    if any(v == 0 for v in known):
        return {"pass": 0, "parts": parts}
    if any(v is None for v in parts.values()):
        return {"pass": None, "parts": parts}
    return {"pass": 1, "parts": parts}


def overall_pipeline(it: dict, llm: LLM) -> dict:
    """faith_v2 (LLM claim decomposition) AND correct_v1 AND cite_v1 AND deterministic language/abstention checks."""
    return _pipeline(it, llm, faith_v2)


def overall_pipeline_v3(it: dict, llm: LLM) -> dict:
    """Same, with faith_v3 (deterministic sentence split) as the faithfulness component."""
    return _pipeline(it, llm, faith_v3)


JUDGES = {
    "faith_v0": faith_v0,
    "faith_v1": faith_v1,
    "faith_v2": faith_v2,
    "cite_v1": cite_v1,
    "correct_v0": correct_v0,
    "correct_v1": correct_v1,
    "overall_v0": overall_v0,
    "overall_pipeline": overall_pipeline,
    "faith_v3": faith_v3,
    "overall_pipeline_v3": overall_pipeline_v3,
}
