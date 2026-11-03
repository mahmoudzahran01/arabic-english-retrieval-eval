"""Arabic orthographic normalization, light stemming and tokenization.

Why this module exists (see EXPLAINER.md): Arabic has many surface forms for the same word.
A lexical retriever (BM25) matches *tokens*, so every spelling variant is a different token unless
we collapse them first. Neural embedders learn some of this invariance, but not all of it.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat (fatha/damma/kasra, tanwin, shadda, sukun), superscript alef, Quranic annotation marks.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"  # kashida: stretches letters visually, carries no meaning
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_YEH_FORMS = str.maketrans({"ى": "ي", "ئ": "ي"})  # alef maqsura / hamza-on-yeh -> yeh
_WAW_FORMS = str.maketrans({"ؤ": "و"})  # hamza-on-waw -> waw
_TEH_MARBUTA = str.maketrans({"ة": "ه"})  # ta marbuta -> heh (common IR convention)
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Raw tokens keep combining marks / tatweel inside the word (what a naive whitespace tokenizer would do).
_RAW_TOKEN = re.compile(r"[\wؐ-ًؚ-ٰٟۖ-ۭـ]+", re.UNICODE)
_NORM_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize_ar(
    text: str,
    *,
    alef: bool = True,
    yeh: bool = True,
    teh_marbuta: bool = True,
    diacritics: bool = True,
    tatweel: bool = True,
    digits: bool = True,
) -> str:
    """Collapse the common Arabic orthographic variants. Each flag can be switched off for ablations."""
    text = unicodedata.normalize("NFKC", text)  # also folds presentation forms (e.g. ﻻ -> لا)
    if digits:
        text = text.translate(_DIGITS)
    if diacritics:
        text = _DIACRITICS.sub("", text)
    if tatweel:
        text = text.replace(_TATWEEL, "")
    if alef:
        text = text.translate(_ALEF_FORMS)
    if yeh:
        text = text.translate(_YEH_FORMS).translate(_WAW_FORMS)
    if teh_marbuta:
        text = text.translate(_TEH_MARBUTA)
    return text


# Simplified light stemmer in the spirit of Larkey et al. "light10": strip clitic prefixes
# (conjunction wa-, article al-, prepositions bi-/ka-/li-/fa-) and a few common suffixes.
# Applied AFTER normalize_ar, so teh marbuta is already heh and alef maqsura already yeh.
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


def light_stem(token: str) -> str:
    if not token or not ("؀" <= token[0] <= "ۿ"):
        return token
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p) :]
            break
    else:
        if token.startswith("و") and len(token) >= 5:
            token = token[1:]
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str, mode: str = "norm") -> list[str]:
    """mode: 'raw'   - lowercase + split, keep diacritics/tatweel/variants (naive baseline)
    'norm'  - Arabic orthographic normalization, then split
    'stem'  - 'norm' + light prefix/suffix stripping
    """
    if mode == "raw":
        return _RAW_TOKEN.findall(text.lower())
    toks = _NORM_TOKEN.findall(normalize_ar(text).lower())
    if mode == "stem":
        return [light_stem(t) for t in toks]
    if mode == "norm":
        return toks
    raise ValueError(f"unknown tokenization mode {mode!r}")


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def dedupe_key(text: str) -> str:
    """Aggressive canonical form used to detect duplicate passages/questions across sources."""
    return _WS.sub(" ", _PUNCT.sub(" ", normalize_ar(text).lower())).strip()
