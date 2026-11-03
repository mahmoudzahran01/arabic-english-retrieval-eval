"""Small text utilities: Arabic normalisation, digit handling, language detection.

Everything here is deterministic and unit-tested in tests/rageval/test_textnorm.py. The planted-error
generators and the string-match baselines depend on these, so they must be boring.
"""

from __future__ import annotations

import re
import unicodedata

_AR_INDIC = "٠١٢٣٤٥٦٧٨٩"  # U+0660..0669
_EXT_INDIC = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0..06F9 (Persian/Urdu forms)
_TO_ASCII = {ord(c): str(i) for i, c in enumerate(_AR_INDIC)}
_TO_ASCII.update({ord(c): str(i) for i, c in enumerate(_EXT_INDIC)})
_TO_ARINDIC = {ord(str(i)): c for i, c in enumerate(_AR_INDIC)}

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_NUM_RE = re.compile(r"\d+(?:[.,،٫]\d+)*")


def to_ascii_digits(s: str) -> str:
    """٣٠٨ -> 308 (also Persian forms)."""
    return s.translate(_TO_ASCII)


def to_arabic_indic(s: str) -> str:
    """308 -> ٣٠٨ (only touches ASCII digits)."""
    return s.translate(_TO_ARINDIC)


def norm_ar(s: str) -> str:
    """Light Arabic normalisation: strip diacritics/tatweel, unify alef/yaa/taa-marbuta."""
    s = unicodedata.normalize("NFKC", s)
    s = _DIACRITICS.sub("", s).replace(_TATWEEL, "")
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    return s


def norm(s: str) -> str:
    """Case-fold, digit-fold, Arabic-fold, collapse whitespace/punctuation for string matching."""
    s = norm_ar(to_ascii_digits(s)).lower()
    s = s.replace("٬", ",").replace("،", ",").replace("٫", ".")
    s = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", s)  # 1,000 / 1 000 -> 1000
    s = re.sub(r"[^\w\s.%]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_NUMERIC_REF = re.compile(r"^[\d.,%\s$€£-]+$")


def contains_ref(answer: str, ref: str) -> bool:
    """True if the (normalised) reference string occurs in the (normalised) answer.

    Numeric references must match as a whole number, not as a substring of a longer one ("5" is NOT in "565", "40" is
    NOT in "400"): a plain substring test silently accepts wrong numbers.
    """
    r, a = norm(ref), norm(answer)
    if not r:
        return False
    if _NUMERIC_REF.match(r):
        return re.search(rf"(?<![\d.]){re.escape(r)}(?![\d]|\.\d)", a) is not None
    return r in a


def numbers(s: str) -> list[str]:
    """All numbers in s as ASCII strings with separators stripped ('1,000' -> '1000')."""
    out = []
    for m in _NUM_RE.findall(to_ascii_digits(s)):
        out.append(re.sub(r"[,،]", "", m).replace("٫", "."))
    return out


def arabic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ")
    return ar / len(letters)


def lang_of(s: str) -> str:
    """'ar' if most letters are Arabic script, else 'en'. Good enough for ar/en-only data."""
    return "ar" if arabic_ratio(s) >= 0.5 else "en"


ABSTAIN_PATTERNS = [
    r"not in the documents?",
    r"not (?:found|present|mentioned|stated|available) in the (?:provided )?(?:documents?|passages?|context)",
    r"(?:documents?|passages?|context) (?:do(?:es)? not|don'?t) (?:contain|mention|provide|include|state)",
    r"غير موجود(?:ة)? في المستندات",
    r"لا (?:توجد|يوجد|تتضمن|تحتوي|تذكر)[^.]{0,60}المستندات",
    r"المستندات (?:لا|لم)[^.]{0,40}(?:تحتوي|تتضمن|تذكر|تقدم)",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.I)
_CITE_RE = re.compile(r"P\d{3}")


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search(norm_ar(answer)) or _ABSTAIN_RE.search(answer))


def cited_ids(answer: str) -> list[str]:
    """Passage ids cited anywhere in the answer, in order, de-duplicated. Chunk ids P012.c1 -> P012."""
    seen, out = set(), []
    for m in _CITE_RE.findall(answer):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
