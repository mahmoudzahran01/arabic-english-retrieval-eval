"""Digit handling, string-match references, abstention / language / citation detectors."""

from arretrieval.rageval import textnorm as T


def test_digits_and_numbers():
    assert T.to_ascii_digits("٣٠٨ نقطة") == "308 نقطة"
    assert T.to_arabic_indic("1995") == "١٩٩٥"
    assert T.numbers("1,500 and ٣٠٨ and 12.5") == ["1500", "308", "12.5"]
    assert T.contains_ref("The Panthers gave up ٣٠٨ points", "308")  # numeral variants match
    assert not T.contains_ref("The Panthers gave up 309 points", "308")  # near miss does not
    assert not T.contains_ref("The engine runs at 565 degrees", "5")  # numeric refs are whole-number matches
    assert not T.contains_ref("sent 400 men", "40") and T.contains_ref("sent 40 men.", "40")
    assert T.contains_ref("population 1,000", "1000") and T.contains_ref("32.9% did not vote", "32.9%")
    assert T.contains_ref("اللغة العربيّة", "العربية")  # diacritics / taa-marbuta folded


def test_abstention_and_language_and_citation():
    assert T.is_abstention("Not in the documents.")
    assert T.is_abstention("غير موجود في المستندات.")
    assert T.is_abstention("The documents do not contain the answer to this question.")
    assert not T.is_abstention("The Panthers gave up 308 points [P012].")
    assert T.lang_of("كم عدد النقاط؟") == "ar" and T.lang_of("How many points?") == "en"
    assert T.cited_ids("Yes [P012] and [P044], again [P012].") == ["P012", "P044"]
