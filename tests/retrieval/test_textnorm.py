"""Hand-checked examples for the Arabic normalizer, light stemmer and tokenizer."""

from arretrieval.retrieval import textnorm as T


def test_arabic_normalization_variants_collapse():
    # alef variants, ya/alef-maqsura, ta-marbuta/ha, tatweel, diacritics, Arabic-Indic digits
    assert T.normalize_ar("أحمد إبراهيم آمن") == "احمد ابراهيم امن"
    assert T.normalize_ar("مستشفى") == T.normalize_ar("مستشفي")
    assert T.normalize_ar("مدرسة") == T.normalize_ar("مدرسه")
    assert T.normalize_ar("كتـــاب") == "كتاب"
    assert T.normalize_ar("كَتَبَ") == "كتب"
    assert T.normalize_ar("سنة ٢٠١٥") == "سنه 2015"


def test_tokenize_modes():
    # raw keeps variants distinct, norm merges them, stem also strips al-/wa- and plural/feminine suffixes
    assert T.tokenize("الْكِتَاب", "raw") != T.tokenize("الكتاب", "raw")
    assert T.tokenize("الْكِتَاب", "norm") == T.tokenize("الكتاب", "norm") == ["الكتاب"]
    assert T.tokenize("الكتاب", "stem") == ["كتاب"]
    assert T.tokenize("والمعلمون", "stem") == ["معلم"]
    assert T.tokenize("Hello, World 42", "norm") == ["hello", "world", "42"]
    assert T.light_stem("hello") == "hello"  # non-Arabic tokens untouched
    assert T.light_stem("ال") == "ال"  # never strips a token to nothing
