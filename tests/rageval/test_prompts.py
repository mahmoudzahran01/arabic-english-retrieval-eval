"""The generator prompts differ exactly where the experiments say they do."""

import pytest

from arretrieval.rageval import prompts as P


def test_v1_asks_for_citations_and_exact_abstention():
    s = P.system_prompt("v1", "en")
    assert "[P012]" in s and f'"{P.ABSTAIN_EN}"' in s and "same language as the question" in s


def test_abstention_string_follows_the_question_language():
    assert P.ABSTAIN_AR in P.system_prompt("v1", "ar") and P.ABSTAIN_EN not in P.system_prompt("v1", "ar")


def test_v3_adds_an_explicit_output_language_instruction():
    assert "write your answer in Arabic" in P.system_prompt("v3", "ar")
    assert "write your answer in English" in P.system_prompt("v3", "en")
    assert "write your answer in" not in P.system_prompt("v1", "en")


def test_v2_adds_strict_grounding_rules_only():
    s1, s2 = P.system_prompt("v1", "en"), P.system_prompt("v2", "en")
    assert s2.startswith(s1) and "Never use outside knowledge" in s2


def test_unknown_version_raises():
    with pytest.raises(ValueError):
        P.system_prompt("v9", "en")


def test_user_prompt_lists_documents_with_ids_and_language():
    msgs = P.messages("Q?", [{"id": "P001", "lang": "ar", "text": "نص"}], "v1", "en")
    assert msgs[1]["content"].startswith("Documents:\n[P001] (ar) نص") and msgs[1]["content"].endswith("Question: Q?\nAnswer:")
