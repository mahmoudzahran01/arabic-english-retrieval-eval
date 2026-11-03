"""Tolerant JSON extraction from LLM replies."""

from arretrieval.rageval.llm import parse_json


def test_parse_json_single_element_list():
    assert parse_json('[{"verdict": "SUPPORTED"}]') == [{"verdict": "SUPPORTED"}]  # regression: used to return the inner dict
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_json('["x", "y"]') == ["x", "y"]
