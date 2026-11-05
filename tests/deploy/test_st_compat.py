"""The comparison logic of the sentence-transformers compatibility check (no sentence-transformers needed: it works on the files the check writes)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from deploy.st_compat import check_compat, compare_compat


def _write(d, label, name, kind, emb, **info):
    base = {"loads": emb is not None, "max_seq_length": 256, "pooling": "mean", "dimension": 4, "module_classes": []}
    base.update(info)
    (d / f"{label}__{name}.{kind}.json").write_text(json.dumps(base))
    if emb is not None:
        np.savez(d / f"{label}__{name}.{kind}.npz", emb=emb)


def _run(d, tol=1e-5, names=("m",)):
    out = d / "out.json"
    argv = ["compare_compat", str(d), *names, "--json", str(out), "--tol", str(tol)]
    old, sys.argv = sys.argv, argv
    try:
        compare_compat.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())


@pytest.fixture
def emb():
    return np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)


def test_matching_embeddings_pass_and_st5_original_is_expected_to_fail(tmp_path, emb, capsys):
    _write(tmp_path, "st6", "m", "orig", emb)
    _write(tmp_path, "st6", "m", "compat", emb)
    _write(tmp_path, "st5", "m", "orig", None, error="ModuleNotFoundError")
    _write(tmp_path, "st5", "m", "compat", emb + 2e-7)
    res = _run(tmp_path)["m"]
    assert res["passes"] and res["max_abs_diff_st6orig_vs_st5compat"] < 1e-5
    assert "| m | yes | no | yes | yes | 256 / 256 | mean / mean |" in capsys.readouterr().out


def test_a_difference_above_tolerance_or_another_pooling_or_length_fails(tmp_path, emb):
    _write(tmp_path, "st6", "m", "orig", emb)
    _write(tmp_path, "st5", "m", "compat", emb + 1e-3)
    assert not _run(tmp_path)["m"]["passes"]
    _write(tmp_path, "st5", "m", "compat", emb, pooling="cls")
    assert not _run(tmp_path)["m"]["passes"]
    _write(tmp_path, "st5", "m", "compat", emb, max_seq_length=512)
    assert not _run(tmp_path)["m"]["passes"]
    _write(tmp_path, "st5", "m", "compat", emb)
    assert _run(tmp_path)["m"]["passes"]


def test_a_compat_folder_that_does_not_load_fails(tmp_path, emb):
    _write(tmp_path, "st6", "m", "orig", emb)
    _write(tmp_path, "st5", "m", "compat", None, error="x")
    assert not _run(tmp_path)["m"]["passes"]


def test_pooling_mode_reads_both_library_generations():
    class Old:
        def get_pooling_mode_str(self):
            return "cls"

    class New:
        pooling_mode = "mean"

    assert check_compat.pooling_mode(Old()) == "cls" and check_compat.pooling_mode(New()) == "mean"
