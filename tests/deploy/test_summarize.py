"""CPU unit tests for the deployment helpers (no torch / onnxruntime needed)."""

from __future__ import annotations

import json

import numpy as np

from deploy import common, summarize

DIR_OF = {"s_ar": "ar->ar", "s_en": "en->en", "x_ar2en": "ar->en", "x_en2ar": "en->ar"}


def _perq(rng, n=400, shift=0.0):
    return {k: np.clip(rng.random(n) + shift, 0, 1) for k in DIR_OF}


def test_point_estimate_is_macro_of_direction_means():
    rng = np.random.default_rng(0)
    p = _perq(rng)
    pe = summarize.point_estimate(p, DIR_OF)
    assert abs(pe["macro"] - np.mean([p[k].mean() for k in DIR_OF])) < 1e-12


def test_paired_bootstrap_of_identical_systems_is_zero():
    rng = np.random.default_rng(1)
    p = _perq(rng)
    ci = summarize.macro_bootstrap(p, {k: v.copy() for k, v in p.items()}, DIR_OF, n_boot=200)
    assert ci["macro"]["lo"] == 0.0 and ci["macro"]["hi"] == 0.0


def test_paired_bootstrap_detects_a_shift_and_is_reproducible():
    rng = np.random.default_rng(2)
    a = _perq(rng)
    b = {k: v + 0.05 for k, v in a.items()}
    ci1 = summarize.macro_bootstrap(a, b, DIR_OF, n_boot=300)
    ci2 = summarize.macro_bootstrap(a, b, DIR_OF, n_boot=300)
    assert ci1 == ci2
    assert 0.049 < ci1["macro"]["lo"] <= ci1["macro"]["hi"] < 0.051
    assert set(ci1) == {"ar->ar", "en->en", "ar->en", "en->ar", "macro"}


def test_single_system_ci_brackets_the_estimate():
    rng = np.random.default_rng(3)
    a = _perq(rng)
    ci = summarize.macro_bootstrap(a, None, DIR_OF, n_boot=300)["macro"]
    est = summarize.point_estimate(a, DIR_OF)["macro"]
    assert ci["lo"] < est < ci["hi"]


def test_direction_map_covers_the_test_slices():
    dm = summarize.default_direction_map()
    assert {"xquad_ar2en", "xquad_en2ar", "tydi_ar", "tydi_en", "arcd_ar"} <= set(dm)


def test_mem_available_parses_meminfo(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal: 100 kB\nMemAvailable:   16777216 kB\n")
    assert abs(common.mem_available_gb(str(f)) - 16.0) < 1e-9
    assert common.mem_available_gb(str(tmp_path / "missing")) is None


def test_wait_for_memory_returns_when_enough(monkeypatch):
    monkeypatch.setattr(common, "mem_available_gb", lambda *a, **k: 40.0)
    common.wait_for_memory(15.0, poll_s=0, timeout_s=1)


def test_wait_for_memory_times_out(monkeypatch):
    monkeypatch.setattr(common, "mem_available_gb", lambda *a, **k: 3.0)
    try:
        common.wait_for_memory(15.0, poll_s=0, timeout_s=-1)
    except TimeoutError as e:
        assert "3.0" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected a TimeoutError")


def test_percentile_and_dir_size(tmp_path):
    assert common.percentile([1, 2, 3, 4, 5], 50) == 3.0
    (tmp_path / "a.bin").write_bytes(b"x" * 2**20)
    assert abs(common.dir_size_mb(tmp_path) - 1.0) < 1e-9


def test_markdown_tables_render_from_synthetic_results(tmp_path):
    rng = np.random.default_rng(4)
    q = tmp_path / "quality"
    q.mkdir()
    ref = _perq(rng)
    variants = {"torch-fp32": ref, "onnx-int8": {k: np.clip(v - 0.01, 0, 1) for k, v in ref.items()}}
    for variant, p in variants.items():
        np.savez_compressed(q / f"m__{variant}.test.perq.npz", **{k: v.astype(np.float32) for k, v in p.items()})
        (q / f"m__{variant}.test.json").write_text(json.dumps({"model": "m", "variant": variant}))
    quality = summarize.load_quality(tmp_path)
    assert set(quality) == {("m", "torch-fp32"), ("m", "onnx-int8")}
    md = summarize.markdown_quality(quality, "m", DIR_OF)
    assert "PyTorch fp32 (reference)" in md and "ONNX int8 (dynamic)" in md and "reference" in md
    row = summarize.quality_rows(quality, "m", DIR_OF)[1]
    assert -0.0105 < row["delta"]["macro"] < -0.0095
