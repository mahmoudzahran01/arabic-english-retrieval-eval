"""The WiSE-curve loader accepts the documented schema and its variants; the plot is written; unreadable input draws nothing."""

from __future__ import annotations

import json

import pytest

from report import make_figures as mf


def _write(tmp_path, obj):
    p = tmp_path / "alpha_curve.json"
    p.write_text(json.dumps(obj))
    return p


DOC = {
    "selected_alpha": 0.75,
    "points": [
        {
            "alpha": 1.0,
            "test": {"ar->en": 0.739, "macro": 0.848},
            "guards": {"MIRACLRetrievalHardNegatives.v2": 0.668, "NamaaMrTydiReranking": 0.694},
        },
        {
            "alpha": 0.0,
            "test": {"ar->en": 0.633, "macro": 0.748},
            "guards": {"MIRACLRetrievalHardNegatives.v2": 0.79, "NamaaMrTydiReranking": 0.794, "SemRel24STS": 0.591},
        },
        {
            "alpha": 0.75,
            "test": {"ar->en": 0.735, "macro": 0.844},
            "guards": {"MIRACLRetrievalHardNegatives.v2": 0.757, "NamaaMrTydiReranking": 0.721},
        },
    ],
}


def test_documented_schema_is_read_sorted_and_scaled(tmp_path):
    pts, sel = mf.load_curve(_write(tmp_path, DOC))
    assert sel == 0.75 and [p.alpha for p in pts] == [0.0, 0.75, 1.0]
    assert pts[0].values["ar->en"] == pytest.approx(63.3) and pts[0].values["miracl"] == pytest.approx(79.0)
    xs, ys = mf.series(pts, "semrel")
    assert xs == [0.0] and ys == [pytest.approx(59.1)]  # a series may exist for some alphas only


def test_percent_values_are_not_rescaled(tmp_path):
    doc = {"points": [{"alpha": 0.5, "test": {"ar->en": 70.0, "macro": 80.0}}]}
    pts, sel = mf.load_curve(_write(tmp_path, doc))
    assert sel is None and pts[0].values["macro"] == 80.0


def test_dict_keyed_by_alpha_and_flat_keys(tmp_path):
    doc = {
        "meta": {"selected_alpha": 0.5},
        "curve": {
            "0.25": {"test_ar2en": 0.7, "test_macro": 0.8, "miracl": 0.77},
            "0.5": {"ar->en": 0.72, "macro": 0.82, "MIRACL-HN": 0.76},
        },
    }
    pts, sel = mf.load_curve(_write(tmp_path, doc))
    assert sel == 0.5 and [p.alpha for p in pts] == [0.25, 0.5]
    assert pts[0].values == {"ar->en": pytest.approx(70.0), "macro": pytest.approx(80.0), "miracl": pytest.approx(77.0)}


def test_unreadable_or_empty_input_gives_no_points(tmp_path):
    assert mf.load_curve(tmp_path / "missing.json") == ([], None)
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert mf.load_curve(p) == ([], None)
    assert mf.load_curve(_write(tmp_path, {"points": [{"alpha": 0.5, "unrelated": 3}]}))[0] == []


def test_plot_writes_pdf_and_png_and_names_missing_series(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "FIG_DIR", tmp_path)
    pts, sel = mf.load_curve(
        _write(tmp_path, {"selected_alpha": 0.75, "points": [{"alpha": a, "test": {"ar->en": 0.6 + a / 10}} for a in (0, 0.5, 0.75, 1)]})
    )
    paths = mf.plot(pts, sel, tmp_path / "alpha_curve")
    assert [p.suffix for p in paths] == [".pdf", ".png"] and all(p.stat().st_size > 1000 for p in paths)
