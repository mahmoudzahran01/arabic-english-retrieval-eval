"""Study tables on small synthetic inputs: pending files give TODO cells, discovered files give numbers; quotes are LaTeX-safe; the trail extractor finds its entries."""

from __future__ import annotations

import json

import numpy as np
import pytest

from report import extract_prereg, study_tables, tabcommon


class Host(study_tables.StudyTables):
    def __init__(self):
        from deploy import summarize as DS

        self.nums, self.dir_of = {}, DS.default_direction_map()

    def num(self, name, value):
        self.nums[name] = value


@pytest.fixture
def v2dir(tmp_path, monkeypatch):
    for mod in (tabcommon, study_tables):
        monkeypatch.setattr(mod, "V2", tmp_path)
    return tmp_path


def _system(path, stem, split, ar2en, macro=None, n=20, seed=0):
    rng = np.random.default_rng(seed)
    slices = {"xquad_ar2en": ar2en, "xquad_en2ar": ar2en + 0.1, "tydi_ar": 0.7, "tydi_en": 0.8}
    perq = {f"{k}|mrr@10": np.clip(rng.normal(v, 0.1, n), 0, 1) for k, v in slices.items()}
    np.savez_compressed(path / f"{stem}.{split}.perq.npz", **perq)
    dirs = {
        "ar->en": ar2en,
        "en->ar": ar2en + 0.1,
        "ar->ar": 0.7,
        "en->en": 0.8,
        "macro": macro if macro is not None else (ar2en * 2 + 0.1 + 1.5) / 4,
    }
    (path / f"{stem}.{split}.json").write_text(
        json.dumps(
            {"slices": {k: {"mrr@10": v, "n": n} for k, v in slices.items()}, "directions": {k: {"mrr@10": v} for k, v in dirs.items()}}
        )
    )


def test_latex_text_escapes_and_maps_unicode():
    s = tabcommon.latex_text("a_b & 5% #1 {x} ~ ^ -> ≥ α é")
    assert r"a\_b \& 5\% \#1 \{x\}" in s and r"$\to$" in s and r"$\ge$" in s and r"$\alpha$" in s and s.endswith("?")
    assert "_" not in s.replace(r"\_", "")


def test_signed_formatting_never_prints_negative_zero():
    assert tabcommon.sfmt(-0.04) == "0.0" and tabcommon.sfmt(0.04) == "0.0"
    assert tabcommon.sfmt(-1.26) == "$-$1.3" and tabcommon.spct(0.0123) == "+1.2"
    assert r"\_" in tabcommon.todo("alpha_curve.json pending")


PROGRESS = """# log
- RESULTS 2026-11-04 ~11:05: stuff
- WiSE of m3_b: raw 85.3. Selection rule fixed BEFORE any guard eval: dev macro within 0.5 of raw -> alpha 0.75. Queued more.
## 2026-11-04 ~14:00 coordinator: fix the MIRACL forgetting properly (replay), PRE-REGISTERED selection rule
- RULE: qualifies if devM >= x
- 13:53 replay set after filters: ...
## 2026-11-04 ~19:05 - PRE-REGISTRATION: validation suite for the final pick
Order of work: a -> b
- 2026-11-04 ~19:20 CORRECTION to the above
- 18:36 RE-PRIORITISATION (coordinator): not part of the rule
## 2026-11-04 ~21:35 - AMENDMENT 2 to the pre-registered suite selection
DECISIONS: A. LoRA rerun
- 21:23 drive15.sh started (log line, not part of the rule)
"""


def test_extract_prereg_finds_entries_and_cuts_at_the_right_lines():
    e = {x["id"]: x for x in extract_prereg.extract(PROGRESS)}
    assert e["alpha_rule"]["timestamp"] == "2026-11-04 11:05" and e["alpha_rule"]["text"].endswith("alpha 0.75.")
    assert (
        e["rule_1"]["timestamp"] == "2026-11-04 14:00" and "RULE: qualifies" in e["rule_1"]["text"] and "13:53" not in e["rule_1"]["text"]
    )
    assert "CORRECTION" in e["amendment_1"]["text"] and "RE-PRIORITISATION" not in e["amendment_1"]["text"]
    assert e["amendment_2"]["timestamp"] == "2026-11-04 21:35" and "drive15" not in e["amendment_2"]["text"]
    assert not any(x["missing"] for x in e.values())
    assert all(x["missing"] for x in extract_prereg.extract("nothing here"))


def test_prereg_tex_marks_missing_entries(v2dir):
    h = Host()
    assert "missing" in h.prereg_tex() and "TODO" in h.prereg_tex()
    (v2dir / "data").mkdir()
    (v2dir / "data" / "prereg_trail.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"id": "x", "title": "Rule", "timestamp": "t", "text": "## H\n- a_b", "missing": False},
                    {"id": "y", "title": "Gone", "missing": True},
                ]
            }
        )
    )
    tex = h.prereg_tex()
    assert r"a\_b" in tex and "## " not in tex and "entry not found" in tex


def test_confirmatory_table_pending_then_discovered(v2dir):
    h = Host()
    assert "pending" in h.tab_confirmatory() and "TODO" in h.nums["NumConfSeeds"]
    _system(v2dir, "base_bge-m3", "test", 0.60)
    for seed, v in ((16, 0.70), (17, 0.72)):
        _system(v2dir, f"fin_s{seed}", "test", v, seed=seed)
    _system(v2dir, "fin_s15", "test", 0.9)  # not a confirmatory seed: must be ignored
    assert list(h.confirmatory_systems()) == ["fin"] and sorted(h.confirmatory_systems()["fin"]) == [16, 17]
    tex = h.tab_confirmatory()
    assert "fin, seed 16" in tex and "fin, seed 17" in tex and "mean of 2 seeds" in tex
    assert h.nums["NumConfSeeds"] == "16, 17" and h.nums["NumConfArEnMean"] == "71.0"
    assert h.nums["NumConfDeltaArEn"].startswith("+")


def test_scaling_and_negative_tables_from_run_files(v2dir, monkeypatch):
    def run(stem, ar, macro, xl=None, devm=None):
        d = {"directions": {"ar->en": {"mrr@10": ar}, "en->ar": {"mrr@10": ar + 0.1}, "macro": {"mrr@10": macro}}}
        (v2dir / f"{stem}.dev.json").write_text(json.dumps(d))
        if xl is not None:
            (v2dir / f"{stem}.train.json").write_text(
                json.dumps({"data": {"by_slice": {"train_xl_ar2en": 3000, "train_xl2_ar2en": xl, "train_xl2_en2ar": xl // 2}}})
            )
        if devm is not None:
            (v2dir / f"{stem}.devM.json").write_text(json.dumps({"slices": {"dev_miracl_ar": {"mrr@10": devm}}}))

    run("p1_r0", 0.65, 0.77, xl=0)
    run("p2_s_f25", 0.667, 0.776, xl=1000)
    run("p2_s_full", 0.678, 0.778, xl=4000)
    monkeypatch.setattr(study_tables, "SCALING", {"small": [("none", "p1_r0"), ("first", "p2_s_f25"), ("full", "p2_s_full")], "base": []})
    h = Host()
    tex = h.tab_scaling()
    assert "1{,}500" in tex and "6{,}000" in tex and h.nums["NumScaleSmallFirstShare"] == "61"
    run("p1_kd1", 0.662, 0.770)
    run("m3_b_s13", 0.81, 0.853, devm=0.7914)
    run("m3_r_s13", 0.808, 0.849, devm=0.7875)
    monkeypatch.setattr(
        study_tables,
        "NEGATIVE",
        [
            ("KD", "p1_kd1", "p1_r0", "small", "no"),
            ("Replay", "m3_r_s13", "m3_b_s13", "m3", "no"),
            ("LoRA", "m3_lo_s13", "m3_b_s13", "m3", "no"),
        ],
    )
    neg = h.tab_negative()
    assert "KD & 0.0 & +1.2 & -- & no" in neg
    assert "Replay & $-$0.4 & $-$0.2 & $-$0.4 & no" in neg and "run pending" in neg
    assert h.nums["NumNegRows"] == "2"
