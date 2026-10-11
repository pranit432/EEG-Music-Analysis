"""Tests for the Streamlit pages. Run from the project folder:  python -m pytest tests -q

Each page is run on its own with AppTest. Note: AppTest cannot re-run after changing a single-select
pills widget (it iterates the selected string letter by letter), so widget interactions that go through
pills are checked in the browser instead; here the starting subject is set through session state.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # AppTest does not add the script folder


def run_page(page, **state):
    at = AppTest.from_string(f"import views\nviews.{page}()", default_timeout=180)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_explore_defaults_to_s09_and_shows_alpha_rise():
    at = run_page("explore_page")
    assert not at.exception
    assert at.selectbox(key="exp_sub").value == "S09"
    assert float(at.metric[0].value) > 0.3  # S09 alpha during music is about 0.5
    assert at.metric[0].delta.startswith("+")  # higher than baseline


def test_explore_for_s01_shows_almost_no_alpha_change():
    at = run_page("explore_page", focus_sub="S01")
    assert not at.exception
    assert float(at.metric[0].value) < 0.2


def test_explore_preprocessing_defaults_match_group_analysis():
    at = run_page("explore_page")
    values = {s.label: s.value for s in at.slider}
    assert values["Reject windows above (µV peak-to-peak)"] == 200
    assert values["Drop channels noisier than (× median)"] == 3.0


def test_group_page_reports_the_alpha_result():
    at = run_page("group_page")
    assert not at.exception
    m = metrics(at)
    assert m["Subjects higher in music"] == "16 of 18"
    assert float(m["Holm-corrected p"]) < 0.01


def test_listen_page_renders_baseline_slower_music_than_baseline_is_faster():
    at = run_page("listen_page")
    assert not at.exception
    tempos = [float(m.value.split()[0]) for m in at.metric]  # baseline first, then E01
    assert len(tempos) == 2
    assert tempos[0] > 120 > 90 > tempos[1]  # S09: fast baseline, slower music


def test_generate_page_checks_the_audio_against_the_eeg():
    at = run_page("generate_page")
    assert not at.exception
    m = metrics(at)  # defaults to S09 E01
    assert 70 < float(m["Tempo from EEG"].split()[0]) < 85
    detected = float(m["Tempo found in the audio"].split()[0])
    assert abs(detected - 76) < 5  # S09 E01 is detected at its written tempo
    assert float(m["Alpha read back from pitch"]) > 0.7
    assert m["Busy bars spotted"] == "1 of 1"  # S09 E01 has a single busy bar


def test_about_page_counts_the_data():
    at = run_page("about_page")
    assert not at.exception
    m = metrics(at)
    assert m["Subjects"] == "18" and m["Recordings"] == "162"  # 18 baselines + 144 music trials, no duplicate part-files
