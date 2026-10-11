"""Tests for the generator and the audio analysis, using synthetic input (no EEG data needed).
Run from the project folder:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import analyze_audio as aa
import generate as gen
from sonify import SR

RANGES = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # identity scaling for (alpha, beta/alpha, amplitude)


def test_tempo_error_allows_half_and_double_but_not_other_ratios():
    assert aa.tempo_error(100, 100) == (0, 0)
    raw, folded = aa.tempo_error(50, 100)
    assert raw == 50 and folded == 0
    assert aa.tempo_error(200, 100)[1] == 0
    assert aa.tempo_error(66.7, 100)[1] > 25  # 2/3 is a different beat level, not an octave error


def test_detect_tempo_on_a_click_track():
    clicks = librosa.clicks(times=np.arange(0, 20, 60 / 100), sr=SR, length=20 * SR)
    assert aa.tempo_error(aa.detect_tempo(clicks), 100)[1] < 5


def test_pitch_error_is_near_zero_for_known_notes():
    melody = [(i * 0.5, 0.45, p) for i, p in enumerate([60, 64, 67, 72])]
    y = gen.render_notes([(o, d, p, 100, 0.3) for o, d, p in melody], 2.0)
    err, coverage = aa.pitch_error(y.astype(np.float32), melody)
    assert err < 0.2 and coverage == 1


def test_calm_bar_has_four_melody_notes_and_alert_bar_eight():
    calm, *_ = gen.bar_notes(0, (0.5, 0.2, 0.5), 100, RANGES)
    alert, *_ = gen.bar_notes(0, (0.5, 0.8, 0.5), 100, RANGES)
    assert (len(calm), len(alert)) == (4, 8)


def test_higher_alpha_puts_the_melody_higher():
    low, *_ = gen.bar_notes(0, (0.1, 0.2, 0.5), 100, RANGES)
    high, *_ = gen.bar_notes(0, (0.9, 0.2, 0.5), 100, RANGES)
    assert np.mean([n[2] for n in high]) > np.mean([n[2] for n in low])


def test_notes_fade_out_instead_of_ending_with_a_click():
    w = gen.tone(60, 0.5, 100, 0.3)
    assert abs(w[-1]) < 0.01 * np.abs(w).max()  # an abrupt end was being counted as an extra note onset


def test_alpha_and_busy_bars_can_be_read_back_from_the_audio(tmp_path):
    alpha = np.linspace(0.05, 0.95, 16)
    ratio = np.where(np.arange(16) % 3 == 0, 0.8, 0.2)  # every third bar is busy
    g = gen.generate(np.column_stack([alpha, ratio, np.full(16, 0.5)]), RANGES)
    gen.write_piece(g, tmp_path / "piece")
    r = aa.analyze(tmp_path / "piece")
    assert r["alpha_rho"] > 0.9
    assert r["busy_written"] == 6 and r["busy_found"] == 6 and r["busy_right"] == 16
    assert r["tempo_err_folded_pct"] < 5


def test_generated_piece_has_one_fixed_tempo_from_mean_beta_alpha():
    feats = np.array([[0.5, 0.2, 0.5]] * 3 + [[0.5, 0.8, 0.5]])  # mean beta/alpha = 0.35
    g = gen.generate(feats, RANGES)
    assert abs(g["bpm"] - (60 + 80 * 0.35)) < 1e-6
    assert abs(g["seconds"] - 4 * 4 * 60 / g["bpm"]) < 1e-6
