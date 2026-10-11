"""Check generated music by analysing the audio (round trip: EEG -> music -> what an analyser finds in it).

  tempo:    librosa tempo estimate on the mix, compared with the BPM the EEG produced. Estimators often pick half
            or double the tempo, so we report the raw error and the error after allowing x0.5 / x2.
  alpha:    the lowest melody note of each bar, as read by a pitch tracker (pYIN) on the melody stem, correlated
            with that bar's alpha. Shows how much of the alpha signal can be recovered from the audio.
  busy bars: bars where an onset detector counts six or more melody notes (the music writes four or eight),
            compared with which bars were written as busy.
  pitch:    pYIN against the notes that were written. Clean synthetic tones are easy to track, so this is ~0 and
            only confirms the synthesiser; it is kept as a sanity check, not as a result.

Usage: python src/analyze_audio.py [subject] [recordings...]    (run generate.py first)
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from librosa.feature.rhythm import tempo as estimate_tempo  # explicit: librosa.feature.rhythm isn't always loaded lazily
from scipy.stats import spearmanr

from sonify import ROOT, SR

HOP = 512
BUSY_AT = 6  # melody notes per bar at or above which a bar counts as busy (written bars have 4 or 8)


def detect_tempo(y, sr=SR, max_bpm=200):
    """Tempo from the audio's rhythm pattern, searching 40-200 BPM. (librosa's beat tracker was unreliable on these
    smooth synthetic tones: it often fell back on its default guess or locked on to the bar instead of the beat.)"""
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    return float(estimate_tempo(onset_envelope=oenv, sr=sr, hop_length=HOP, max_tempo=max_bpm)[0])


def tempo_error(detected, intended):
    """-> (raw % error, % error after allowing half/double tempo)."""
    raw = 100 * abs(detected - intended) / intended
    folded = min(100 * abs(detected * k - intended) / intended for k in (0.5, 1, 2))
    return raw, folded


def track_pitch(y, sr=SR):
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C3"), fmax=librosa.note_to_hz("C7"), sr=sr)
    return f0, voiced, librosa.times_like(f0, sr=sr)


def note_pitches(track, melody):
    """Pitch (MIDI) the tracker reads in the middle of each written note; NaN where it found none."""
    f0, voiced, times = track
    out = []
    for onset, dur, _ in melody:
        sel = (times >= onset + 0.25 * dur) & (times <= onset + 0.75 * dur) & voiced
        out.append(float(librosa.hz_to_midi(np.median(f0[sel]))) if sel.any() else np.nan)
    return np.array(out)


def pitch_error(y, melody, sr=SR, track=None):
    """Median |detected - written| in semitones over the middle of each melody note, and the share of notes found."""
    det = note_pitches(track or track_pitch(y, sr), melody)
    err = np.abs(det - np.array([m[2] for m in melody]))
    return float(np.nanmedian(err)), float(np.mean(~np.isnan(err)))


def bar_of(onsets, bar_s, bars):
    return np.minimum((np.asarray(onsets) / bar_s).astype(int), bars - 1)


def recover(y, meta, track):
    """What can be read back out of the melody stem: per-bar pitch (vs alpha) and which bars were busy."""
    melody, bar_s, alpha = meta["melody"], meta["bar_s"], np.array(meta["alpha"])
    bars = len(alpha)
    bar = bar_of([m[0] for m in melody], bar_s, bars)
    det = note_pitches(track, melody)
    low = np.array([np.nanmin(det[bar == b]) if (bar == b).any() and not np.all(np.isnan(det[bar == b])) else np.nan
                    for b in range(bars)])
    ok = ~np.isnan(low)
    rho = float(spearmanr(alpha[ok], low[ok]).statistic) if ok.sum() > 2 and len(set(low[ok])) > 1 else float("nan")
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time")
    found = np.bincount(bar_of(onsets, bar_s, bars), minlength=bars)
    written = np.bincount(bar, minlength=bars)
    return {"alpha_rho": rho, "bar_alpha": alpha.tolist(), "bar_pitch": low.tolist(),
            "busy_written": int((written >= BUSY_AT).sum()), "busy_right": int(((found >= BUSY_AT) == (written >= BUSY_AT)).sum()),
            "busy_found": int(((found >= BUSY_AT) & (written >= BUSY_AT)).sum()),
            "bars": bars}


def analyze(stem):
    """stem: path without extension, as written by generate.write_piece."""
    meta = json.loads(Path(f"{stem}.json").read_text())
    y, _ = librosa.load(f"{stem}.wav", sr=SR)
    ym, _ = librosa.load(f"{stem}_melody.wav", sr=SR)
    detected = detect_tempo(y)
    raw, folded = tempo_error(detected, meta["bpm"])
    track = track_pitch(ym)
    pitch_err, coverage = pitch_error(ym, meta["melody"], track=track)
    return {"intended_bpm": meta["bpm"], "detected_bpm": detected, "tempo_err_pct": raw,
            "tempo_err_folded_pct": folded, "pitch_err_semitones": pitch_err, "pitch_coverage": coverage,
            **recover(ym, meta, track)}


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "S01"
    wanted = sys.argv[2:] or ["BS", "E01"]
    for rec in wanted:
        r = analyze(ROOT / "results" / f"generate_{subject}_{rec}")
        print(f"{rec:>4}: tempo intended {r['intended_bpm']:6.1f}, detected {r['detected_bpm']:6.1f} "
              f"(error {r['tempo_err_pct']:4.1f}%, {r['tempo_err_folded_pct']:4.1f}% allowing x0.5/x2) | "
              f"alpha recovered rho {r['alpha_rho']:.2f} | busy bars right {r['busy_right']}/{r['bars']} "
              f"({r['busy_written']} written busy) | pitch sanity {r['pitch_err_semitones']:.2f} st")


if __name__ == "__main__":
    main()
