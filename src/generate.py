"""EEG-driven music generation: a fuller arrangement than sonify.py (melody + bass + pad).

One recording -> one piece with a FIXED tempo, so the tempo can be checked from the audio later.
  mean beta/alpha (whole recording) -> tempo        (60-140 BPM, scaled per subject like sonify.py)
  each 2 s window = one 4/4 bar:
    relative alpha  -> melody register (C major pentatonic; higher alpha = higher notes)
    beta/alpha      -> note density (quarters when calm, eighths when alert) and brightness
    amplitude       -> loudness
  chords cycle I-vi-IV-V in C major (an arbitrary choice, not derived from the EEG).

Usage: python src/generate.py [subject] [recordings...]     e.g. python src/generate.py S09 BS E01
Writes results/generate_<subject>_<rec>.wav (mix), _melody.wav (melody alone) and .json (the intended notes).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from sonify import BPM_RANGE, PATTERN, ROOT, SCALE, SR, subject_features, to_unit

CHORDS = [(48, (0, 4, 7)), (45, (0, 3, 7)), (53, (0, 4, 7)), (55, (0, 4, 7))]  # C, Am, F, G: (bass root, triad)
DENSE_AT = 0.5  # beta/alpha (0-1) above which a bar uses eighth notes
EIGHTH_PATTERN = PATTERN + [0, 2, 4, 5]


def bar_notes(bar, feats, bpm, ranges):
    """-> (melody, bass, pad) note lists for one bar; each note is (onset_s, dur_s, midi, velocity, brightness)."""
    alpha, ratio, amp = feats
    a, r, v = to_unit(alpha, *ranges[0]), to_unit(ratio, *ranges[1]), to_unit(amp, *ranges[2])
    beat = 60 / bpm
    t0 = bar * 4 * beat
    vel = int(50 + 60 * v)
    bright = 0.1 + 0.6 * r
    pattern = EIGHTH_PATTERN if r > DENSE_AT else PATTERN
    step = 4 * beat / len(pattern)
    base = int(round(a * (len(SCALE) - 1 - max(pattern))))
    melody = [(t0 + i * step, step * 0.9, SCALE[base + off], vel, bright) for i, off in enumerate(pattern)]
    root, triad = CHORDS[bar % len(CHORDS)]
    bass = [(t0 + k * 2 * beat, 2 * beat * 0.95, root, vel, 0.1) for k in range(2)]
    pad = [(t0, 4 * beat, root + 12 + iv, int(vel * 0.6), 0.05) for iv in triad]
    return melody, bass, pad


def tone(pitch, dur, vel, bright):
    t = np.arange(int(SR * dur)) / SR
    f = 440 * 2 ** ((pitch - 69) / 12)
    wave = np.sin(2 * np.pi * f * t) + bright * np.sin(4 * np.pi * f * t) + bright * 0.5 * np.sin(6 * np.pi * f * t)
    env = np.minimum(1, t / 0.01) * np.exp(-2.5 * t / dur)  # 10 ms attack, then decay
    env *= np.minimum(1, (dur - t) / 0.03)  # 30 ms fade-out, so the note doesn't end with a click
    return wave * env * (vel / 127)


def render_notes(notes, total_s):
    audio = np.zeros(int(SR * total_s) + SR)
    for onset, dur, pitch, vel, bright in notes:
        i = int(SR * onset)
        w = tone(pitch, dur, vel, bright)
        audio[i:i + len(w)] += w
    return audio


def generate(feats, ranges):
    """EEG window features of one recording -> dict with mix, melody stem, tempo and the intended melody."""
    ratio = to_unit(feats[:, 1], *ranges[1])
    bpm = BPM_RANGE[0] + (BPM_RANGE[1] - BPM_RANGE[0]) * ratio.mean()
    parts = [bar_notes(b, f, bpm, ranges) for b, f in enumerate(feats)]
    melody, bass, pad = ([n for p in parts for n in p[k]] for k in range(3))
    total = len(feats) * 4 * 60 / bpm
    stems = [render_notes(x, total) for x in (melody, bass, pad)]
    mix = 0.5 * stems[0] + 0.35 * stems[1] + 0.25 * stems[2]
    return {"bpm": float(bpm), "bar_s": 4 * 60 / float(bpm), "seconds": total, "bars": len(feats), "mix": mix,
            "melody_audio": stems[0], "alpha": [float(a) for a in feats[:, 0]],
            "melody": [(float(o), float(d), int(p)) for o, d, p, _, _ in melody]}


def save_wav(audio, path):
    wavfile.write(path, SR, (audio / max(1, np.abs(audio).max()) * 32767 * 0.9).astype(np.int16))


def write_piece(g, stem):
    """Write <stem>.wav (mix), <stem>_melody.wav and <stem>.json (what the EEG asked for, for analyze_audio)."""
    save_wav(g["mix"], f"{stem}.wav")
    save_wav(g["melody_audio"], f"{stem}_melody.wav")
    Path(f"{stem}.json").write_text(json.dumps({k: g[k] for k in ("bpm", "bar_s", "alpha", "melody")}))


def render(subject, rec, feats, ranges):
    """Write results/generate_<subject>_<rec>.* and return a summary dict."""
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    g = generate(feats[rec], ranges)
    stem = out / f"generate_{subject}_{rec}"
    write_piece(g, stem)
    return {"recording": rec, "bars": g["bars"], "seconds": g["seconds"], "bpm": g["bpm"], "stem": str(stem)}


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "S01"
    wanted = sys.argv[2:] or ["BS", "E01"]
    feats, ranges = subject_features(subject)
    for rec in wanted:
        r = render(subject, rec, feats, ranges)
        print(f"{rec:>4}: {r['bars']} bars, {r['seconds']:5.1f} s, {r['bpm']:5.1f} BPM -> {Path(r['stem']).name}.wav")


if __name__ == "__main__":
    main()
