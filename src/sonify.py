"""Turn EEG band power into music (sonification).

Each clean 2 s EEG window becomes one bar of four notes:
  alpha power (relative)  -> pitch   (higher alpha = higher notes, C major pentatonic)
  beta/alpha ratio        -> tempo   (higher ratio = faster, 60-140 BPM)
  signal amplitude        -> loudness
Ranges are scaled across ALL of a subject's recordings (not per file), so a calm baseline and an
energetic trial sound different from each other.

Usage: python src/sonify.py [subject] [recordings...]     e.g. python src/sonify.py S01 BS E01 E05
Writes results/sonify_<subject>_<recording>.mid and .wav
"""
import sys
from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

from band_power import BAD_CH_MAD, EDGE_S, REJECT_UV, WIN_S, clean
from group_analysis import music_span
from load_edf import load

ROOT = Path(__file__).parent.parent
SCALE = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81]  # C major pentatonic, two octaves
PATTERN = [0, 2, 4, 2]  # scale-step offsets for the four notes in a bar
BPM_RANGE = (60, 140)
SR = 22050


def window_features(path, reject_uv=REJECT_UV, bad_ch_mad=BAD_CH_MAD):
    """Per clean window: (relative alpha, log beta/alpha, amplitude in uV)."""
    raw = load(path)
    span = music_span(raw)  # None for baseline recordings
    if span:
        raw.crop(*span)
    raw, _ = clean(raw, bad_ch_mad)
    sf = raw.info["sfreq"]
    data = raw.get_data(tmin=EDGE_S, tmax=raw.times[-1] - EDGE_S)
    w = int(WIN_S * sf)
    out = []
    for i in range(data.shape[1] // w):
        x = data[:, i * w:(i + 1) * w]
        if np.ptp(x, axis=1).max() * 1e6 >= reject_uv:
            continue
        f, p = welch(x, fs=sf, nperseg=int(sf))
        p = p.mean(axis=0)
        band = lambda lo, hi: p[(f >= lo) & (f < hi)].sum()
        alpha, beta = band(8, 13), band(13, 30)
        out.append((alpha / band(1, 45), np.log(beta / alpha), x.std() * 1e6))
    return np.array(out)


def to_unit(values, lo, hi):
    return np.clip((values - lo) / (hi - lo), 0, 1)


def make_notes(feats, ranges):
    """-> list of (midi_pitch, velocity, bpm, bar_index, beat_in_bar)."""
    notes = []
    for bar, (alpha, ratio, amp) in enumerate(feats):
        a, r, v = (to_unit(alpha, *ranges[0]), to_unit(ratio, *ranges[1]), to_unit(amp, *ranges[2]))
        bpm = BPM_RANGE[0] + (BPM_RANGE[1] - BPM_RANGE[0]) * r
        base = int(round(a * (len(SCALE) - 1 - max(PATTERN))))
        for beat, off in enumerate(PATTERN):
            notes.append((SCALE[base + off], int(50 + 60 * v), bpm, bar, beat))
    return notes


def write_midi(notes, path):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    for pitch, vel, bpm, bar, beat in notes:
        if beat == 0:
            tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
        tr.append(mido.Message("note_on", note=pitch, velocity=vel, time=0))
        tr.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    mid.save(path)


def write_wav(notes, path):
    chunks = []
    for pitch, vel, bpm, bar, beat in notes:
        dur = 60 / bpm
        t = np.arange(int(SR * dur)) / SR
        f = 440 * 2 ** ((pitch - 69) / 12)
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
        chunks.append(tone * np.exp(-3 * t / dur) * (vel / 127) * 0.4)
    audio = np.concatenate(chunks)
    wavfile.write(path, SR, (audio / max(1, np.abs(audio).max()) * 32767).astype(np.int16))
    return len(audio) / SR


def subject_features(subject):
    """Window features for every recording of a subject, plus subject-wide normalisation ranges."""
    sdir = ROOT / "data/tempo/music1" / subject
    files = sorted(f for f in sdir.glob(f"{subject}*_filtered.edf") if "_0001" not in f.name)
    feats = {f.name[len(subject):].split("_")[0]: window_features(f) for f in files}
    feats = {k: v for k, v in feats.items() if len(v)}
    allf = np.vstack(list(feats.values()))
    return feats, [tuple(np.percentile(allf[:, i], [5, 95])) for i in range(3)]


def render(subject, rec, feats, ranges):
    """Write results/sonify_<subject>_<rec>.mid/.wav and return a summary dict."""
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    notes = make_notes(feats[rec], ranges)
    stem = out / f"sonify_{subject}_{rec}"
    write_midi(notes, f"{stem}.mid")
    secs = write_wav(notes, f"{stem}.wav")
    return {"recording": rec, "bars": len(feats[rec]), "seconds": secs, "mean_bpm": np.mean([n[2] for n in notes]),
            "mean_pitch": np.mean([n[0] for n in notes]), "mean_velocity": np.mean([n[1] for n in notes]),
            "wav": f"{stem}.wav", "mid": f"{stem}.mid"}


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "S01"
    wanted = sys.argv[2:] or ["BS", "E01"]
    feats, ranges = subject_features(subject)
    for rec in wanted:
        r = render(subject, rec, feats, ranges)
        print(f"{rec:>4}: {r['bars']} bars, {r['seconds']:5.1f} s, mean BPM {r['mean_bpm']:5.1f}, "
              f"mean pitch {r['mean_pitch']:5.1f}, mean velocity {r['mean_velocity']:5.1f}"
              f"  -> {Path(r['wav']).name}/.mid")


if __name__ == "__main__":
    main()
