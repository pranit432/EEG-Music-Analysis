"""Compute EEG band power per recording for one subject (baseline vs music trials).

Artifact handling:
  1. channels whose std is > BAD_CH_MAD x the median channel std are dropped
     (before the average reference, so they cannot contaminate the others)
  2. 2 s windows with peak-to-peak > REJECT_UV on any remaining channel are rejected

Usage: python src/band_power.py [S01]
Writes results/<subject>_band_power.csv and prints a table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from load_edf import load

ROOT = Path(__file__).parent.parent
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
NON_EEG = {"Trigger", "CM", "A1", "A2", "X1", "X2", "X3"}
EDGE_S = 2  # seconds trimmed from each end to avoid filter edge effects
WIN_S = 2
BAD_CH_MAD = 3  # channel is bad if std > 3 x median channel std
REJECT_UV = 200  # window is bad if peak-to-peak exceeds this on any channel


def clean(raw, bad_ch_mad=BAD_CH_MAD):
    """Keep scalp EEG, strip 'EEG X-Pz' to 'X', drop noisy channels, average-reference.
    Returns (raw, names of dropped channels)."""
    raw.rename_channels({c: c.replace("EEG ", "").replace("-Pz", "") for c in raw.ch_names})
    raw.pick([c for c in raw.ch_names if c not in NON_EEG])
    bad = []
    for _ in range(3):  # detect on an average-referenced signal; repeat as bad channels leave the reference
        raw.set_eeg_reference("average", verbose="error")
        std = raw.get_data().std(axis=1)
        new_bad = [c for c, s in zip(raw.ch_names, std) if s > bad_ch_mad * np.median(std)]
        if not new_bad:
            break
        raw.drop_channels(new_bad)
        bad += new_bad
    return raw, bad


def band_powers(raw, reject_uv=REJECT_UV):
    """Welch PSD per clean window, integrated per band -> ({band: power per channel}, kept, total)."""
    sf = raw.info["sfreq"]
    data = raw.get_data(tmin=EDGE_S, tmax=raw.times[-1] - EDGE_S)
    w = int(WIN_S * sf)
    wins = [data[:, i * w:(i + 1) * w] for i in range(data.shape[1] // w)]
    good = [x for x in wins if np.ptp(x, axis=1).max() * 1e6 < reject_uv]
    if not good:
        return None, 0, len(wins)
    freqs, psd = welch(np.stack(good), fs=sf, nperseg=w)  # (windows, channels, freqs)
    psd = psd.mean(axis=0)
    df = freqs[1] - freqs[0]
    bp = {b: psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df for b, (lo, hi) in BANDS.items()}
    return bp, len(good), len(wins)


def summarise(raw, bad, label, reject_uv=REJECT_UV):
    bp, kept, total = band_powers(raw, reject_uv)
    row = {"recording": label, "bad_ch": ",".join(bad) or "-", "win_kept": f"{kept}/{total}"}
    if bp is None:
        return row
    allp = sum(bp.values())
    for b, p in bp.items():
        row[f"{b}_rel"] = (p / allp).mean()
    mean = {b: p.mean() for b, p in bp.items()}
    row["beta_alpha"] = mean["beta"] / mean["alpha"]
    row["theta_beta"] = mean["theta"] / mean["beta"]
    # frontal alpha asymmetry: ln(right) - ln(left); NaN if F3/F4 was dropped as bad
    n = raw.ch_names
    row["faa_F4_F3"] = np.log(bp["alpha"][n.index("F4")]) - np.log(bp["alpha"][n.index("F3")]) if {"F3", "F4"} <= set(n) else np.nan
    return row


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "S01"
    files = sorted((ROOT / "data/tempo/music1" / subject).glob("*_filtered.edf"))
    rows = []
    for f in files:
        raw, bad = clean(load(f))
        rows.append(summarise(raw, bad, f.stem.replace("_filtered", "")))
    df = pd.DataFrame(rows).set_index("recording")
    out = ROOT / "results" / f"{subject}_band_power.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out)
    print(df.round(3).to_string())
    print(f"\nSaved {out}")
