"""Load one DSI-24 EDF recording and print its basic properties.

Usage: python src/load_edf.py [path/to/file.edf]
"""
import sys
from pathlib import Path

import mne

DEFAULT = Path(__file__).parent.parent / "data/tempo/music1/S01/S01E01_filtered.edf"


def load(path):
    return mne.io.read_raw_edf(path, preload=True, verbose="error")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    raw = load(path)
    print(f"File:      {path.name}")
    print(f"Channels:  {len(raw.ch_names)}")
    print(f"           {raw.ch_names}")
    print(f"Sample Hz: {raw.info['sfreq']}")
    print(f"Duration:  {raw.times[-1]:.1f} s")
    print(f"Filter:    {raw.info['highpass']}-{raw.info['lowpass']} Hz")
