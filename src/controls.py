"""Controls for the baseline-vs-music alpha result.

A. Drift: does alpha just climb with time in the session? (trial-order trend, baseline vs first trial)
B. Eyes closed: do blinks (Fp1/Fp2 amplitude) drop during music, and is the alpha rise posterior?

Usage: python src/controls.py
Writes results/per_recording.csv and results/controls_stats.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from band_power import BANDS, EDGE_S, REJECT_UV, WIN_S, band_powers, clean
from group_analysis import MIN_MUSIC_S, MIN_WINDOWS, TRIAL, music_span
from load_edf import load

ROOT = Path(__file__).parent.parent
FRONTAL = ["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"]
POSTERIOR = ["P3", "P4", "T5", "T6"]  # O1/O2 are bad in most recordings
BLINK_UV = 100  # Fp window peak-to-peak above this counts as containing a blink


def blink_stats(raw):
    """Fp1/Fp2 window peak-to-peak on ALL windows (the 200 uV rejection would hide blinks)."""
    fp = [c for c in ("Fp1", "Fp2") if c in raw.ch_names]
    if not fp:
        return np.nan, np.nan
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=fp, tmin=EDGE_S, tmax=raw.times[-1] - EDGE_S) * 1e6
    w = int(WIN_S * sf)
    p2p = np.array([np.ptp(data[:, i * w:(i + 1) * w], axis=1).mean() for i in range(data.shape[1] // w)])
    return np.median(p2p), (p2p > BLINK_UV).mean()


def region_alpha(bp, names, region):
    idx = [names.index(c) for c in region if c in names]
    if not idx:
        return np.nan
    allp = sum(bp.values())
    return (bp["alpha"][idx] / allp[idx]).mean()


def measure(path, music):
    raw = load(path)
    if music:
        span = music_span(raw)
        if span is None:
            return None
        raw.crop(*span)
    raw, _ = clean(raw)
    bp, kept, _ = band_powers(raw)
    if bp is None or kept < MIN_WINDOWS:
        return None
    allp = sum(bp.values())
    names = raw.ch_names
    fp_p2p, blink_frac = blink_stats(raw)
    return {"alpha_rel": (bp["alpha"] / allp).mean(), "log_beta_alpha": np.log(bp["beta"].mean() / bp["alpha"].mean()),
            "alpha_frontal": region_alpha(bp, names, FRONTAL), "alpha_posterior": region_alpha(bp, names, POSTERIOR),
            "fp_p2p_uv": fp_p2p, "blink_frac": blink_frac}


def collect():
    rows = []
    for sdir in sorted((ROOT / "data/tempo/music1").glob("S*")):
        sub = sdir.name
        base = measure(sdir / f"{sub}BS_filtered.edf", music=False)
        if base:
            rows.append({"subject": sub, "trial": 0, **base})
        for f in sorted(sdir.glob("*_filtered.edf")):
            if "BS" in f.name or not TRIAL.match(f.name):
                continue
            m = measure(f, music=True)
            if m:
                rows.append({"subject": sub, "trial": int(re.search(r"E(\d+)", f.name).group(1)), **m})
    return pd.DataFrame(rows)


def paired(a, b, label, out):
    d = (b - a).dropna()
    p = stats.wilcoxon(d).pvalue
    out.append({"test": label, "n": len(d), "a_mean": a[d.index].mean(), "b_mean": b[d.index].mean(), "wilcoxon_p": p,
                "n_b_higher": int((d > 0).sum())})


def main():
    df = collect()
    df.to_csv(ROOT / "results/per_recording.csv", index=False)
    music = df[df.trial > 0]
    base = df[df.trial == 0].set_index("subject")
    mean_music = music.groupby("subject").mean(numeric_only=True)
    out = []

    # A. drift ------------------------------------------------------------------------------
    for m in ["alpha_rel", "log_beta_alpha"]:
        rho = music.groupby("subject").apply(
            lambda g: stats.spearmanr(g.trial, g[m])[0] if len(g) >= 5 else np.nan, include_groups=False).dropna()
        out.append({"test": f"A. {m}: trend across music trials (per-subject Spearman rho vs 0)", "n": len(rho),
                    "a_mean": rho.mean(), "b_mean": np.nan, "wilcoxon_p": stats.wilcoxon(rho).pvalue,
                    "n_b_higher": int((rho > 0).sum())})
        first = music[music.trial == 1].set_index("subject")[m]
        last = music.sort_values("trial").groupby("subject")[m].last()
        paired(base[m], first, f"A. {m}: baseline vs FIRST music trial (E01)", out)
        paired(first, last.reindex(first.index), f"A. {m}: first vs LAST music trial", out)

    # B. eyes closed -------------------------------------------------------------------------
    for m in ["fp_p2p_uv", "blink_frac"]:
        paired(base[m], mean_music[m], f"B. {m}: baseline vs music", out)
    paired(base["alpha_frontal"], mean_music["alpha_frontal"], "B. alpha_rel frontal: baseline vs music", out)
    paired(base["alpha_posterior"], mean_music["alpha_posterior"], "B. alpha_rel posterior: baseline vs music", out)
    ratio_b = base["alpha_posterior"] / base["alpha_frontal"]
    ratio_m = mean_music["alpha_posterior"] / mean_music["alpha_frontal"]
    paired(ratio_b, ratio_m, "B. posterior/frontal alpha ratio: baseline vs music", out)

    st = pd.DataFrame(out)
    st.to_csv(ROOT / "results/controls_stats.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
